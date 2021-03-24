import torch
import torch.nn as nn
from models.vqvae import VQVAE
import copy


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


OUT_DIM = {2: 39, 4: 35, 6: 31}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = torch.tanh(h_norm)
        self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class LambdaModule(nn.Module):
    def __init__(self, fn):
        # should contain no weights to be safe
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class PixelVQEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters=32, share_embeddings=True):
        # global OUT_DIM
        if obs_shape[1] == 20:
            # OUT_DIM = {num_layers: 5}
            out_dim = 5
        elif obs_shape[1] == 80:
            out_dim = 20
        else:
            raise NotImplementedError

        assert len(obs_shape) == 3
        super().__init__()

        h_dim = num_filters
        n_res_layers = 2
        res_h_dim = num_filters
        embedding_dim = feature_dim
        n_embeddings = 32  # FIXME
        beta = 1  # FIXME

        self.vqvae = VQVAE(
            channel_dim=obs_shape[0],
            h_dim=h_dim, res_h_dim=res_h_dim,
            n_res_layers=n_res_layers,
            n_embeddings=n_embeddings,
            embedding_dim=embedding_dim,
            beta=beta
        )
        del self.vqvae.decoder

        # copied form PixelEncoder
        self.feature_dim = feature_dim
        # self.num_layers = num_layers

        self.share_embeddings = share_embeddings
        # out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        if share_embeddings:
            self.fc = nn.Sequential(
                nn.Linear(embedding_dim*out_dim*out_dim, feature_dim),
            )
        else:
            self.fc = nn.Sequential(
                copy.deepcopy(self.vqvae.vector_quantization.embedding),
                LambdaModule(lambda x: x.flatten(start_dim=1)),
                nn.Linear(embedding_dim * out_dim * out_dim, feature_dim),
            )
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def copy_conv_weights_from(self, source):
        self.vqvae = source.vqvae

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        z_e = self.vqvae.encoder(obs)
        z_e = self.vqvae.pre_quantization_conv(z_e)
        vq_out = self.vqvae.vector_quantization(z_e)
        embedding_loss, z_q, perplexity, _, min_encoding_indices = vq_out

        self.outputs['z_q'] = z_q
        self.outputs['embedding_loss'] = embedding_loss
        self.outputs['perplexity'] = perplexity

        if self.share_embeddings:
            return z_q.flatten(start_dim=1)

        return min_encoding_indices

    def forward(self, obs, detach=False):
        return PixelEncoder.forward(self, obs=obs, detach=detach)

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2 and k != 'vq_enc':
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        # for i in range(self.num_layers):
        #     L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        # L.log_param('train_encoder/fc', self.fc, step)
        # L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder,
                       'pixel_vq': PixelVQEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )


if __name__ == "__main__":
    # obs_shape = (3, 20, 20)
    # m1 = PixelVQEncoder(obs_shape=obs_shape, feature_dim=12, num_layers=None, num_filters=32)
    # m2 = PixelVQEncoder(obs_shape=obs_shape, feature_dim=12, num_layers=None, num_filters=32)
    #
    # obs = torch.ones((4, *obs_shape))
    #
    # def test():
    #     m1.zero_grad()
    #     m2.zero_grad()
    #
    #     vq_out_1 = m1.vqvae(obs)
    #     vq_out_2 = m2.vqvae(obs)
    #
    #     vq_out_1[0].backward()
    #     vq_out_2[0].backward()
    #     print(m1.vqvae.encoder.conv_stack[0].weight.grad.mean())
    #     print(m2.vqvae.encoder.conv_stack[0].weight.grad.mean())
    #
    # test()
    # test()
    # m1.copy_conv_weights_from(m2)
    # test()

    m1 = nn.Linear(2, 3)
    m2 = copy.deepcopy(m1)
    # m1.weight.data *= 2

    print(m1.weight, m2.weight)