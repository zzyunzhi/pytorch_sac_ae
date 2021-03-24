#!/bin/bash
#SBATCH --ntasks=1

#SBATCH --partition=iris --qos=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

#SBATCH --job-name="vq"
#SBATCH --output=/iris/u/yzzhang/data/%j-%x-node-%n-task-%t.out

#SBATCH --mail-user=yzzhang@stanford.edu
#SBATCH --mail-type=ALL

###SBATCH --nodes=1

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "working directory = "$SLURM_SUBMIT_DIR
echo "SLURM_JOB_CPUS_PER_NODE="$SLURM_JOB_CPUS_PER_NODE
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES

source /iris/u/yzzhang/.bashrc
conda activate torch92
export MUJOCO_GL=osmesa
cd /iris/u/yzzhang/projects/pytorch_sac_ae

srun python /iris/u/yzzhang/projects/pytorch_sac_ae/train.py --work_dir ./data/vq/ --encoder_type pixel_vq --decoder_type pixel_vq --image_size 80

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

