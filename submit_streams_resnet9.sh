#!/bin/bash
#SBATCH -J cyclegan_dist_test
#SBATCH --output=cyclegan_16_resnet_9blocks.out
#SBATCH -t 15:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH --exclusive

. ~/.cycle_gan/bin/activate

srun python profile_streams.py --batch_size 16 --netG resnet_9blocks
