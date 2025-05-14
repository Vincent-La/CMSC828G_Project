#!/bin/bash
#SBATCH -J cyclegan_dist_test
#SBATCH --output=cyclegan_16_streams_unet.out
#SBATCH -t 10:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH --exclusive

. ~/.cycle_gan/bin/activate

srun python dist_test.py --batch_size 16 --netG unet_256
