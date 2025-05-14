#!/bin/bash
#SBATCH -J cyclegan_profile
#SBATCH --output=cyclegan_default_18.out
#SBATCH -t 15:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH --exclusive

#source ~/.bashrc
#micromamba activate CYCLEGAN
. ~/.cycle_gan/bin/activate

srun python profile_model.py --dataroot /scratch/zt1/project/cmsc828/shared/cgan_perf/datasets/maps \
                             --name maps_cyclegan \
                             --model cycle_gan \
                             --n_epochs 1 \
                             --n_epochs_decay 0 \
                             --batch_size 18 \
                             --no_html

