# CMSC828G Final Project

+ Original [README](original_README.md)
+ Dataset path (shell space): `/afs/shell.umd.edu/project/cmsc828/user/vla/cyclegan/datasets/maps`
+ Dataset path (scratch space): `/scratch/zt1/project/cmsc828/user/vla/datasets/maps`

Note: Zaratan `shell` space is not accesible by compute nodes so copy dataset over to `scratch` space before launching a job or make sure shared scratch space already exists 

<!-- # Intel Extension for Pytorch for Kineto Profiling Support
+ See [installation](https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.1.10%2Bxpu&os=linux%2Fwsl2&package=pip) -->


## profile.py
Adapted `train.py` code to profile CycleGAN

Usage:
```
python profile_model.py --dataroot /scratch/zt1/project/cmsc828/user/vla/datasets/maps \
                        --name maps_cyclegan \
                        --model cycle_gan \
                        --n_epochs 1 \
                        --n_epochs_decay 0
```

Make sure to set `--dataroot` to directory in `scratch` space. Total number of epochs is `--n_epochs` + `--n_epochs_decay`
