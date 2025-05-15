# Cycling Faster Through CycleGAN

+ Original [README](original_README.md)
+ Dataset path (scratch space): `/scratch/zt1/project/cmsc828/shared/cgan_perf/datasets/maps`

Note: Zaratan `shell` space is not accesible by compute nodes

## Final Report PDF
[Cycling_Faster_Through_CycleGAN.pdf](Cycling_Faster_Through_CycleGAN.pdf)

## profile.py
Profling script for baseline setup. Adapted `train.py` code to profile CycleGAN

Usage:
```
python profile_model.py --dataroot /scratch/zt1/project/cmsc828/user/vla/datasets/maps \
                        --name maps_cyclegan \
                        --model cycle_gan \
                        --n_epochs 1 \
                        --n_epochs_decay 0
```

Make sure to set `--dataroot` to directory in `scratch` space. Total number of epochs is `--n_epochs` + `--n_epochs_decay`

## profile_streams.py
Profiling script for parallel forward pass implementation via CUDA streams

Usage:
```
python profile_streams.py --batch_size 16 \
                          --netG resnet_9blocks
```

## Anaylsis Notebooks using Holistic Trace Analysis (HTA)
Generates a bunch of plotly plots, outputs cleared to reduce notebook file sizes
+ [profiling_analysis_default.ipynb](profiling_analysis_default.ipynb)
+ [profiling_analysis_streams.ipynb](profiling_analysis_streams.ipynb)

## HTA Plots
+ See [plots/](plots/)

## Various Submit Scripts on Zaratan
+ Profile baseline setup, ResNet9 generators: [submit_baseline_resnet9.sh](submit_baseline_resnet9.sh)
+ Profile baseline setup, U-Net generators: [submit_baseline_unet.sh](submit_baseline_unet.sh)
+ Profile parallel CUDA streams forward pass, ResNet9 generators: [submit_streams_resnet9.sh](submit_streams_resnet9.sh)
+ Profile parallel CUDA streams forward pass, U-Net generators: [submit_streams_unet.sh](submit_streams_unet.sh)
