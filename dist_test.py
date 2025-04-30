from models.cycle_gan_model_parallel import CycleGAN
from data import create_dataset, find_dataset_using_name
import argparse
import torch
import os

# single cuda device for now
device = torch.device(0)

dataset_class = find_dataset_using_name('unaligned')
opt = argparse.Namespace()
opt.dataroot = '/fs/nexus-scratch/vla/datasets/maps'
opt.max_dataset_size = float('inf')
opt.input_nc = 3
opt.output_nc = 3
opt.direction = 'AtoB'
opt.phase = 'train'
opt.preprocess = 'resize_and_crop'
opt.load_size = 286
opt.crop_size = 256
opt.no_flip = False
opt.serial_batches = False

dataset = dataset_class(opt)

model = CycleGAN(device=device)

schedule = torch.profiler.schedule(wait=5, warmup=5, active=20, repeat=3)
trace_handler = torch.profiler.tensorboard_trace_handler(dir_name = './dist_traces', use_gzip=False)
os.makedirs('./dist_traces', exist_ok=True)

with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,          # NOTE: set to True for memory profiling, TODO: currently setting to True causes a runtime error: https://github.com/pytorch/pytorch/pull/150102
            on_trace_ready=trace_handler,
            # experimental_config = experimental_config
) as prof:

    # one epoch for now
    # for i in range(1):

    # TODO: adapt update learning rates? maybe unecessary

    print('Start Profiling!')
    # for i, data in enumerate(dataset):
    for i in range(len(dataset)):
        
        data = dataset[i]

        if i == 100:
            break

        # forward pass
        model(data)
        model.optimize_parameters()
        prof.step()
        

    print('Outputting memory timeline')
    # Construct the memory timeline HTML plot.
    prof.export_memory_timeline(f"dist_timeline.html", device="cuda:0")

    print('metrics table:')
    # metrics table 
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
