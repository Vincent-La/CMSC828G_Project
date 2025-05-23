from models.cycle_gan_model_parallel import CycleGAN
from data import create_dataset, find_dataset_using_name
import argparse
import torch
import os


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default = 1, help = 'Batch size')
parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
args = parser.parse_args()

# single cuda device for now
device = torch.device(0)

dataset_class = find_dataset_using_name('unaligned')
opt = argparse.Namespace()
opt.dataroot = '/scratch/zt1/project/cmsc828/shared/cgan_perf/datasets/maps'
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
opt.num_threads = 4

# NOTE: experiment w/ this to increase GPU utilization with cuda streams
opt.batch_size = args.batch_size
print(f'batch_size: {opt.batch_size}')

opt.netG = args.netG
print(f'netG: {opt.netG}')

# create dataset
dataset = dataset_class(opt)
# create dataloader
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=not opt.serial_batches,
                                         num_workers=int(opt.num_threads)
)

print('Dataset loaded!')

model = CycleGAN(device=device)
schedule = torch.profiler.schedule(wait=5, warmup=5, active=20, repeat=1)

output_dir = f'/scratch/zt1/project/cmsc828/user/vla/{opt.batch_size}_{opt.netG}_dist_test'
os.makedirs(output_dir, exist_ok=True)
trace_handler = torch.profiler.tensorboard_trace_handler(dir_name = output_dir, use_gzip=False)

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

    # TODO: adapt update learning rates? maybe unecessary
    print('Start Profiling!', flush=True)
    for i, data in enumerate(dataloader):
        
        # forward pass
        model(data)
        model.optimize_parameters()
        prof.step()

    # Construct the memory timeline HTML plot.
    prof.export_memory_timeline(os.path.join(output_dir, f"dist_timeline_largebatch.html"), device="cuda:0")

    # metrics table 
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
