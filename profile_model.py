"""
Adapted train.py to profile model

"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
# from util.visualizer import Visualizer
import torch
import os

# def trace_handler(p):
#     sort_by_keyword = "self_" + 'cuda' + "_time_total"
#     output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
#     print(output)
#     p.export_chrome_trace("./traces/trace_" + str(p.step_num) + ".json")

if __name__ == '__main__':

    os.makedirs('./traces', exist_ok=True)

    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations


   # Initialize the profiler context with record_shapes, profile_memory,
   # and with_stack set to True.
   # NOTE: adapted from: https://pytorch.org/blog/understanding-gpu-memory-1/ and https://hta.readthedocs.io/en/latest/source/intro/trace_collection.html 
    

    output_dir = '/scratch/zt1/project/cmsc828/user/vla/single_gpu_traces'
    os.makedirs(output_dir, exist_ok=True)
    schedule = torch.profiler.schedule(wait=5, warmup=5, active=20, repeat=1)
    trace_handler = torch.profiler.tensorboard_trace_handler(dir_name = output_dir, use_gzip=False)


    # libkineto integration? https://github.com/pytorch/kineto/issues/973
    # experimental_config = torch.profiler._ExperimentalConfig(
    #                         profiler_metrics=[
    #                             "kineto__tensor_core_insts",
    #                             "dram__bytes_read.sum",
    #                             "dram__bytes_write.sum"],
    #                         profiler_measure_per_kernel=False),
    
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

        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
            model.update_learning_rate()    # update learning rates in the beginning of every epoch.
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                # if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                #     save_result = total_iters % opt.update_html_freq == 0
                #     model.compute_visuals()
                #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                # if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                #     losses = model.get_current_losses()
                #     t_comp = (time.time() - iter_start_time) / opt.batch_size
                #     visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                #     if opt.display_id > 0:
                #         visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                #     model.save_networks(save_suffix)

                iter_data_time = time.time()

                # signal to profiler that next step has starteds
                prof.step()

            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    
    # Construct the memory timeline HTML plot.
    prof.export_memory_timeline(f"timeline.html", device="cuda:0")

    # metrics table 
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))

