import time
import torch
import os
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html
from util.visualizer import Visualizer, save_images
from pytorch_fid.fid_score import calculate_fid_given_paths
"""
This one measures the FID during training. You need to create validations sets. 
We keep it here, but not recommended to use, since we have access to GT, metrics like PSNR/SSIM are better.
"""

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    val_opts = TestOptions().parse()
    val_opts.phase = 'val'
    val_opts.num_threads = 0  # test code only supports num_threads = 0
    val_opts.batch_size = 1  # test code only supports batch_size = 1
    val_opts.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    val_opts.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    val_opts.display_id = -1
    val_opts.aspect_ratio = 1.0
    opt.val_metric_freq = 1

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    val_dataset = create_dataset(val_opts)  # create a dataset given opt.dataset_mode and other options
    web_dir = os.path.join(val_opts.results_dir, val_opts.name,
                           '{}_{}'.format(val_opts.phase, val_opts.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    dataset_size = len(dataset)  # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1
    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        if epoch % opt.val_metric_freq == 0:
            print('Evaluating FID for validation set at epoch %d, iters %d, at dataset %s' % (
                epoch, total_iters, opt.name))
            model.eval()
            for i, data in enumerate(val_dataset):
                model.set_input(data)  # unpack data from data loader
                model.test()  # run inference

                visuals = model.get_current_visuals()  # get image results
                if opt.direction == 'BtoA':
                    visuals = {'fake_A': visuals['fake_A']}
                    suffix1 = 'fake_A'
                    suffix2 = 'valA'
                else:
                    visuals = {'fake_B': visuals['fake_B']}
                    suffix1 = 'fake_B'
                    suffix2 = 'valB'

                img_path = model.get_image_paths()  # get image paths
                if i % 50 == 0:  # save images to an HTML file
                    print('processing (%04d)-th image... %s' % (i, img_path))
                save_images(webpage, visuals, img_path, aspect_ratio=val_opts.aspect_ratio,
                            width=val_opts.display_winsize)
            fid_value = calculate_fid_given_paths(
                paths=(('./results/{d}/val_latest/images/'+suffix1).format(d=opt.name), ('{d}/'+suffix2).format(d=opt.dataroot)),
                batch_size=50, cuda='0', dims=2048)
            visualizer.print_current_fid(epoch, fid_value)
            visualizer.plot_current_fid(epoch, fid_value)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
