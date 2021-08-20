"""
Test script for BIDeN. This script will run all test cases ( 2^N -1, N = max number of component).
We test the detailed case results of Task I using this script.
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html
import util.util as util
from itertools import combinations


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    opt.test_choice = 0
    dic = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H'}
    items = []
    all = []
    for i in range(opt.max_domain):
        items += dic[i]
    for s in range(1, (len(items) + 1)):
        for p in combinations(items, s):
            cur = ''
            for i in range(s):
                cur = cur + p[i]
            all.append(cur)

    for j in range(len(all)):
        current = all[j]
        opt.test_input = current
        if j == 0:
            model.setup(opt)  # regular setup: load and print networks; create schedulers
            model.parallelize()
        print("Current test", current)
        for i, data in enumerate(dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference

