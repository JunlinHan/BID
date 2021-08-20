import torch
from .base_model import BaseModel
from . import networks
import random
from .losses import VGGLoss
from numpy import *
import itertools
import scipy.stats as st
import cv2
import numpy as np

""" 
BIDeN model for Task III, joint shadow/reflection/watermark removal

Sample usage:
Optional visualization:
python -m visdom.server

Train and test:
python train.py --dataroot ./datasets/jointremoval_v1 --name task3_v1 --model jointremoval --dataset_mode jointremoval
python test.py --dataroot ./datasets/jointremoval_v1 --name task3_v1 --model jointremoval --dataset_mode jointremoval --version 1

Or:
python train.py --dataroot ./datasets/jointremoval_v2 --name task3_v2 --model jointremoval --dataset_mode jointremoval
python test.py --dataroot ./datasets/jointremoval_v2 --name task3_v2 --model jointremoval --dataset_mode jointremoval --version 2

For all test cases (V2 as examples here):
python test.py --dataroot ./datasets/jointremoval_v2 --name task3_v2 --model jointremoval --dataset_mode jointremoval --test_input B
python test.py --dataroot ./datasets/jointremoval_v2 --name task3_v2 --model jointremoval --dataset_mode jointremoval --test_input BC
python test.py --dataroot ./datasets/jointremoval_v2 --name task3_v2 --model jointremoval --dataset_mode jointremoval --test_input BD
python test.py --dataroot ./datasets/jointremoval_v2 --name task3_v2 --model jointremoval --dataset_mode jointremoval --test_input BCD
python test.py --dataroot ./datasets/jointremoval_v2 --name task3_v2 --model jointremoval --dataset_mode jointremoval --test_input C
python test.py --dataroot ./datasets/jointremoval_v2 --name task3_v2 --model jointremoval --dataset_mode jointremoval --test_input D
python test.py --dataroot ./datasets/jointremoval_v2 --name task3_v2 --model jointremoval --dataset_mode jointremoval --test_input CD

Datasets:
jointremoval : Version 1
jointremoval2: Version 2
"""


class JOINTREMOVALModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for BIDeN model
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
        parser.add_argument('--lambda_Ln', type=float, default=30.0, help='weight for L1/L2 loss')
        parser.add_argument('--lambda_VGG', type=float, default=10.0, help='weight for VGG loss')
        parser.add_argument('--lambda_BCE', type=float, default=1.0, help='weight for BCE loss')
        parser.add_argument('--test_input', type=str, default='B', help='test images, B = shadow,'
                                                                          ' C = reflection, D = watermark')
        parser.add_argument('--max_domain', type=int, default=3, help='max number of source components.')
        parser.add_argument('--prob1', type=float, default=0.8, help='probability of adding shadow (A)')
        parser.add_argument('--prob2', type=float, default=0.5, help='probability of adding reflection, watermark(B,C)')
        parser.add_argument('--version', type=int, default=1, help='1 for Version one (V1), 2 for Version two (V2),'
                                                                   'only matters for testing')
        opt, _ = parser.parse_known_args()
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # Specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'Ln', 'VGG', 'BCE']
        self.visual_names = ['fake_A', 'fake_B', 'fake_D', 'real_A2', 'real_A', 'real_B', 'real_D', 'real_D2', 'real_C',
                             'real_input']
        self.model_names = ['D', 'E', 'H1', 'H2', 'H3']

        # Define networks (both generator and discriminator)
        # Define Encoder E.
        self.netE = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'encoder', opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                      opt.no_antialias_up, self.gpu_ids, opt)
        # Define Heads H.
        self.netH1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'head', opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                       opt.no_antialias_up, self.gpu_ids, opt)
        self.netH2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'head', opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                       opt.no_antialias_up, self.gpu_ids, opt)
        self.netH3 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'head', opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                       opt.no_antialias_up, self.gpu_ids, opt)
        self.label = torch.zeros(self.opt.max_domain).to(self.device)
        # Define Discriminator D.
        self.netD = networks.define_D(opt.output_nc, self.opt.max_domain, 'BIDeN_D', opt.n_layers_D, opt.normD,
                                      opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.accuracy = 0
        self.all_count = 0
        self.width = self.opt.crop_size

        if self.isTrain:
            # Define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionVGG = VGGLoss(opt).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.criterionBCE = torch.nn.BCEWithLogitsLoss().to(self.device)
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netE.parameters(), self.netH1.parameters(),
                                self.netH2.parameters(), self.netH3.parameters()),
                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.real_C = self.real_C[:bs_per_gpu]
        self.real_D = self.real_D[:bs_per_gpu]
        self.real_D2 = self.real_D2[:bs_per_gpu]
        self.real_A2 = self.real_A2[:bs_per_gpu]
        self.forward()
        if self.opt.isTrain:
            self.compute_D_loss().backward()  # calculate gradients for D
            self.compute_G_loss().backward()  # calculate graidents for G

    def optimize_parameters(self):
        # forward
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        A: Shadow-free image.
        A2: Shadow image.
        B: Shadow mask.
        C: Reflection layer image.
        D: Watermark image.
        D2: Watermark mask.
        temp_A, temp_A2 load A, A2, used in adding reflection/watermark.
        Default normalization makes data between [-1,1], we scale some of them to [0,1] for adding reflection/watermark.
        """
        self.temp_A = (input['A'] + 1.0) / 2.0
        self.temp_A2 = (input['A2'] + 1.0) / 2.0
        self.real_A = input['A'].to(self.device)
        self.real_B = (input['B'] + 1.0) / 2.0
        self.real_C = (input['C'] + 1.0) / 2.0
        self.real_D = input['D1'].to(self.device)
        self.real_A2 = input['A2'].to(self.device)
        self.real_D2 = (input['D2'] + 1.0) / 2.0
        self.image_paths = input['A_paths']

    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters>.
        We have another version of forward (forward_test) used in testing.
        """
        # Add shadow or not. Note we do not "add" shadow, we use shadow image and shadow-free image.
        p = torch.rand(self.opt.max_domain)
        self.label[0] = 1 if p[0] < self.opt.prob1 else 0

        # Add reflection, watermark or not.
        for i in range(1, self.opt.max_domain):
            if p[i] < self.opt.prob2:
                self.label[i] = 1
            else:
                self.label[i] = 0

        # Based on the label, starts adding shadow/reflection/watermark.
        temp = self.temp_A[0] if self.label[0] == 0 else self.temp_A2[0]
        temp = temp.numpy()
        if self.label[1] == 1:
            k_sz = np.linspace(1, 5, 80)
            sigma = k_sz[np.random.randint(0, len(k_sz))]
            c = self.real_C.numpy()[0]
            _, _, temp = self.syn_data(temp, c, sigma)
        if self.label[2] == 1:
            A = 0.8 + 0.2 * random.random()
            d = self.real_D2.numpy()[0]
            temp = temp * (1 - d) + A * d

        # Convert process image temp to tensor.
        self.real_input = torch.from_numpy(temp.reshape(1, 3, self.width, self.width))
        self.real_B = (self.real_B * 2.0 - 1.0).to(self.device)
        self.real_C = (self.real_C * 2.0 - 1.0).to(self.device)
        self.real_D2 = (self.real_D2 * 2.0 - 1.0).to(self.device)
        self.real_input = self.real_input.type_as(self.real_A)
        self.real_input = (self.real_input * 2.0 - 1.0).to(self.device)

        # Get the reconstructed results. We do not require the reconstruction of reflection layer image.
        self.fake_all = self.netE(self.real_input)
        self.fake_A = self.netH1(self.fake_all)
        self.fake_B = self.netH2(self.fake_all)
        self.fake_D = self.netH3(self.fake_all)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake1 = self.fake_A.detach()
        fake2 = self.fake_B.detach()
        fake3 = self.fake_D.detach()
        pred_fake1 = self.netD(0, fake1)
        pred_fake2 = self.netD(0, fake2)
        pred_fake3 = self.netD(0, fake3)
        self.loss_D_fake = self.criterionGAN(pred_fake1, False) \
                           + self.criterionGAN(pred_fake2, False) * self.label[0] \
                           + self.criterionGAN(pred_fake3, False) * self.label[2]

        # Real
        self.pred_real1 = self.netD(0, self.real_A)
        self.pred_real2 = self.netD(0, self.real_B)
        self.pred_real3 = self.netD(0, self.real_D)
        self.loss_D_real = self.criterionGAN(self.pred_real1, True) \
                           + self.criterionGAN(self.pred_real2, True) * self.label[0] \
                           + self.criterionGAN(self.pred_real3, True) * self.label[2]

        # BCE loss, netD(1) for the source prediction branch.
        self.predict_label = self.netD(1, self.real_input).view(self.opt.max_domain)
        self.loss_BCE = self.criterionBCE(self.predict_label, self.label)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_GAN \
                      + self.loss_BCE * self.opt.lambda_BCE

        return self.loss_D

    def compute_G_loss(self):
        pred_fake1 = self.netD(0, self.fake_A)
        pred_fake2 = self.netD(0, self.fake_B)
        pred_fake3 = self.netD(0, self.fake_D)

        self.loss_G_GAN = self.criterionGAN(pred_fake1, True) \
                          + self.criterionGAN(pred_fake2, True) * self.label[0] \
                          + self.criterionGAN(pred_fake3, True) * self.label[2]

        self.loss_Ln = self.criterionL1(self.real_A, self.fake_A) \
                       + self.criterionL2(self.real_B, self.fake_B) * self.label[0] \
                       + self.criterionL2(self.real_D, self.fake_D) * self.label[2]

        self.loss_VGG = self.criterionVGG(self.fake_A, self.real_A) \
                        + self.criterionVGG(self.fake_B, self.real_B) * self.label[0] \
                        + self.criterionVGG(self.fake_D, self.real_D) * self.label[2]

        self.loss_G = self.loss_G_GAN * self.opt.lambda_GAN + self.loss_Ln * self.opt.lambda_Ln \
                      + self.loss_VGG * self.opt.lambda_VGG

        return self.loss_G

    def forward_test(self):
        gt_label = [0] * self.opt.max_domain
        if 'B' in self.opt.test_input:
            gt_label[0] = 1
        if 'C' in self.opt.test_input:
            gt_label[1] = 1
        if 'D' in self.opt.test_input:
            gt_label[2] = 1
        temp = self.temp_A[0] if gt_label[0] == 0 else self.temp_A2[0]
        temp = temp.numpy()
        if gt_label[1] == 1:
            # Fixed during testing, which is kernel size 11.
            sigma = 2.5
            c = self.real_C.numpy()[0]
            _, _, temp = self.syn_data(temp, c, sigma)
        if gt_label[2] == 1:
            # Fix 0.9 during testing.
            A = 0.9
            d = self.real_D2.numpy()[0]
            temp = temp * (1 - d) + A * d
        self.real_input = torch.from_numpy(temp.reshape(1, 3, self.width, self.width))
        self.real_B = (self.real_B * 2.0 - 1.0).to(self.device)
        self.real_C = (self.real_C * 2.0 - 1.0).to(self.device)
        self.real_D2 = (self.real_D2 * 2.0 - 1.0).to(self.device)
        self.real_input = self.real_input.type_as(self.real_A)
        self.real_input = (self.real_input * 2.0 - 1.0).to(self.device)
        self.fake_all = self.netE(self.real_input)
        self.predict_label = self.netD(1, self.real_input).view(self.opt.max_domain)
        predict_label = torch.where(self.predict_label > 0.0, 1, 0)
        self.fake_A = self.netH1(self.fake_all)
        self.fake_B = self.netH2(self.fake_all)
        self.fake_D = self.netH3(self.fake_all)
        if predict_label.tolist() == gt_label:
            self.accuracy = self.accuracy + 1
        self.all_count = self.all_count + 1
        total_num = 540 if self.opt.version == 1 else 408
        if self.all_count == total_num:
            print("Accuracy: ", self.accuracy / self.all_count)

    # Functions for adding reflection, borrowed from "Single Image Reflection Separation with Perceptual Losses".
    def gkern(self, kernlen=100, nsig=1):
        """Returns a 2D Gaussian kernel array."""
        interval = (2 * nsig + 1.) / kernlen
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel / kernel.max()
        return kernel

    def syn_data(self, t, r, sigma):
        g_mask = self.gkern(self.width, 3)
        g_mask = np.stack((g_mask, g_mask, g_mask))
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)
        sz = int(2 * np.ceil(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
        blend = r_blur + t
        att = 1.08 + np.random.random() / 10.0
        for i in range(3):
            maski = blend[:, :, i] > 1
            mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
        r_blur[r_blur >= 1] = 1
        r_blur[r_blur <= 0] = 0
        alpha1 = g_mask
        alpha2 = 1 - np.random.random() / 5.0
        r_blur_mask = np.multiply(r_blur, alpha1)
        blend = r_blur_mask + t * alpha2
        t = np.power(t, 1 / 2.2)
        r_blur_mask = np.power(r_blur_mask, 1 / 2.2)
        blend = np.power(blend, 1 / 2.2)
        blend[blend >= 1] = 1
        blend[blend <= 0] = 0
        return t, r_blur_mask, blend
