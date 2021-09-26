import torch
from .base_model import BaseModel
from . import networks
import random
from .losses import VGGLoss
from numpy import *
import itertools
import cv2
import numpy as np
from random import choice


""" 
BIDeN model for Task II, real-scenario deraining.

Sample usage:
Optional visualization:
python -m visdom.server

Train:
python train.py --dataroot ./datasets/rain --name task2 --model rain --dataset_mode rain

For all 6 testing cases, run following:
python test.py --dataroot ./datasets/rain --name task2 --model rain --dataset_mode rain --test_input B 
python test.py --dataroot ./datasets/rain --name task2 --model rain --dataset_mode rain --test_input BC
python test.py --dataroot ./datasets/rain --name task2 --model rain --dataset_mode rain --test_input BD --haze_intensity 0
python test.py --dataroot ./datasets/rain --name task2 --model rain --dataset_mode rain --test_input BD --haze_intensity 2
python test.py --dataroot ./datasets/rain --name task2 --model rain --dataset_mode rain --test_input BDE --haze_intensity 1
python test.py --dataroot ./datasets/rain --name task2 --model rain --dataset_mode rain --test_input BCDE --haze_intensity 1  
"""

class RAINModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for BIDeN.
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
        parser.add_argument('--lambda_Ln', type=float, default=30.0, help='weight for L1/L2 loss')
        parser.add_argument('--lambda_VGG', type=float, default=10.0, help='weight for VGG loss')
        parser.add_argument('--lambda_BCE', type=float, default=1.0, help='weight for BCE loss')
        parser.add_argument('--test_input', type=str, default='B', help='test images, B = rain streak,'
                                                                          ' C = snow, D = haze, E = raindrop.')
        parser.add_argument('--max_domain', type=int, default=4, help='max number of source components.')
        parser.add_argument('--prob1', type=float, default=1.0, help='probability of adding rain streak (A)')
        parser.add_argument('--prob2', type=float, default=0.5, help='probability of adding other components')
        parser.add_argument('--haze_intensity', type=int, default=1, help='intensity of haze, only matters for testing. '
                                                                          '0: light, 1: moderate, 2: heavy.')
        opt, _ = parser.parse_known_args()
        return parser

    def __init__(self, opt):
        # Specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'Ln', 'VGG', 'BCE']
        self.visual_names = ['fake_A', 'fake_B', 'fake_C', 'fake_D', 'fake_E', 'real_A', 'real_B', 'real_C', 'real_D',
                             'real_E', 'real_input']
        self.model_names = ['D', 'E', 'H1', 'H2', 'H3', 'H4', 'H5']

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
        self.netH4 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'head', opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                       opt.no_antialias_up, self.gpu_ids, opt)
        self.netH5 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'head', opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                       opt.no_antialias_up, self.gpu_ids, opt)

        self.label = torch.zeros(self.opt.max_domain).to(self.device)
        self.netD = networks.define_D(opt.output_nc, self.opt.max_domain, 'BIDeN_D', opt.n_layers_D, opt.normD,
                                      opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.accuracy = 0
        self.all_count = 0
        self.width = self.opt.crop_size

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionVGG = VGGLoss(opt).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.criterionBCE = torch.nn.BCEWithLogitsLoss().to(self.device)
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netE.parameters(), self.netH1.parameters(),
                                self.netH2.parameters(), self.netH3.parameters(),
                                self.netH4.parameters(), self.netH5.parameters()),
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
        self.real_E = self.real_E[:bs_per_gpu]
        self.real_E2 = self.real_E2[:bs_per_gpu]
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
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        A: CityScape image.
        B: Rain streak mask.
        C: Snow mask.
        D: Haze. (Transmission map)
        D1: Light, D2: Medium, D3: Heavy.
        E: Raindrop mask.
        E2: Raindrop texture, paired with E.
        temp_A, used in adding rain streak/snow/haze/raindrop.
        Default normalization makes data between [-1,1], we scale some of them to [0,1] for adding rain components.
        """
        self.temp_A = (input['A'] + 1.0) / 2.0
        self.real_A = input['A'].to(self.device)
        self.real_B = (input['B'] + 1.0) / 2.0
        self.real_C = (input['C'] + 1.0) / 2.0
        haze = ['D1', 'D2', 'D3']
        if self.isTrain:
            # Randomly choose one.
            self.real_D = (input[choice(haze)] + 1.0) / 2.0
        else:
            # During test, specify a certain intensity.
            self.real_D = (input[haze[self.opt.haze_intensity]] + 1.0) / 2.0
        self.real_E = (input['E1'] + 1.0) / 2.0
        self.real_E2 = (input['E2'] + 1.0) / 2.0
        self.image_paths = input['A_paths']

    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters>.
        We have another version of forward (forward_test) used in testing.
        """
        p = torch.rand(self.opt.max_domain)
        # Add rain streak or not. We set rain streak to be always added in the default setting.
        self.label[0] = 1 if p[0] < self.opt.prob1 else 0
        # Add snow/haze/raindrop or not.
        for i in range(1, self.opt.max_domain):
            if p[i] < self.opt.prob2:
                self.label[i] = 1
            else:
                self.label[i] = 0
        label_sum = torch.sum(self.label, 0)
        # Based on the label, starts adding rain components.
        temp = self.temp_A[0].numpy()
        if self.label[0] == 1:
            A = 0.8 + 0.2 * random.random()
            b = self.real_B.numpy()[0]
            temp = self.generate_img(temp, b, A)
        if self.label[1] == 1:
            A = 0.8 + 0.2 * random.random()
            c = self.real_C.numpy()[0]
            temp = self.generate_img(temp, c, A)
        if self.label[2] == 1:
            A = 0.8 + 0.2 * random.random()
            d = self.real_D.numpy()[0]
            temp = self.generate_haze(temp, d, A)
        if self.label[3] == 1:
            e1 = self.real_E.numpy()[0]
            e2 = self.real_E2.numpy()[0]
            e1 = np.transpose(e1, (2, 1, 0))
            e2 = np.transpose(e2, (2, 1, 0))
            position_matrix, alpha = self.get_position_matrix(e2, e1)
            temp = np.transpose(temp, (2, 1, 0))
            temp = self.composition_img(temp, alpha, position_matrix, rate=0.8 + 0.18 * random.random())

        # Convert process image temp to tensor.
        self.real_input = torch.from_numpy(temp.reshape(1, 3, self.width, self.width))
        self.real_B = (self.real_B * 2.0 - 1.0).to(self.device)
        self.real_C = (self.real_C * 2.0 - 1.0).to(self.device)
        self.real_D = (self.real_D * 2.0 - 1.0).to(self.device)
        self.real_E = (self.real_E * 2.0 - 1.0).to(self.device)
        self.real_input = self.real_input.type_as(self.real_A)
        self.real_input = (self.real_input * 2.0 - 1.0).to(self.device)

        # Get the reconstructed results.
        self.fake_all = self.netE(self.real_input)
        self.fake_A = self.netH1(self.fake_all)
        self.fake_B = self.netH2(self.fake_all)
        self.fake_C = self.netH3(self.fake_all)
        self.fake_D = self.netH4(self.fake_all)
        self.fake_E = self.netH5(self.fake_all)
        self.loss_sum = label_sum

    def compute_D_loss(self):
        """Calculate GAN loss and BCE loss for the discriminator"""
        fake1 = self.fake_A.detach()
        fake2 = self.fake_B.detach()
        fake3 = self.fake_C.detach()
        fake4 = self.fake_D.detach()
        fake5 = self.fake_E.detach()
        pred_fake1 = self.netD(0, fake1)
        pred_fake2 = self.netD(0, fake2)
        pred_fake3 = self.netD(0, fake3)
        pred_fake4 = self.netD(0, fake4)
        pred_fake5 = self.netD(0, fake5)
        self.loss_D_fake = self.criterionGAN(pred_fake1, False) \
                           + self.criterionGAN(pred_fake2, False) * self.label[0] \
                           + self.criterionGAN(pred_fake3, False) * self.label[1] \
                           + self.criterionGAN(pred_fake4, False) * self.label[2] \
                           + self.criterionGAN(pred_fake5, False) * self.label[3]

        self.pred_real1 = self.netD(0, self.real_A)
        self.pred_real2 = self.netD(0, self.real_B)
        self.pred_real3 = self.netD(0, self.real_C)
        self.pred_real4 = self.netD(0, self.real_D)
        self.pred_real5 = self.netD(0, self.real_E)

        self.loss_D_real = self.criterionGAN(self.pred_real1, True) \
                           + self.criterionGAN(self.pred_real2, True) * self.label[0] \
                           + self.criterionGAN(self.pred_real3, True) * self.label[1] \
                           + self.criterionGAN(self.pred_real4, True) * self.label[2] \
                           + self.criterionGAN(self.pred_real5, True) * self.label[3]

        # BCE loss, netD(1) for the source prediction branch.

        self.predict_label = self.netD(1, self.real_input).view(self.opt.max_domain)
        self.loss_BCE = self.criterionBCE(self.predict_label, self.label)

        # combine loss and calculate gradients.
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_GAN + \
                      self.loss_BCE * self.opt.lambda_BCE
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN loss, Ln loss, VGG loss for the generator"""
        # netD(0) for the separation branch.
        pred_fake1 = self.netD(0, self.fake_A)
        pred_fake2 = self.netD(0, self.fake_B)
        pred_fake3 = self.netD(0, self.fake_C)
        pred_fake4 = self.netD(0, self.fake_D)
        pred_fake5 = self.netD(0, self.fake_E)

        self.loss_G_GAN = self.criterionGAN(pred_fake1, True) \
                          + self.criterionGAN(pred_fake2, True) * self.label[0] \
                          + self.criterionGAN(pred_fake3, True) * self.label[1] \
                          + self.criterionGAN(pred_fake4, True) * self.label[2] \
                          + self.criterionGAN(pred_fake5, True) * self.label[3]

        self.loss_Ln = self.criterionL1(self.real_A, self.fake_A) \
                       + self.criterionL2(self.real_B, self.fake_B) * self.label[0] \
                       + self.criterionL2(self.real_C, self.fake_C) * self.label[1] \
                       + self.criterionL1(self.real_D, self.fake_D) * self.label[2] \
                       + self.criterionL2(self.real_E, self.fake_E) * self.label[3]

        self.loss_VGG = self.criterionVGG(self.fake_A, self.real_A) \
                        + self.criterionVGG(self.fake_B, self.real_B) * self.label[0] \
                        + self.criterionVGG(self.fake_C, self.real_C) * self.label[1] \
                        + self.criterionVGG(self.fake_D, self.real_D) * self.label[2] \
                        + self.criterionVGG(self.fake_E, self.real_E) * self.label[3]

        self.loss_G = self.loss_G_GAN * self.opt.lambda_GAN + self.loss_Ln * self.opt.lambda_Ln + self.loss_VGG * self.opt.lambda_VGG

        return self.loss_G

    def forward_test(self):
        gt_label = [0] * self.opt.max_domain
        if 'B' in self.opt.test_input:
            gt_label[0] = 1
        if 'C' in self.opt.test_input:
            gt_label[1] = 1
        if 'D' in self.opt.test_input:
            gt_label[2] = 1
        if 'E' in self.opt.test_input:
            gt_label[3] = 1
        temp = self.temp_A[0].numpy()
        if gt_label[0] == 1:
            # Fixed A during testing.
            A = 0.9
            b = self.real_B.numpy()[0]
            temp = self.generate_img(temp, b, A)
        if gt_label[1] == 1:
            # Fixed A during testing.
            A = 0.9
            c = self.real_C.numpy()[0]
            temp = self.generate_img(temp, c, A)
        if gt_label[2] == 1:
            # Fixed A during testing.
            A = 0.9
            d = self.real_D.numpy()[0]
            temp = self.generate_haze(temp, d, A)
        if gt_label[3] == 1:
            e1 = self.real_E.numpy()[0]
            e2 = self.real_E2.numpy()[0]
            e1 = np.transpose(e1, (2, 1, 0))
            e2 = np.transpose(e2, (2, 1, 0))
            position_matrix, alpha = self.get_position_matrix(e2, e1)
            temp = np.transpose(temp, (2, 1, 0))
            # Fixed rate = 0.9 during testing.
            temp = self.composition_img(temp, alpha, position_matrix, rate=0.9)
        self.real_input = torch.from_numpy(temp.reshape(1, 3, self.width, self.width))
        self.real_B = (self.real_B * 2.0 - 1.0).to(self.device)
        self.real_C = (self.real_C * 2.0 - 1.0).to(self.device)
        self.real_D = (self.real_D * 2.0 - 1.0).to(self.device)
        self.real_E = (self.real_E * 2.0 - 1.0).to(self.device)
        self.real_input = self.real_input.type_as(self.real_A)
        self.real_input = (self.real_input * 2.0 - 1.0).to(self.device)
        self.fake_all = self.netE(self.real_input)
        self.fake_A = self.netH1(self.fake_all)
        self.fake_B = self.netH2(self.fake_all)
        self.fake_C = self.netH3(self.fake_all)
        self.fake_D = self.netH4(self.fake_all)
        self.fake_E = self.netH5(self.fake_all)
        self.predict_label = self.netD(1, self.real_input).view(self.opt.max_domain)
        predict_label = torch.where(self.predict_label > 0.0, 1, 0)
        if predict_label.tolist() == gt_label:
            self.accuracy = self.accuracy + 1
        self.all_count = self.all_count + 1
        if self.all_count == 500:
            print("Accuracy: ", self.accuracy / self.all_count)

    # Rain streak, snow.
    def generate_img(self, img1, img2, A):
        img1 = img1 * (1 - img2) + A * img2
        return img1

    # Haze.
    def generate_haze(self, img1, img2, A):
        img1 = img1 * img2 + A * (1 - img2)
        return img1

    # The following functions are for raindrop, please check more details at ./raindrop.
    def get_position_matrix(self, texture, alpha):
        alpha = cv2.blur(alpha, (5, 5))
        position_matrix = np.mgrid[0:self.width, 0:self.width]
        position_matrix[0, :, :] = position_matrix[0, :, :] + texture[:, :, 2] * (texture[:, :, 0])
        position_matrix[1, :, :] = position_matrix[1, :, :] + texture[:, :, 1] * (texture[:, :, 0])
        position_matrix = position_matrix * (alpha[:, :, 0] > 0.3)

        return position_matrix, alpha

    def composition_img(self, img, alpha, position_matrix, rate, length=2):
        h, w = img.shape[0:2]
        dis_img = img.copy()
        for x in range(h):
            for y in range(w):
                u, v = int(position_matrix[0, x, y] / length), int(position_matrix[1, x, y] / length)
                if u != 0 and v != 0:
                    if (u < h) and (v < w):
                        dis_img[x, y, :] = dis_img[u, v, :]
                    elif u < h:
                        dis_img[x, y, :] = dis_img[u, np.random.randint(0, w - 1), :]
                    elif v < w:
                        dis_img[x, y, :] = dis_img[np.random.randint(0, h - 1), v, :]
        dis_img = cv2.blur(dis_img, (3, 3)) * rate
        img = alpha * dis_img + (1 - alpha) * img
        return np.transpose(img, (2, 1, 0))
