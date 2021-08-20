import torch
from .base_model import BaseModel
from . import networks
from .losses import VGGLoss
from numpy import *
import itertools

""" 
BIDeN model for Task I, Mixed image decomposition across multiple domains. Max number of domain = 2.

Sample usage:
Optional visualization:
python -m visdom.server

Train:
python train.py --dataroot ./datasets/image_decom --name biden2 --model biden2 --dataset_mode unaligned2

For test:
Test a single case:
python test.py --dataroot ./datasets/image_decom --name biden2 --model biden2 --dataset_mode unaligned2 --test_input A
python test.py --dataroot ./datasets/image_decom --name biden2 --model biden2 --dataset_mode unaligned2 --test_input AB
... ane other cases.
change test_input to the case you want.

Test all cases:
python test2.py --dataroot ./datasets/image_decom --name biden2 --model biden2 --dataset_mode unaligned2
"""

class BIDEN2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for BIDeN model
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
        parser.add_argument('--lambda_Ln', type=float, default=30.0, help='weight for L1/L2 loss')
        parser.add_argument('--lambda_VGG', type=float, default=10.0, help='weight for VGG loss')
        parser.add_argument('--lambda_BCE', type=float, default=1.0, help='weight for BCE loss')
        parser.add_argument('--test_input', type=str, default='A', help='test mixed images.')
        parser.add_argument('--max_domain', type=int, default=2, help='max number of source components.')
        parser.add_argument('--prob', type=float, default=0.9, help='probability of adding a component')
        parser.add_argument('--test_choice', type=int, default=1, help='choice for test mode, 1 for one case,'
                                                                       ' 0 for all cases. Will be set automatically.')
        opt, _ = parser.parse_known_args()
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'Ln', 'VGG', 'BCE']
        self.visual_names = ['fake_A', 'fake_B', 'real_A', 'real_B', 'real_input']
        self.model_names = ['D', 'E', 'H1', 'H2']

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

        self.label = torch.zeros(self.opt.max_domain).to(self.device)
        self.netD = networks.define_D(opt.output_nc, self.opt.max_domain, 'BIDeN_D', opt.n_layers_D, opt.normD,
                                      opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.correct = 0
        self.all_count = 0
        self.psnr_count = [0] * self.opt.max_domain
        self.acc_all = 0
        self.test_time = 0
        self.criterionL2 = torch.nn.MSELoss().to(self.device)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionVGG = VGGLoss(opt).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionBCE = torch.nn.BCEWithLogitsLoss().to(self.device)
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netE.parameters(), self.netH1.parameters(),
                                self.netH2.parameters()),
                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def data_dependent_initialize(self, data):
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
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
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters>.
        We have another version of forward (forward_test) used in testing.
        """
        # If no images are used in mixing, run again.
        label_sum = 0
        while label_sum < 1:
            p = torch.rand(self.opt.max_domain)
            for i in range(self.opt.max_domain):
                # Note here the probability is actually (1-self.opt.prob).
                if p[i] < self.opt.prob:
                    self.label[i] = 1
                else:
                    self.label[i] = 0
            label_sum = torch.sum(self.label, 0)

        self.real_input = (self.real_A * self.label[0] + self.real_B * self.label[1]) / label_sum
        self.fake_all = self.netE(self.real_input)
        self.fake_A = self.netH1(self.fake_all)
        self.fake_B = self.netH2(self.fake_all)
        self.loss_sum = label_sum


    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake1 = self.fake_A.detach()
        fake2 = self.fake_B.detach()
        pred_fake1 = self.netD(0,fake1)
        pred_fake2 = self.netD(0,fake2)
        self.loss_D_fake = self.criterionGAN(pred_fake1, False) * self.label[0] \
                           + self.criterionGAN(pred_fake2, False) * self.label[1]

        # Real
        self.pred_real1 = self.netD(0,self.real_A)
        self.pred_real2 = self.netD(0,self.real_B)

        self.loss_D_real = self.criterionGAN(self.pred_real1, True) * self.label[0] \
                           + self.criterionGAN(self.pred_real2,True) * self.label[1]

        # combine loss and calculate gradients
        self.predict_label = self.netD(1,self.real_input).view(self.opt.max_domain)
        self.loss_BCE = self.criterionBCE(self.predict_label, self.label)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_GAN + self.loss_BCE * self.opt.lambda_BCE
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake1 = self.netD(0,self.fake_A)
        pred_fake2 = self.netD(0,self.fake_B)

        self.loss_G_GAN = self.criterionGAN(pred_fake1, True) * self.label[0] \
                          + self.criterionGAN(pred_fake2, True) * self.label[1]

        self.loss_Ln = self.criterionL1(self.real_A, self.fake_A) * self.label[0] \
                       + self.criterionL1(self.real_B,self.fake_B) * self.label[1]

        self.loss_VGG = self.criterionVGG(self.fake_A, self.real_A) * self.label[0] \
                        + self.criterionVGG(self.fake_B, self.real_B) * self.label[1]

        self.loss_G = self.loss_G_GAN * self.opt.lambda_GAN + self.loss_Ln * self.opt.lambda_Ln \
                      + self.loss_VGG * self.opt.lambda_VGG

        return self.loss_G

    def forward_test(self):
        # Test case 1, test a single case only, write the output images.
        if self.opt.test_choice == 1:
            gt_label = [0] * self.opt.max_domain
            if 'A' in self.opt.test_input:
                gt_label[0] = 1
            if 'B' in self.opt.test_input:
                gt_label[1] = 1
            self.real_input = (self.real_A * gt_label[0] + self.real_B * gt_label[1]) / sum(gt_label)
            self.fake_all = self.netE(self.real_input)
            self.predict_label = self.netD(1, self.real_input).view(self.opt.max_domain)
            predict_label = torch.where(self.predict_label > 0.0, 1, 0)
            self.fake_A = self.netH1(self.fake_all)
            self.fake_B = self.netH2(self.fake_all)
            if predict_label.tolist() == gt_label:
                self.correct = self.correct + 1
            self.all_count = self.all_count + 1
            if self.all_count == 300:
                print(self.correct / self.all_count)
        else:
            # Test case 0, test all cases, do not write output images.
            gt_label = [0] * self.opt.max_domain
            if 'A' in self.opt.test_input:
                gt_label[0] = 1
            if 'B' in self.opt.test_input:
                gt_label[1] = 1
            self.real_input = (self.real_A * gt_label[0] + self.real_B * gt_label[1])/ sum(gt_label)
            self.predict_label = self.netD(1, self.real_input).view(self.opt.max_domain)
            predict_label = torch.where(self.predict_label > 0.0, 1, 0)
            self.fake_all = self.netE(self.real_input)
            self.fake_A = self.netH1(self.fake_all)
            self.fake_B = self.netH2(self.fake_all)

            # Normalize to 0-1 for PSNR calculation.
            self.fake_A = (self.fake_A + 1)/2
            self.real_A = (self.real_A + 1)/2
            self.fake_B = (self.fake_B + 1)/2
            self.real_B = (self.real_B + 1)/2

            if gt_label[0] == 1:
                mse = self.criterionL2(self.fake_A, self.real_A)
                psnr = 10 * log10(1 / mse.item())
                self.psnr_count[0] += psnr
            if gt_label[1] == 1:
                mse = self.criterionL2(self.fake_B, self.real_B)
                psnr = 10 * log10(1 / mse.item())
                self.psnr_count[1] += psnr
            if predict_label.tolist() == gt_label:
                self.correct = self.correct + 1
            self.all_count = self.all_count + 1
            if self.all_count % 300 == 0:
                acc = self.correct / self.all_count
                print("Accuracy for current: ",acc)
                if gt_label[0] == 1:
                    print("PSNR_A: ", self.psnr_count[0]/ self.all_count)
                if gt_label[1] == 1:
                    print("PSNR_B: ", self.psnr_count[1]/ self.all_count)
                self.all_count = 0
                self.correct = 0
                self.psnr_count = [0] * self.opt.max_domain
                self.acc_all += acc
                self.test_time +=1
                if mean(gt_label) == 1:
                    print("Overall Accuracy:", self.acc_all/self.test_time)




