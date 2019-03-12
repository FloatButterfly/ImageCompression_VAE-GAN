import numpy as np
import torch
import torch.nn.functional as F

from . import networks
from .base_model import BaseModel


class labelZvaeModel(BaseModel):
    def name(self):
        return 'labelZvaeModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    """
    --A: sketch
    --B: ground truth
    
    """

    def initialize(self, opt):
        if opt.isTrain:
            assert opt.batch_size % 2 == 0  # load two images at one time.

        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.tensor_name = ['z_encoded']
        self.loss_names = ['G_GAN', 'D', 'G_L1', 'z_L1', 'kl']
        self.visual_names = ['real_A_encoded', 'real_B_encoded', 'fake_B_encoded']
        self.logname = 'testZ_log.txt'
        # specify the models you want to save to the disk. The program will call base_model.save_networks and
        # base_model.load_networks
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_E = opt.isTrain or not opt.no_encode
        use_vae = True

        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz + 1, opt.ngf, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type,
                                      gpu_ids=self.gpu_ids, where_add=self.opt.where_add, upsample=opt.upsample)
        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        use_sigmoid = opt.gan_mode == 'dcgan'  # default lsgan
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          use_sigmoid=use_sigmoid, init_type=opt.init_type, num_Ds=opt.num_Ds,
                                          gpu_ids=self.gpu_ids)

        if use_E:
            self.model_names += ['E']
            self.netE = networks.define_E(opt.output_nc, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, gpu_ids=self.gpu_ids, vaeLike=use_vae)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(mse_loss=not use_sigmoid).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            ## see use
            self.criterionZ = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if use_E:
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_E)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

    def is_train(self):
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.label = input['label'].to(self.device)
        self.label = torch.squeeze(self.label, 1)

    def creat_labels(self):
        pass

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    # no need for random z
    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.to(self.device)

    def encode(self, input_image):
        mu, logvar = self.netE.forward(input_image)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    def z_encode(self):
        z, logvar = self.netE(self.real_B)
        self.z_encoded = z
        return self.z_encoded

    def test(self, z0=None, encode=False):
        with torch.no_grad():
            if encode:  # use encoded z
                z0, logvar = self.netE(self.real_B)
                # std = logvar.mul(0.5).exp_()
                # eps = self.get_z_random(std.size(0), std.size(1))
                # z0 = eps.mul(std).add_(mu)
                self.z_encoded = z0
                self.logvar = logvar
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
            z0_label = torch.cat([z0, self.label_org], dim=1)
            self.fake_B = self.netG(self.real_A, z0_label)
            print(z0_label)
            # return self.z_encoded, self.real_A, self.fake_B, self.real_B, self.logvar
            return self.real_A, self.fake_B, self.real_B

    def forward(self):
        # get real images
        total_size = self.opt.batch_size
        self.real_A_encoded = self.real_A[0:total_size]
        self.real_B_encoded = self.real_B[0:total_size]
        print(self.label, self.label.shape)
        self.label_org = self.label
        # self.label_org = torch.unsqueeze(self.label, -1)
        # get encoded z
        self.z_encoded, self.mu, self.logvar = self.encode(self.real_B_encoded)
        self.z_encoded_label = torch.cat([self.z_encoded, self.label_org], dim=1)
        # print(self.z_encoded_label, self.z_encoded_label.shape)
        # get random z
        # self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.opt.nz)
        # generate fake_B_encoded
        self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded_label)

        if self.opt.conditional_D:  # tedious conditoinal data
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
            # self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
            # self.real_data_random = torch.cat([self.real_A[half_size:], self.real_B_random], 1)
        else:
            self.fake_data_encoded = self.fake_B_encoded
            self.real_data_encoded = self.real_B_encoded

        # compute z_predict with fake_B_encoded
        if self.opt.lambda_z > 0.0:
            self.mu2, logvar2 = self.netE(self.fake_B_encoded)
        # if self.opt.lambda_z > 0.0:
        #     self.mu2, logvar2 = self.netE(self.fake_B_random)  # mu2 is a point estimate

    # ***************change point******************
    def classification_loss(self, logit, target):
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def backward_D(self, netD, real, fake, label):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake, _ = netD(fake.detach())
        # real
        pred_real, cls_real = netD(real)
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        loss_D_cls = self.classification_loss(cls_real, label)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real + loss_D_cls
        print("loss_D_cls: ", loss_D_cls, "loss_D", loss_D)
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real, loss_D_cls]

    def backward_G_GAN(self, fake, label, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake, cls_fake = netD(fake)
            print("cls_fake: ", cls_fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
            loss_G_cls = self.classification_loss(cls_fake, label)
            loss_G_GAN += loss_G_cls
            print("loss_G_cls: ", loss_G_cls, "loss_G_GAN", loss_G_GAN)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.label_org, self.netD, self.opt.lambda_GAN)
        #
        # 2. KL loss
        if self.opt.lambda_kl > 0.0:
            kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
            self.loss_kl = torch.sum(kl_element).mul_(-0.5) * self.opt.lambda_kl
        else:
            self.loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_kl
        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        self.set_requires_grad(self.netD, True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded,
                                                         self.label_org)
            self.optimizer_D.step()

    def backward_G_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = torch.mean(torch.abs(self.mu2 - self.z_encoded)) * self.opt.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad(self.netD, False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()
        self.optimizer_E.step()
        #
        # update G only
        if self.opt.lambda_z > 0.0:
            self.optimizer_G.zero_grad()
            self.optimizer_E.zero_grad()
            self.backward_G_alone()
            self.optimizer_G.step()
        #

    def optimize_parameters(self):
        self.forward()
        self.update_G_and_E()
        self.update_D()
