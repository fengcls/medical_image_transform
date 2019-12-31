import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks as networks
import torch.autograd as autograd

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0)
        parser.set_defaults(no_lsgan=True)
        parser.set_defaults(norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_256')

        return parser

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        use_cuda = True
        LAMBDA = 10 # Gradient penalty lambda hyperparameter
        BATCH_SIZE = real_data.size(0)
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(BATCH_SIZE, torch.tensor(real_data.nelement()/BATCH_SIZE,dtype=torch.int)).contiguous().view(real_data.size())
        alpha = alpha.cuda() if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if use_cuda:
            interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                      disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty
   
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        if opt.using_gan:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        else:
            self.loss_names = ['G_L1']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        if opt.which_model_netG == ['unet_2d_256', 'unet_2d_128', 'unet_2d_64', 'unet_2d_32']:
            self.netG = networks.define_G((opt.timesteps-1)*opt.input_nc, opt.output_nc, opt.ngf, # flatten ch and time
                                  opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        else:
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                  opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            print('use_sigmoid:',use_sigmoid)
            self.netD = networks.define_D(opt.input_nc*2, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                          conv3d_kw=opt.conv3d_kw, conv3d_padw=opt.conv3d_padw)
            # replace opt.input_nc + opt.output_nc with #channel
        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionGD = networks.gradient_loss

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        ### Fake
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = pred_fake.mean()
        
        ### Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = -pred_real.mean()
                
        gradient_penalty = self.calc_gradient_penalty(self.netD, real_AB, fake_AB.detach())
        
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * self.opt.lambda_D_cmp_gp + gradient_penalty
        
        self.loss_D.backward()

    def backward_G(self, using_gan=True):
        # First, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        if using_gan:
            # Second, G(A) should fake the discriminator
            fake_AB = torch.cat((self.real_A, self.fake_B), 1) # time dim
            pred_fake = self.netD(fake_AB) # switch time channel

            self.loss_G_GAN = -pred_fake.mean()

            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        else:
            self.loss_G = self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self, using_gan=True):
        self.forward()

        if using_gan:
            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G(using_gan)
        self.optimizer_G.step()
