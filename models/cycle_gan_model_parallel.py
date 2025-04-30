'''
stripped down version of cycle_gan_model.py
'''
import torch
import torch.nn as nn
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

# https://pytorch.org/tutorials/intermediate/pipelining_tutorial.html

# stripped down pytorch module CycleGAN class
class CycleGAN(nn.Module):

    def __init__(self, isTrain = True, device = 'cuda'):
        super().__init__()
        self.isTrain = isTrain
        self.device = device
        self.to(self.device)
        self.optimizers = []

        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']

        # NOTE: pulling out default values from options.base_options
        # input image channels
        self.input_nc = 3
        # output image channels
        self.output_nc = 3
        # of gen filters in last conv layer
        self.ngf = 64
        # generator architecture
        self.netG = 'resnet_9blocks'
        # normalization type
        self.norm = 'instance'
        # no dropout for generator
        # NOTE: this gets flipped for whatever reason but copying what we did for sequential profiling
        self.no_droupout = True
        # tensor inititialization
        self.init_type = 'normal'
        # scaling factor for tensor init
        self.init_gain = 0.02
        # of discrim filters in first conv layer
        self.ndf = 64
        # discriminator architecture
        self.netD = 'basic'
        # honestly not really sure
        self.n_layers_D = 3

        self.lambda_identity = 0.5
        self.pool_size = 50
        self.lr = 0.0002
        self.beta1 = 0.5
        self.gan_mode = 'lsgan'
        self.lambda_A =  10.0                          
        self.lambda_B =  10.0                          
        self.lambda_identity = 0.5  

        self.direction = 'AtoB'

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(self.input_nc, self.output_nc, self.ngf, self.netG, self.norm,
                                        not self.no_droupout , self.init_type, self.init_gain, []).to(self.device)
        self.netG_B = networks.define_G(self.output_nc, self.input_nc, self.ngf, self.netG, self.norm,
                                        not self.no_droupout , self.init_type, self.init_gain, []).to(self.device)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(self.output_nc, self.ndf, self.netD,
                                            self.n_layers_D, self.norm, self.init_type, self.init_gain, []).to(self.device)
            self.netD_B = networks.define_D(self.input_nc, self.ndf, self.netD,
                                            self.n_layers_D, self.norm, self.init_type, self.init_gain, []).to(self.device)
            

       
            if self.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(self.input_nc == self.output_nc)
            self.fake_A_pool = ImagePool(self.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(self.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(self.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)



        # TODO:
        # if implementing DDP assign device according to assigned rank
        # https://discuss.pytorch.org/t/distributed-data-parallel-and-cuda-graph/169998/4
        # NOTE: experimenting w/ two cuda streams (on same device)
        self.s1 = torch.cuda.Stream(device = self.device)
        self.s2 = torch.cuda.Stream(device = self.device)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self, x):
        
        # syncs forward streams
        self.s1.wait_stream(torch.cuda.current_stream())
        self.s2.wait_stream(torch.cuda.current_stream())

        # TODO: want to find a way to split loading of data in streams ?
        self.set_input(x)

        with torch.cuda.stream(self.s1):
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))

        with torch.cuda.stream(self.s2):
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        torch.cuda.current_stream().wait_stream(self.s1)
        torch.cuda.current_stream().wait_stream(self.s2)

        # NOTE: supposedly good practice, ensures caching allocator safety of memory created
        # on one stream and used on another
        self.real_A.record_stream(self.s1)
        self.real_B.record_stream(self.s2)


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()



    # NOTE: https://pytorch.org/docs/stable/notes/cuda.html#stream-semantics-of-backward-passes
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights


        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad