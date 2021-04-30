import torch
from .base_model import BaseModel
from .emb_model import EmbModel
from . import networks


class MyModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='my', input_nc=5, output_nc=3, niter=50, niter_decay=50, display_port=9333)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['comp', 'real', 'harmonized']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        opt.input_nc = 4  # !!!!
        opt.output_nc = 4  # !!!!
        embedding = EmbModel(opt)
        embedding.save_dir = opt.embedding_save_dir
        embedding.load_networks('latest')
#         embedding.set_requires_grad(['G'], False)
        embedding.print_networks(verbose=False)
        embedding.eval()
        self.embedding = embedding
        
        self.bg_intermediates = {}
        self.fg_intermediates = {}

        
        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizers.append(self.optimizer_G)
        else:
            self.criterionL1 = torch.nn.MSELoss()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.image_paths = input['img_path']
        
        mask = input['mask'].to(self.device)
        white_mask = torch.ones_like(mask) * 255
        
        self.comp = input['comp'].to(self.device)
        self.m_comp = torch.cat([self.comp, mask], 1).to(self.device)
#         self.bg = torch.cat([comp, white_mask], 1).to(self.device)
        
#         self.bg_embedding = self.embedding(self.bg).detach()
#         self.bg_intermediates, embedding.intermediates = embedding.intermediates, {}
        self.fg_embedding = self.embedding(self.m_comp).detach()
        self.fg_intermediates, self.embedding.intermediates = self.embedding.intermediates, {}
        
        self.real = input['real'].to(self.device)
        self.depth = input['depth'].to(self.device)
        self.inputs = torch.cat([self.m_comp, self.depth], 1).to(self.device)
        
        self.netG.fg_embedding = self.fg_embedding
#         self.netG.bg_embedding = self.bg_embedding
        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.harmonized = self.netG(self.inputs)  # G(A)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        # fake_AB = torch.cat((self.comp, self.harmonized), 1)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.harmonized, self.real) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
