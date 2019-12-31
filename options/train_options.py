from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--epoch_step', type=int, default=2000)
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        parser.add_argument('--maxepoch', type=int, default=1280000, metavar='N', help='number of epochs to train (default: 1280000)')

        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

        parser.add_argument('--timesteps', type=int, default= 5, metavar='N', help='time steps to take sub volume, also used to specify StackTemporal')

        parser.add_argument('--rowsize', type=int, default= 128, metavar='N', help='the row size of the volume')
        parser.add_argument('--colsize', type=int, default= 128, metavar='N', help='the col size of the volume')

        parser.add_argument('--unit_ch',default=64,type=int)

        parser.add_argument('--using_wgan',default=False,action='store_true')

        parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')

        self.isTrain = True
        return parser
