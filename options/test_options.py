from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')        
        parser.add_argument('--testingset', default = 'UCSDped2', help='')
        parser.add_argument('--trainingset', default = 'ped1_ped2', help='')
        parser.add_argument('--img_size', default = 128, help='',type=int)
        parser.add_argument('--gpu', default = "1", help='',type=str)

        parser.add_argument('--modelsubfolder', default = "", help='')
        parser.add_argument('--feature', default = 'image', help='image|of_m|of_m+of_h+of_v+image')
        parser.add_argument('--weights_name', default='weights.pth')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        
        # parser.set_defaults(model='test')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        
        self.isTrain = False
        return parser
