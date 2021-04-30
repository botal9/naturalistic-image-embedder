import cv2
import os.path
import torch
import torchvision.transforms.functional as tf
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def read_rgba(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img.astype(np.uint8)

def read_rgb(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return img.astype(np.uint8)


class EmbDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
#         parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0,
                             no_flip=True, preprocess='none')  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_paths = []
        self.isTrain = opt.isTrain
        self.dataset_root = opt.dataset_root
        if opt.isTrain==True:
            print('loading training file: ')
            self.trainfile = os.path.join(opt.dataset_root, 'train.txt')
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(line.rstrip())
        elif opt.isTrain==False:
            print('loading test file')
            self.trainfile = os.path.join(opt.dataset_root, 'test.txt')
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        self.image_paths.append(line.rstrip())
        self.transform = get_transform(opt)
        self.image_size = [1024, 1024]

    def __getitem__(self, index):
        comp_name = self.image_paths[index]
        comp_path = os.path.join(self.dataset_root, 'comp', comp_name)
        
        parts = comp_name.split('_')
        real_name = '_'.join([parts[0], parts[2]]).replace('-', '_').replace('_bg', '');
        real_path = os.path.join(self.dataset_root, 'real', real_name)

        try:
            comp = read_rgba(comp_path)
            real = read_rgb(real_path)
        except Exception as e:
            print('Exception', comp_path, real_path, '\n')
            print(e)
        
        comp = self.transform(comp)
        real = self.transform(real)
        
        real[:,:,3] = comp[:,:,3]  # copy mask

        return {'comp': comp, 'real': real, 'img_path': comp_name}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
