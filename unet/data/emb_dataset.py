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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)

def read_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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
        file_type = 'train' if self.isTrain else 'test'
        print(f'loading {file_type} file')
        self.file = os.path.join(opt.dataset_root, f'{file_type}.txt')
        with open(self.file, 'r') as f:
                for line in f.readlines():
                    self.image_paths.append(line.rstrip())
        self.transform = get_transform(opt)
        self.image_size = [256, 256]

    def __getitem__(self, index):
        real_name = self.image_paths[index]
        real_path = os.path.join(self.dataset_root, 'real', real_name)
        comp_name = real_name.replace('_r', '_c');
        comp_path = os.path.join(self.dataset_root, 'comp', comp_name)
        mask_name = real_name.replace('_r', '_m');
        mask_path = os.path.join(self.dataset_root, 'mask', mask_name)

        comp = read_rgb(comp_path)
        real = read_rgb(real_path)
        mask = read_mask(mask_path)
        
        comp = self.norm3(comp)
        real = self.norm3(real)
        mask = np.expand_dims(mask, axis=0)
        
        return {'comp': comp, 'real': real, 'mask': mask, 'img_path': comp_name}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
