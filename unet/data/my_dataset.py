import cv2
import os.path
import torch
import torchvision.transforms.functional as F
from data.base_dataset import BaseDataset, get_transform, read_ldr, read_hdr, read_mask
from PIL import Image
import numpy as np
import torchvision.transforms as tf


class MyDataset(BaseDataset):
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
        file_type = 'train' if self.isTrain else 'test'
        print(f'loading {file_type} file')
        self.file = os.path.join(opt.dataset_root, f'validated_{file_type}.txt')
        with open(self.file, 'r') as f:
                for line in f.readlines():
                    self.image_paths.append(os.path.join(opt.dataset_root, line.rstrip()))
        self.transform = get_transform(opt)
        self.image_size = (256, 256)
        
#         self.print_networks(True)
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = self.image_paths[index]
        name_parts = path.split('_')
        mask_path = self.image_paths[index].replace('composite_images','masks')
        mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
        target_path = self.image_paths[index].replace('composite_','real_')
        target_path = target_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')
        depth_path = self.image_paths[index].replace('composite_','depth_').replace('jpg', 'png')

        comp = read_ldr(path)
        real = read_ldr(target_path)
        mask = read_mask(mask_path)
        depth = read_mask(depth_path)
            
        comp = cv2.resize(comp, self.image_size)
        mask = cv2.resize(mask, self.image_size)
        real = cv2.resize(real, self.image_size)
        depth = cv2.resize(depth, self.image_size)

        comp = self.norm3(comp)
        mask = F.to_tensor(mask)
        real = self.norm3(real)
        depth = self.norm1(depth)

        return {'comp': comp, 'real': real, 'img_path': path, 'mask': mask, 'depth': depth}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
