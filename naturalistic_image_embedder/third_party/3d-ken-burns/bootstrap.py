import torch
import torchvision

import base64
import cupy
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
import glob
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy
import os
import random
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import urllib
import zipfile

##########################################################

assert (int(str('').join(torch.__version__.split('.')[0:2])) >= 12)  # requires at least pytorch version 1.2.0

##########################################################

objCommon = {}

_main_folder = os.path.split(__file__)[0]
_models_folder = os.path.join(_main_folder, 'models')

exec(open(f'{_main_folder}/common.py', 'r').read())

exec(open(f'{_models_folder}/disparity-estimation.py', 'r').read())
exec(open(f'{_models_folder}/disparity-adjustment.py', 'r').read())
exec(open(f'{_models_folder}/disparity-refinement.py', 'r').read())
exec(open(f'{_models_folder}/pointcloud-inpainting.py', 'r').read())

##########################################################


def _depth_map(npyImage):
    torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

    fltFocal = max(npyImage.shape[0], npyImage.shape[1]) / 2.0
    fltBaseline = 40.0

    tenImage = torch.FloatTensor(numpy.ascontiguousarray(
        npyImage.transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
    tenDisparity = disparity_estimation(tenImage)
    tenDisparity = disparity_adjustment(tenImage, tenDisparity)
    tenDisparity = disparity_refinement(
        torch.nn.functional.interpolate(input=tenImage, size=(tenDisparity.shape[2] * 4, tenDisparity.shape[3] * 4),
                                        mode='bilinear', align_corners=False), tenDisparity)
    tenDisparity = torch.nn.functional.interpolate(input=tenDisparity, size=(tenImage.shape[2], tenImage.shape[3]),
                                                   mode='bilinear', align_corners=False) * (
                               max(tenImage.shape[2], tenImage.shape[3]) / 256.0)
    tenDepth = (fltFocal * fltBaseline) / (tenDisparity + 0.0000001)

    npyDisparity = tenDisparity[0, 0, :, :].cpu().numpy()
    npyDepth = tenDepth[0, 0, :, :].cpu().numpy()
    depthMap = (npyDisparity / fltBaseline * 255.0).clip(0.0, 255.0).astype(numpy.uint8)

    return depthMap, npyDepth
