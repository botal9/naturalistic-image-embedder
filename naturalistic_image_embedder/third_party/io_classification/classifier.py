# Copyright (c) 2017 Bolei Zhou and MIT CSAIL Computer Vision
# Places project page: http://places2.csail.mit.edu/

import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F

import os
import numpy as np
import cv2
from PIL import Image

SOURCES = {
    'categories_places365.txt':
        'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt',
    'IO_places365.txt': 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt',
    'labels_sunattribute.txt': 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt',
    'W_sceneattribute_wideresnet18.npy':
        'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy',
    'wideresnet18_places365.pth.tar': 'http://places2.csail.mit.edu/models_places365/wideresnet18_places365.pth.tar',
    'wideresnet.py': 'https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py',
}


def full_name(file_name):
    return os.path.join(os.path.split(__file__)[0], file_name)


def ensure_exists(file_name):
    full_file_name = full_name(file_name)
    if not os.access(full_file_name, os.W_OK):
        source_url = SOURCES[file_name]
        os.system(f'wget {source_url} -O {full_file_name}')


# hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module


def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_categories = 'categories_places365.txt'
    ensure_exists(file_name_categories)
    classes = list()
    with open(full_name(file_name_categories)) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    ensure_exists(file_name_IO)
    with open(full_name(file_name_IO)) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    ensure_exists(file_name_attribute)
    with open(full_name(file_name_attribute)) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]

    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    ensure_exists(file_name_W)
    W_attribute = np.load(full_name(file_name_W))

    return classes, labels_IO, labels_attribute, W_attribute


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def returnTF():
    # load the image transformer
    tf = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


features_blobs = []


def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))


def load_model():
    # this model has a last conv feature map as 14x14
    model_file = 'wideresnet18_places365.pth.tar'
    ensure_exists(model_file)

    wideresnet_file = 'wideresnet.py'
    ensure_exists(wideresnet_file)

    from . import wideresnet

    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(full_name(model_file), map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    model.eval()

    # hook the feature extractor
    features_names = ['layer4', 'avgpool']  # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model


def predict_image_attributes(image, verbose=False):
    classes, labels_IO, labels_attribute, W_attribute = load_labels()
    model = load_model()
    tf = returnTF()  # image transformer

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax < 0] = 0

    img = Image.open(image)
    with torch.no_grad():
        input_img = V(tf(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    io_image = np.mean(labels_IO[idx[:10].numpy()])  # vote for the indoor or outdoor
    location = 'indoor' if io_image < 0.5 else 'outdoor'

    responses_attribute = W_attribute.dot(features_blobs[1])
    idx_a = np.argsort(responses_attribute)
    scene_attributes = [labels_attribute[idx_a[i]] for i in range(-1, -10, -1)]
    scene_categories = [(probs[i], classes[idx[i]]) for i in range(5)]

    if verbose:
        print(f'RESULT ON {image}')

        # output the IO prediction
        print(f'--TYPE OF ENVIRONMENT: {location}')

        # output the prediction of scene category
        print('--SCENE CATEGORIES:')
        for i in range(0, 5):
            print(f'{probs[i]:.3f} -> {classes[idx[i]]}')

        # output the scene attributes
        print('--SCENE ATTRIBUTES:')
        print(', '.join(scene_attributes))

        # generate class activation mapping
        print('Class activation map is saved as cam.jpg')
        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

        # render the CAM and output
        img = cv2.imread('image.jpg')
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.4 + img * 0.5
        cv2.imwrite('cam.jpg', result)

    return location, scene_attributes, scene_categories
