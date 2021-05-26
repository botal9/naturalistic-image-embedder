import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import ntpath
import time
import torch

DISPLAY_SIZE = (256, 256)

def to_img(tensor):
    np_img = np.squeeze(tensor.detach().to('cpu').numpy())
    np_img = np_img.transpose((1, 2, 0))
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
    np_img = cv2.resize(np_img, DISPLAY_SIZE)
    return np_img


class LocalVisualizer:
    def __init__(self, opt):
        self.losses = {}
        self.losses['train'] = []
        self.losses['test'] = []
        self.phase = 'train'
        self.isTrain = True
        # create a logging files to store training losses
        now = time.strftime('%c')
        self.train_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'train_loss_log.txt')
        with open(self.train_log_name, 'w') as log_file:
            log_file.write(f'=============== Train Loss ({now}) ================\n')
        self.test_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'test_loss_log.txt')
        with open(self.test_log_name, 'w') as log_file:
            log_file.write(f'================ Test Loss ({now}) ================\n')

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.epochs = np.arange(1, epoch + 1)
    
    def set_phase(self, phase):
        self.phase = phase
        self.isTrain = phase == 'train'
    
    def log(self, message):
        log_file_name = self.train_log_name if self.isTrain else self.test_log_name
        print(message)  # print the message
        with open(log_file_name, 'a') as log_file:
            log_file.write('%s\n' % message)  # save the message
    
    def add_epoch_loss(self, loss):
        self.losses[self.phase].append(loss)
        message = f'({self.phase.capitalize()} epoch: {self.epoch}, total) {loss:.3f}'
        self.log(message)
            
    def print_current_losses(self, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = f'({self.phase.capitalize()} epoch: {self.epoch}, iters: {iters}, time: {t_comp:.3}, data: {t_data:.3}) '
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        self.log(message)
    
    def display_visuals(self, visuals):
        if len(visuals['real'].shape) == 4:
            visuals['real'] = visuals['real'][0]
            visuals['comp'] = visuals['comp'][0]
            visuals['harmonized'] = visuals['harmonized'][0]
            
        real = to_img(visuals['real'])
        comp = to_img(visuals['comp'])
        harmonized = to_img(visuals['harmonized'])

        print(f'{self.phase.capitalize()} epoch {self.epoch}: composition/real/harmonized')
        display = np.concatenate([comp, real, harmonized], axis=1)
        fig = plt.figure(figsize=(15, 15))
        plt.imshow(display)
        plt.show()
    
    
    def plot_epoch_losses(self):
        fig = plt.figure(figsize=(10, 10))
        plt.plot(self.epochs, self.losses['train'], label='Train', color='blue')
        plt.plot(self.epochs, self.losses['test'], label='Test', color='red')
        plt.legend()
        plt.show()