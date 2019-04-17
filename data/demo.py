import os

from data import common

import numpy as np
import scipy.misc as misc
from PIL import Image, ImageFilter

import torch
import torch.utils.data as data

class Demo(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.name = 'Testlpt'
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = False

        self.filelist = []
        for f in os.listdir(args.dir_testlpt):
            if (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')):
                self.filelist.append(os.path.join(args.dir_testlpt, f))

    def input_transform(self, lr, std_lr_width, std_lr_height, gauss_std):
        # Resize to standard size
        std_area = std_lr_width * std_lr_height
        area = lr.shape[0] * lr.shape[1]
        if (area > std_area):
            lr = Image.fromarray(lr)
            lr.filter(ImageFilter.GaussianBlur(gauss_std))
            lr = np.array(lr)
        return misc.imresize(lr,(std_lr_height, std_lr_width),self.args.interp)

    def __getitem__(self, idx):
        filename = os.path.split(self.filelist[idx])[-1]
        filename, _ = os.path.splitext(filename)
        lr = misc.imread(self.filelist[idx])
        lr = self.input_transform(lr,self.args.std_lr_width, self.args.std_lr_height, self.args.gauss_std)
        lr = common.set_channel([lr], self.args.n_colors)[0]

        return (common.np2Tensor([lr], self.args.rgb_range)[0], ""), filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

