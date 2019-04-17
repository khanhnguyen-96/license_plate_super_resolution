import os

from data import common

import numpy as np
import scipy.misc as misc
from PIL import Image, ImageFilter

import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.scale = args.scale
        self.idx_scale = 0

        # Rotate square image only
        self.rot = True
        if args.std_lr_width != args.std_lr_height:        
            self.rot = False

        if self.args.interp != 'bicubic' and self.args.interp != 'nearest' and self.args.interp != 'lanczos' and self.args.interp != 'bilinear':
            raise ValueError("interp has wrong value.")
            
        self._set_filesystem(args.dir_data)

        def _load_bin():
            self.images_hr = np.load(self._name_hrbin())
            self.images_lr = [
                np.load(self._name_lrbin(s)) for s in self.scale]

        if args.ext == 'img':
            self.images_hr, self.images_lr = self._scan()
        elif args.ext.find('sep') >= 0:
            self.images_hr, self.images_lr = self._scan()
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:
                    hr = misc.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                for si, s in enumerate(self.scale):
                    for v in self.images_lr[si]:
                        lr = misc.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)

            self.images_hr = [
                v.replace(self.ext, '.npy') for v in self.images_hr
            ]
            self.images_lr = [
                [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
                for i in range(len(self.scale))
            ]

        elif args.ext.find('bin') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load_bin()
            except:
                print('Preparing a binary file')
                bin_path = os.path.join(self.apath, 'bin')
                if not os.path.isdir(bin_path):
                    os.mkdir(bin_path)

                list_hr, list_lr = self._scan()
                hr = [misc.imread(f) for f in list_hr]
                np.save(self._name_hrbin(), hr)
                del hr
                for si, s in enumerate(self.scale):
                    lr_scale = [misc.imread(f) for f in list_lr[si]]
                    np.save(self._name_lrbin(s), lr_scale)
                    del lr_scale
                _load_bin()
        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    def __getitem__(self, idx):
        lr, hr, lr_name = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        ts = common.np2Tensor([lr, hr], self.args.rgb_range)

        return [(ts[0],lr_name), ts[1]]

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_name = self.images_lr[self.idx_scale][idx]
        hr_name = self.images_hr[idx]
        if self.args.ext == 'img':
            lr = misc.imread(lr_name)
            hr = misc.imread(hr_name)
        elif self.args.ext.find('sep') >= 0:
            lr = np.load(lr_name)
            hr = np.load(hr_name)

        return lr, hr, os.path.basename(lr_name)

    def target_transform(self, hr, scale, std_lr_width, std_lr_height):
        # Calculate valid size for HR image
        # h,w = hr.shape[:2]
        # if scale == 4:
            # h = (h//3)*scale
            # w = (w//3)*scale
        # s = h - (h % scale), w - (w % scale)
        if self.args.freesize:
            h,w = std_lr_height,std_lr_width
        else:
            h,w = (std_lr_height*scale, std_lr_width*scale)
        return misc.imresize(hr,(h,w),'bicubic');
        
    def input_transform(self, lr, std_lr_width, std_lr_height, gauss_std):
        # Resize to standard size
        std_area = std_lr_width * std_lr_height
        area = lr.shape[0] * lr.shape[1]
        if area > std_area and gauss_std > -1:
            lr = Image.fromarray(lr)
            lr.filter(ImageFilter.GaussianBlur(gauss_std))
            lr = np.array(lr)        
        return misc.imresize(lr,(std_lr_height, std_lr_width),self.args.interp)

    def _get_patch(self, lr, hr):
        #patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        #multi_scale = len(self.scale) > 1
        lr = self.input_transform(lr,self.args.std_lr_width, self.args.std_lr_height, self.args.gauss_std)
        hr = self.target_transform(hr,scale,lr.shape[1],lr.shape[0])        
        if self.train:
            #lr, hr = common.get_patch(
            #    lr, hr, patch_size, scale, multi_scale=multi_scale
            #)
            lr,_ = common.augment([lr,lr], hflip=False, vflip=False, rot90=False, rot=self.rot, shift=False)
            #lr = common.add_noise(lr, self.args.noise)
        #else:
            # ih, iw = lr.shape[0:2]
            # hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
