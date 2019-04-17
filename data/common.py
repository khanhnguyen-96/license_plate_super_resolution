import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st
import scipy.misc as misc

import torch
from torchvision import transforms

def get_patch(img_in, img_tar, patch_size, scale, multi_scale=False):
    ih, iw = img_in.shape[:2]

    p = scale if multi_scale else 1
    tp = int(p * patch_size)
    ip = int(tp // scale)

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    # print(ih,iw,p,scale,tp,ip,ix,iy,tx,ty) #debug
    # img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    # img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    img_in = img_in[iy:iy + ip, ix:ix + ip]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp]

    # print("img_in.shape ",img_in.shape) # debug
    return img_in, img_tar

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            # img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
            img = np.expand_dims(sc.rgb2gray(img), 2)            
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):        
        # (height, width, channels) to (channels, height, width)
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))        
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(_l) for _l in l]

def add_noise(x, noise='.'):
    if noise is not '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            gaussian_noise = np.random.normal(scale=noise_value, size=x.shape)
            gaussian_noise = gaussian_noise.round()
            x_noise = x.astype(np.int16) + gaussian_noise.astype(np.int16)
            x_noise = x_noise.clip(0, 255).astype(np.uint8)
            return x_noise

    return x


def augment(l, hflip=True, vflip=True, rot90=True, rot=True, shift=True):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot90 and random.random() < 0.5
    k = (random.random()*10)%3+1
    shift = shift and random.random() < 0.5
    rot = rot and random.random() < 0.5

    def shiftOxy(img, ox, oy):
        non = lambda s: s if s<0 else None
        mom = lambda s: max(0,s)
        shift_img = np.zeros_like(img)
        shift_img[mom(oy):non(oy), mom(ox):non(ox)] = img[mom(-oy):non(-oy), mom(-ox):non(-ox)]
        return shift_img

    def _augment(img):
        if hflip: img = img[:, ::-1]
        if vflip: img = img[::-1, :]
        if rot90: img = np.rot90(img,k)
        if shift:
            if random.random() < 0.5:
                r = random.random()
                if r < 0.5:
                    img = shiftOxy(img,-1,0) # shift left
                elif r > 0.5:
                    img = shiftOxy(img,1,0)  # shift right
            else:
                r = random.random()
                if r < 0.5:
                    img = shiftOxy(img,0,-1) # up
                elif r > 0.5:
                    img = shiftOxy(img,0,1)  # down

        if rot:
            n = random.random() * 80 - 40
            img = misc.imrotate(img,n,'bicubic')

        return img

    return [_augment(_l) for _l in l]
