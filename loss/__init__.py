from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_ssim
import ML1Loss

class loss:
    def __init__(self, args):
        self.args = args
        if args.freesize:
            self.scale = 1
        else:
            self.scale = self.args.scale[0]

    def get_loss(self):
        print('Preparing loss function...')

        my_loss = []
        losslist = self.args.loss.split('+')
        for loss in losslist:
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'SmoothL1Loss':
                loss_function = nn.SmoothL1Loss()
            elif loss_type == 'CosineEmbeddingLoss':
                loss_function = nn.CosineEmbeddingLoss()
            elif loss_type == 'SSIM':
                loss_function = pytorch_ssim.SSIM()
            elif loss_type == 'ML1':               
                loss_function = ML1Loss.ML1Loss(self.args.std_lr_height*self.scale, self.args.std_lr_width*self.scale)

            my_loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function})

        if len(losslist) > 1:
            my_loss.append({
                'type': 'Total',
                'weight': 0,
                'function': None})

        print(my_loss)

        return my_loss
