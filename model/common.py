import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

# Activation function as in "Searching for activation functions", ICLR, 2018.
class Swish(nn.Module):
    def __init__(self, beta=10, precision='single'):
        super(Swish, self).__init__()
        if precision == 'half':
            self.beta = torch.tensor(beta, dtype=torch.float16, device=torch.device("cuda"))
        else:
            self.beta = torch.tensor(beta, dtype=torch.float, device=torch.device("cuda"))

    def forward(self, x):
        return x * F.sigmoid(self.beta * x)

def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

#class MeanShift(nn.Conv2d):
#    def __init__(self, rgb_range, rgb_mean, sign=-1):
#        super(MeanShift, self).__init__(3, 3, kernel_size=1)
#        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
#        self.bias.data = sign * torch.Tensor(rgb_mean) * rgb_range
#        # Freeze the MeanShift layer
#        for params in self.parameters():
#            params.requires_grad = False

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        modules = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feat, 4 * n_feat, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if bn: modules.append(nn.BatchNorm2d(n_feat))
                if act: modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feat, 9 * n_feat, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if bn: modules.append(nn.BatchNorm2d(n_feat))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*modules)

