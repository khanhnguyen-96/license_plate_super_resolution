# Enhanced Residual Network for License plate Super-Resolution (ERN4LSR) version 1.0
# Implemented by Nguyen Tran Toan, Oct 02 2018.
#from torchvision import utils

#from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return ERN4LSR(args)

class PixelSplit(nn.Module):
    def __init__(self, inChannels, scale):
        super(PixelSplit, self).__init__()

        self.conv = nn.Conv2d(inChannels, inChannels * scale * scale, scale, stride=scale, groups=inChannels)

    def forward(self, x):
        return self.conv(x)

class Core(nn.Module):
    def __init__(self, inChannels, numConvs, kSize=3):
        super(Core, self).__init__()

        self.numConvs = numConvs

        self.conv_0 = nn.Sequential(*[
            nn.Conv2d(inChannels, inChannels, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
            # common.Swish()
        ])
        
        self.convs = nn.ModuleList()
        for i in range(numConvs):
            self.convs.append(nn.Sequential(*[
                nn.Conv2d(inChannels+inChannels, inChannels, kSize, padding=(kSize-1)//2, stride=1),
                nn.ReLU()
                # common.Swish()
            ]))

    def forward(self, x):
        y0 = self.conv_0(x)
        x = torch.cat((x, y0), 1)
        for i in range(self.numConvs):
            y = self.convs[i](x)
            x = torch.cat((y, y0), 1)
            y0 = y
        return x

class ERN4LSR(nn.Module):
    def __init__(self, args):
        super(ERN4LSR, self).__init__()

        self.r = args.scale[0]
        n_colors = args.n_colors
        kSize = 3 #args.kSize
        n_feats = args.n_feats
        n_convs = args.n_convs

        # Pixel splitter
        self.pixelSplit = PixelSplit(n_colors,self.r)

        self.transition = nn.Conv2d(n_colors * self.r * self.r, n_feats, kSize, padding=(kSize-1)//2, stride=1)
        
        # Core conv
        self.core = Core(n_feats, n_convs, kSize)

        # Up-sampling net
        if self.r == 2 or self.r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(n_feats+n_feats, n_feats * self.r * self.r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(self.r),
                nn.Conv2d(n_feats, n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif self.r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(n_feats+n_feats, n_feats * self.r * self.r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_feats * self.r * self.r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        x = self.pixelSplit(x)
        x = self.core(self.transition(x))
        return self.UPNet(x)
