# Enhanced Residual Dense Network for license plate super-resolution
# Use add instead of cat
# Implemented by Nguyen Tran Toan, Nov 10 2018.

from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return ERDNP(args)

class PixelSplit(nn.Module):
    def __init__(self, inChannels, scale):
        super(PixelSplit, self).__init__()

        self.conv = nn.Conv2d(inChannels, inChannels * scale * scale, scale, stride=scale, groups=inChannels)

    def forward(self, x):
        return self.conv(x)

class Cell(nn.Module):
    def __init__(self, channels, kSize=3):
        super(Cell, self).__init__()
        Cin = Cout = channels        
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, Cout, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()            
        ])

    def forward(self, x):        
        return x + self.conv(x) * 0.1

class RDB(nn.Module):
    def __init__(self, channels, nConvLayers, kSize=3):
        super(RDB, self).__init__()        

        convs = []
        for c in range(nConvLayers):
            convs.append(Cell(channels))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):         
        return self.convs(x) + x

class ERDNP(nn.Module):
    def __init__(self, args):
        super(ERDNP, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (32, 16, 128)
        }[args.RDNconfig]

        # Pixel splitter
        self.pixelSplit = PixelSplit(args.n_colors,r)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors*r*r, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(channels = G0, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        if r == 1 or r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 1 or 2 or 3 or 4.")

    def forward(self, x):
        x = self.pixelSplit(x)
        
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        return self.UPNet(x)
