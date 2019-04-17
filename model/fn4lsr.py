# Fast Network for License plate Super-Resolution
# Implemented by Nguyen Tran Toan, Nov 11 2018.


import torch
import torch.nn as nn


def make_model(args, parent=False):
    return FN4LSR(args)

class PixelSplit(nn.Module):
    def __init__(self, inChannels, scale):
        super(PixelSplit, self).__init__()

        self.conv = nn.Conv2d(inChannels, inChannels * scale * scale, scale, stride=scale, groups=inChannels)

    def forward(self, x):
        return self.conv(x)

class Block(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(inchannels, outchannels, (5,3), padding=(2,1), stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outchannels, outchannels, (3,5), padding=(1,2), stride=1)

    def forward(self, x):
        y  = self.relu(self.conv1(x))
        y2 = self.relu(self.conv2(y)) 
        return torch.cat((y2,y),1)

class FN4LSR(nn.Module):
    def __init__(self, args, kSize = 3):
        super(FN4LSR, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        G = 64
        B = args.n_resblocks

        self.square = args.std_lr_height == args.std_lr_width

        # Pixel splitter        
        self.pixelSplit = PixelSplit(args.n_colors,r)        

        # Blocks
        blocks = []
        if self.square:
            blocksChannels = args.n_colors*r*r        
        else:            
            blocks.append(Block(args.n_colors,G0))
            blocksChannels = G0*2
        for c in range(B):
            blocks.append(Block(blocksChannels,blocksChannels))
            blocksChannels *= 2
        self.blocks = nn.Sequential(*blocks)

        # Up-sampling net
        if r == 1 or r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(blocksChannels, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(blocksChannels, blocksChannels * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(blocksChannels, blocksChannels * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(blocksChannels, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 1 or 2 or 3 or 4.")

    def forward(self, x):
        if self.square:
            x = self.pixelSplit(x)
        return self.UPNet(self.blocks(x))
