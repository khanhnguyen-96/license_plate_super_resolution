# Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
import torch
import torch.nn as nn
import torch.nn.init as init

def make_model(args, parent=False):
    return ESPCN(args)

class ESPCN(nn.Module):
    def __init__(self, args):
        super(ESPCN, self).__init__()

        scale = args.scale[0]
        
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(args.n_colors, 64, (5, 5), (1, 1), (2, 2))
        #self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, args.n_colors * scale ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(scale)

        self._initialize_weights()

    def forward(self, x):
        x = self.act(self.conv1(x))
        #x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        #init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
