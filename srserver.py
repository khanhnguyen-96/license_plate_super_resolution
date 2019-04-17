# Server for license plate super-resolution
# python3.5 srserver.py --model ERDN1 --RDNconfig B --scale 3 --std-lr-width 69 --std-lr-height 69 --pre_train ../experiment/P100/ERDN1_ML1_D16C8G64_69x69_all_rot_noblur/model/model_best_50dB.pt

from socket import *
import argparse
import torch
from PIL import Image
from importlib import import_module
import numpy as np
from option import args

import lpt_utils

def square_to_rect(img):
    if img.size[0] != img.size[1]: # img is not square, then do nothing.
        return img        
    pad = img.size[1] - round(img.size[1]*lpt_utils.lpt_std_size[1]/lpt_utils.lpt_std_size[0])
    y1 = pad//2 + pad%2
    y2 = pad//2
    cropimg = img.crop((0,y1,img.size[0],img.size[1]-y2))
    return cropimg

height = args.std_lr_height
width = args.std_lr_width

module = import_module('model.' + args.model.lower())
model = module.make_model(args)
model.load_state_dict(torch.load(args.pre_train))
model = model.cuda()

# Server setup
host = "192.168.1.121"
port = 9999
addr = (host,port)

UDPSock = socket(AF_INET,SOCK_DGRAM)
UDPSock.bind(addr)

# Receive input
print("Listening...")
height,client_addr = UDPSock.recvfrom(4)
width,_ = UDPSock.recvfrom(4)
height = int.from_bytes(height, byteorder='little')
width = int.from_bytes(width, byteorder='little')
data,_ = UDPSock.recvfrom(3*height*width)
data = np.asarray(data)
data.dtype = np.dtype((np.uint8, (3,height,width)))

print(height) #debug
print(width) #debug
print(data.shape) #debug
print(data.dtype) #debug

data = np.transpose(data, (1,2,0))

# Input transform
input = Image.fromarray(data).resize((width,height),Image.NEAREST)
np_transpose = np.ascontiguousarray(np.transpose(input, (2, 0, 1)).reshape(1,3,height,width))
input = torch.Tensor(np_transpose)
input = input.cuda()

# Do model
out = model(input)
out.clamp_(0, 255)
out = out.cpu().data[0].numpy()
out_data = np.uint8(out[0]).tobytes()

# Send result back
print(client_addr) #debug
if(UDPSock.sendto(out_data,client_addr)):
    print("sent {}".format(len(out_data)))
UDPSock.close()
