# Server for license plate super-resolution
# python3.5 srserver.py --model ERDN1 --RDNconfig B --scale 3 --std-lr-width 69 --std-lr-height 69 --pre_train ../experiment/P100/ERDN1_ML1_D16C8G64_69x69_all_rot_noblur/model/model_best_50dB.pt

from socket import *
import argparse
import torch
from PIL import Image
from importlib import import_module
import numpy as np
from option import args
import linecache
import sys

import lpt_utils

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('{}({}):\n\t{}\n{}: {}'.format(filename, lineno, line.strip(), exc_type, exc_obj))

def square_to_rect(img):
    if img.size[0] != img.size[1]: # img is not square, then do nothing.
        return img        
    pad = img.size[1] - round(img.size[1]*lpt_utils.lpt_std_size[1]/lpt_utils.lpt_std_size[0])
    y1 = pad//2 + pad%2
    y2 = pad//2
    cropimg = img.crop((0,y1,img.size[0],img.size[1]-y2))
    return cropimg

module = import_module('model.' + args.model.lower())
model = module.make_model(args)
model.load_state_dict(torch.load(args.pre_train))
model = model.cuda()

# Server setup
host = "192.168.1.121"
port = 9999
addr = (host,port)
TCPSock = socket()
TCPSock.bind(addr)

# Receive input
print("Listening...")
TCPSock.listen(2)

while True:
    try:
        c,client_addr = TCPSock.accept()
        height = c.recv(4)
        width = c.recv(4)
        height = int.from_bytes(height, byteorder='little')
        width = int.from_bytes(width, byteorder='little')
        print("recv input size: {}x{}".format(height,width)) # debug
        if height <= 0 or width <= 0:
            continue
        data = c.recv(3*height*width,MSG_WAITALL)        
        print("recv {} bytes of data".format(len(data))) # debug
        data = np.asarray(data)
        data.dtype = np.dtype((np.uint8, (height,width,3)))
        
        print("actual input size: {}".format(data.shape)) #debug
        print("input dtype: {}".format(data.dtype)) #debug    

        # Image.fromarray(data).save("input.png") # debug

        # Input transform
        input = Image.fromarray(data).resize((args.std_lr_width,args.std_lr_height),Image.NEAREST)
        print("input resize to {}".format(input.size)) #debug
        np_transpose = np.ascontiguousarray(np.transpose(input, (2, 0, 1)).reshape(1,3,args.std_lr_height,args.std_lr_width))
        input = torch.Tensor(np_transpose)
        input = input.cuda()

        # Do model
        out = model(input)
        out.clamp_(0, 255)
        out = out.cpu().data[0].numpy()
        out_data = np.uint8(out[0])
        out_data = square_to_rect(Image.fromarray(out_data))

        # Send result back
        out_data = out_data.tobytes()        
        if(c.send(out_data)):
            print("sent {} bytes to {}".format(len(out_data),client_addr))
        c.close()
    except ValueError as e:
        PrintException()
    except TypeError as e:
        PrintException()
    except KeyError as e:
        PrintException()
    except IndexError as e:
        PrintException()
    except ZeroDivisionError as e:
        PrintException()
