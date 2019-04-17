# Super resolve an image
# python3.5 sr.py --model ERDN1 --RDNconfig B --scale 3 --std-lr-width 69 --std-lr-height 69 --pre_train ../experiment/P100/ERDN1_ML1_D16C8G64_69x69_all_rot_noblur/model/model_best_50dB.pt --image /workspace/dataset/lpt_square/test/LR/LR_59N1_744.62.bmp

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

module = import_module('model.' + args.model.lower())
model = module.make_model(args)
model.load_state_dict(torch.load(args.pre_train))
model = model.cuda()

if args.image == '':
    print("args.image is empty. Nothing to do.")
    raise SystemExit
input = Image.open(args.image)
input = input.resize((args.std_lr_width,args.std_lr_height),Image.NEAREST)
np_transpose = np.ascontiguousarray(np.transpose(input, (2, 0, 1)).reshape(1,3,args.std_lr_height,args.std_lr_width))
input = torch.Tensor(np_transpose)
input = input.cuda()

out = model(input)
out.clamp_(0, 255)
out = out.cpu().data[0].numpy()
out_img = np.uint8(out[0])
square_to_rect(Image.fromarray(out_img,mode='L')).save("testsr.png")
