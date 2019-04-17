# Crop squared license plate image to rectangle
#
import argparse
import numpy as np
from os import listdir
from os.path import join, basename, isfile, splitext
from PIL import Image
import lpt_utils

parser = argparse.ArgumentParser(description='Crop square image to rectangle')
parser.add_argument("--image-dir", required=True, type=str, help="Images directory")
parser.add_argument("--save-dir", required=True, type=str, help="Save directory")
parser.add_argument("--o", action='store_true', help="Overwrite")
opt = parser.parse_args()

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".JPG", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath)
    return img


class RectBoundingBox():
    def __init__(self, image_dir, overwrite=False):
        super(RectBoundingBox, self).__init__()

        self.overwrite = overwrite
        self.image_dir = image_dir
        self.filenames = [f for f in listdir(image_dir) if is_image_file(join(image_dir, f))]

    def Crop(self):
        for filename in self.filenames:            
            img = load_img(join(self.image_dir,filename))
            if img.size[0] != img.size[1]: # img is not square, then do nothing.
                raise SystemExit
            
            pad = img.size[1] - round(img.size[1]*lpt_utils.lpt_std_size[1]/lpt_utils.lpt_std_size[0])
            y1 = pad//2 + pad%2
            y2 = pad//2
        
            cropimg = img.crop((0,y1,img.size[0],img.size[1]-y2))
            if self.overwrite:
                fullname = join(self.image_dir,filename)
            else:
                fullname = join(opt.save_dir,splitext(filename)[0] + ".png")

            cropimg.save(fullname)
            
bdb = RectBoundingBox(opt.image_dir)
bdb.Crop()
