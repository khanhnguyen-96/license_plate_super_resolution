import argparse
from PIL import Image

parser = argparse.ArgumentParser(description='Resize images')
parser.add_argument('--w', type=int,  default=69, help='width')
parser.add_argument('--h', type=int,  default=48, help='height')
parser.add_argument('--o', action='store_true',help='overwrite')
parser.add_argument('filenames', metavar='filename', type=str, nargs='+', help='input images.')
opt = parser.parse_args()

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"])

for filename in opt.filenames:
    if is_image_file(filename):
        img = Image.open(filename).resize((opt.w,opt.h))
        if opt.o:
            img.save(filename)
        else:
            img.save("{}_res.png".format(filename))

