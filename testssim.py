from skimage import data, img_as_float
from skimage.measure import compare_ssim
from PIL import Image
import numpy as np

K = Image.open("digits/K.tif").convert("L")
X = Image.open("digits/X.tif").convert("L")

im = [Image.open("LR_47K1_200.94_0_SR.png.png").convert("L"),
    Image.open("LR_47K1_200.94_10_SR.png.png").convert("L"),
    Image.open("LR_47K1_200.94_11_SR.png.png").convert("L"),
    Image.open("LR_47K1_200.94_12_SR.png.png").convert("L"),
    Image.open("LR_47K1_200.94_13_SR.png.png").convert("L"),
    Image.open("LR_47K1_200.94_14_SR.png.png").convert("L")]

for i in im:    
    print("K:{}".format(compare_ssim(img_as_float(K),img_as_float(i))))
    print("X:{}".format(compare_ssim(img_as_float(X),img_as_float(i))))

