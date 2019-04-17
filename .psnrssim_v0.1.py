from os import path
import argparse
from PIL import Image
from skimage import data, img_as_float
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from str_sim import search_filename

import matplotlib.pyplot as plt
plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Compute PNSR SSIM, print to standard output')
parser.add_argument('--datasettype', type=str, required=True, help='realreal, realsimu, simureal, simusimu')
parser.add_argument('--testdir', type=str,  required=True, help='Test directory')
parser.add_argument('filenames', metavar='filename', type=str, nargs='+', help='input images to be compute PSNR and SSIM.')
opt = parser.parse_args()

avgpsnr=0
avgssim=0
for filename in opt.filenames:
    img = Image.open(filename).convert('L')

    if opt.datasettype == "realreal" or opt.datasettype == "realsimu":
        srchdir = path.join(opt.testdir,"OR")
    elif opt.datasettype == "simureal" or opt.datasettype == "simusimu":
        srchdir = path.join(opt.testdir,"HRS")
    else:
        raise ValueError("opt.datasettype has wrong value")
    comparename = path.join(srchdir,search_filename(path.basename(filename),srchdir))
    compareimg = Image.open(comparename).convert('L')
    if compareimg.size[0] != img.size[0] or compareimg.size[1] != img.size[1] :
        compareimg = compareimg.resize(img.size,Image.BICUBIC)
    compareimg = img_as_float(compareimg)
    img = img_as_float(img)
    psnr = compare_psnr(img,compareimg)
    avgpsnr += psnr
    ssim = compare_ssim(compareimg,img,data_range=img.max() - img.min())
    avgssim += ssim
    print("{0}: {1}: {2:.2f} / {3:.2f}".format(path.basename(comparename),path.basename(filename),psnr,ssim))
print("avgpsnr / avgssim: {0:.2f} / {1:.2f}".format(avgpsnr/len(opt.filenames),avgssim/len(opt.filenames)))

