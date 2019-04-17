import numpy as np
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
parser.add_argument('--lrext', type=str,  default=".bmp", help='LR image file extension')
parser.add_argument('--srext', type=str,  default=".png", help='SR image file extension')
parser.add_argument('--datasettype', type=str, required=True, help='realreal, realsimu, simureal, simusimu')
parser.add_argument('--testdir', type=str,  required=True, help='Test directory contains folders HRS and LR')
parser.add_argument('--save-dir', type=str,  required=True, help='Directory to save result. If file exists, it will be overwritten.')
parser.add_argument('filenames', metavar='filename', type=str, nargs='+', help='input images to be compute PSNR and SSIM.')
opt = parser.parse_args()

def sort2list(list1, list2):
    return (list(t) for t in zip(*sorted(zip(list1, list2))))

def sort3list(list1, list2, list3):
    return (list(t) for t in zip(*sorted(zip(list1, list2, list3))))

def save_stc(filename, axis, data):
    with open(filename, "w") as f:        
        f.write("Count: {}.\n".format(len(axis)))
        f.write("Total: {}.\n\n".format(sum(data)))
        d, a = sort2list(data, axis)
        for i,ax in enumerate(a):
            f.write("{0}:{1}\n".format(ax,d[i]))

avgpsnr=0
avgssim=0
testresult = []
ssim_size_count = {}
ssim_size_sum = {}
ssim_count = {}
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
    ssim = round(ssim,2)
    avgssim += ssim
    testresult.append("{0}: {1}: {2:.2f} / {3:.2f}\r\n".format(path.basename(comparename),path.basename(filename),psnr,ssim))
    
    lr = Image.open(path.join(opt.testdir,"LR",path.basename(filename.replace('_SR'+opt.srext,opt.lrext))))
    ssim_size_count[lr.size] = ssim_size_count.get(lr.size, 0) + 1;
    ssim_size_sum[lr.size] = ssim_size_sum.get(lr.size, 0) + ssim;
    
    ssim_count[ssim] = ssim_count.get(ssim, 0) + 1;

# Calculate the average SSIM by image size
for k in ssim_size_count.keys():
    ssim_size_count[k] = ssim_size_sum[k]/ssim_size_count[k]

testresult.append("avgpsnr / avgssim: {0:.2f} / {1:.2f}\n".format(avgpsnr/len(opt.filenames),avgssim/len(opt.filenames)))

# Save test result to file
#
with open(path.join(opt.save_dir,"testlog.txt"), "w") as f:    
    f.writelines(testresult)

# Display ssim_size_count
#
_, axis, data = sort3list([w*h for w,h in ssim_size_count.keys()], ["{0} x {1}".format(w,h) for w,h in ssim_size_count.keys()], list(ssim_size_count.values()))
figwidth = 22
fig = plt.figure(figsize=(figwidth,figwidth*9/16))
plt.title("Chỉ số SSIM trung bình theo kích thước", fontsize=24)
plt.xlabel("Kích thước của ảnh (pixel)", fontsize=20)
plt.ylabel("Chỉ số SSIM trung bình", fontsize=20)
plt.scatter(axis, data)
plt.grid(True)
ax0 = fig.axes[0]
for l in ax0.xaxis.get_ticklabels():
    l.set_rotation('vertical')
    l.set_fontsize(11)
for l in ax0.yaxis.get_ticklabels():
    l.set_fontsize(11)
ax0.autoscale(True,'x',True)
ax0.autoscale(True,'y',True)
plt.plot(axis, data)
plt.tight_layout()
plt.savefig(path.join(opt.save_dir,"avg_ssim_size.jpg"))
plt.close(fig)

save_stc(path.join(opt.save_dir,"avg_ssim_size.txt"), axis, data)

# Display ssim_count
#
axis, data = sort2list(list(ssim_count.keys()), list(ssim_count.values()))
figwidth = 20
fig = plt.figure(figsize=(figwidth,figwidth*9/16))
plt.title("Số lượng ảnh theo chỉ số SSIM", fontsize=24)
plt.xlabel("SSIM", fontsize=20)
plt.ylabel("Số lượng ảnh", fontsize=20)
plt.scatter(axis, data)
plt.grid(True)
ax0 = fig.axes[0]
for l in ax0.xaxis.get_ticklabels():    
    l.set_fontsize(16)
for l in ax0.yaxis.get_ticklabels():
    l.set_fontsize(16)
ax0.autoscale(True,'x',True)
ax0.autoscale(True,'y',True)
plt.plot(axis, data)
plt.savefig(path.join(opt.save_dir,"ssim_count.jpg"))
plt.close(fig)

save_stc(path.join(opt.save_dir,"ssim_count.txt"), axis, data)
