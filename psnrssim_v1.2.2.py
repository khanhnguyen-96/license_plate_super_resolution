import numpy as np
from os import path, listdir
import argparse
from PIL import Image
from skimage import data, img_as_float
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from str_sim import search_filename
from lpt_utils import lptnum_of
from plot_utils import autolabel, sort2list, sort3list, save_stc
import operator

import matplotlib.pyplot as plt
plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Compute PNSR, SSIM')
parser.add_argument('--lrext', type=str,  default=".bmp", help='LR image file extension')
parser.add_argument('--srext', type=str,  default=".png", help='SR image file extension')
parser.add_argument('--datasettype', type=str, required=True, help='realreal, realsimu, simureal, simusimu')
parser.add_argument('--digit-dir', type=str,  default="digits", help='Directory contains digit images')
parser.add_argument('--testdir', type=str,  required=True, help='Test directory contains folders HRS and LR')
parser.add_argument('--save-dir', type=str,  required=True, help='Directory to save result. If file exists, it will be overwritten.')
parser.add_argument("--sep", type=str, default="_", help='"_" or "-", char concat lpt_num and a number that distinguish filenames have the same lpt_num.')
parser.add_argument('--ssimT', type=float,  default=0.82, help='Count SSIM > ssimT (Default is 0.82)')
parser.add_argument('filenames', metavar='filename', type=str, nargs='+', help='input images to be compute PSNR and SSIM.')
opt = parser.parse_args()

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"])

# Load digit images
digits = [[f.split(".")[:-1][0], img_as_float(Image.open(path.join(opt.digit_dir, f)).convert('L'))] for f in listdir(opt.digit_dir) if is_image_file(path.join(opt.digit_dir, f))]
digits = sorted(digits)

# Standard parameters of license plate in pixel
lpt_std_size = (70,50)
char_size = (9,21)
Cx = [10,2,10,5]
Cy = [2,4]
Dx = [3,3,9]

# Compute PNSR, SSIM
avgpsnr = 0
avgssim = 0
total_corr_recog_img = 0
total_corr_recog_lpt = 0
corr_recog_size_count = {}
corr_recog_char_count = {}
corr_num_char_count = {}
lpt_unique = {}
testresult = []
char_count = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, 'A':0, 'B':0, 'C':0, 'D':0, 'E':0, 'F':0, 'G':0, 'H':0, 'K':0, 'L':0, 'M':0, 'N':0, 'P':0, 'R':0, 'S':0, 'T':0, 'U':0, 'V':0, 'X':0, 'Y':0, 'Z':0}
# 01234
# 56789
char_pos_count = {0:char_count.copy(), 1:char_count.copy(), 2:char_count.copy(), 3:char_count.copy(), 4:char_count.copy(), 5:char_count.copy(), 6:char_count.copy(), 7:char_count.copy(), 8:char_count.copy(), 9:char_count.copy()}
corr_recog_char_pos_count = {0:char_count.copy(), 1:char_count.copy(), 2:char_count.copy(), 3:char_count.copy(), 4:char_count.copy(), 5:char_count.copy(), 6:char_count.copy(), 7:char_count.copy(), 8:char_count.copy(), 9:char_count.copy()}
ssim_char_sum = {}
ssimT_char_count = {}
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
    if compareimg.size[0] != lpt_std_size[0] or compareimg.size[1] != lpt_std_size[1]:
        compareimg = compareimg.resize(lpt_std_size,Image.BICUBIC)
    if compareimg.size[0] != img.size[0] or compareimg.size[1] != img.size[1]:
        img = img.resize(compareimg.size,Image.BICUBIC)
    compareimg = img_as_float(compareimg)
    img = img_as_float(img)

    # Calculate SSIM for each character
    #
    # Extract character image
    #    
    # The upper-left corner of characters
    L1 = [(Cx[0],Cy[0]-1)]
    L1.append((L1[0][0] + char_size[0] + Cx[1], Cy[0]-1))
    L1.append((L1[1][0] + char_size[0] + Cx[2]+1, Cy[0]-1))
    L1.append((L1[2][0] + char_size[0] + Cx[1], Cy[0]-1))
    L1.append((Cx[0], Cy[0] + char_size[1] + Cy[1]))
    L1.append((L1[4][0] + char_size[0] + Cx[3], L1[4][1]))
    L1.append((L1[5][0] + char_size[0] + Cx[3], L1[4][1]))
    L1.append((L1[6][0] + char_size[0] + Cx[3], L1[4][1]))
    L2 = [(Dx[0], Cy[0] + char_size[1] + Cy[1])]
    L2.append((L2[0][0] + char_size[0] + Dx[1], L2[0][1]))
    L2.append((L2[1][0] + char_size[0] + Dx[1], L2[0][1]))
    L2.append((L2[2][0] + char_size[0] + Dx[2], L2[0][1]))
    L2.append((L2[3][0] + char_size[0] + Dx[1], L2[0][1]))

    # print(path.basename(filename)) # debug
    lpt_num = lptnum_of(path.basename(filename), sep=opt.sep)
    lptnum = lpt_num.split("_")
    lptnum[1] = lptnum[1].replace(".","")

    if len(lptnum[0]) == 5: # e.g. 59MĐ1_123.45
        raise NotImplementedError
    elif len(lptnum[0]) == 3: # e.g. 59F_123.45
        raise NotImplementedError
    elif len(lptnum[0]) == 4: # e.g. 59Z1_123.45
        img_chars = [img[L1[i][1]:char_size[1]+L1[i][1], L1[i][0]:char_size[0]+L1[i][0]] for i in range(len(lptnum[0]))]
        compareimg_chars = [compareimg[L1[i][1]:char_size[1]+L1[i][1], L1[i][0]:char_size[0]+L1[i][0]] for i in range(len(lptnum[0]))]
    else:
        raise ValueError("len(lptnum[0]) != 3 or 4 or 5")

    if len(lptnum[1]) == 4: # e.g. 59Z1_1234
        img_chars.extend([img[L1[i+len(lptnum[0])][1]:L1[i+len(lptnum[0])][1]+char_size[1], L1[i+len(lptnum[0])][0]:L1[i+len(lptnum[0])][0]+char_size[0]] for i in range(len(lptnum[1]))])
        compareimg_chars.extend([compareimg[L1[i+len(lptnum[0])][1]:L1[i+len(lptnum[0])][1]+char_size[1], L1[i+len(lptnum[0])][0]:L1[i+len(lptnum[0])][0]+char_size[0]] for i in range(len(lptnum[1]))])
    elif len(lptnum[1]) == 5: # e.g. 59Z1_123.45
        img_chars.extend([img[L2[i][1]:L2[i][1]+char_size[1], L2[i][0]:L2[i][0]+char_size[0]] for i in range(len(lptnum[1]))])
        compareimg_chars.extend([compareimg[L2[i][1]:L2[i][1]+char_size[1], L2[i
        ][0]:L2[i][0]+char_size[0]] for i in range(len(lptnum[1]))])
    else:
        raise ValueError("len(lptnum[1]) != 4 or 5")    

    # Calculate PSNR, SSIM, lpt recognition
    lpt_num_seq = ''.join(lptnum)
    lpt_recog_seq = ""    

    for i,im in enumerate(img_chars):
        if lpt_num_seq[i] != 'E': # debug
            continue        
        sssss = "{0}_{1}".format(path.basename(filename),lpt_num_seq[i]) # debug
        print(sssss) # debug
        Image.fromarray(im*255).convert("L").save("{}.png".format(sssss)) # debug
        Image.fromarray(compareimg_chars[i]*255).convert("L").save("{0}_{1}_com.png".format(lpt_num_seq,i)) # debug        
        
        ssim = compare_ssim(compareimg_chars[i],im,dynamic_range=max(compareimg_chars[i].max(),im.max())-min(compareimg_chars[i].min(),im.min()))
        ssim = round(ssim if ssim > 0 else 0,2)

        # Count char and add ssim to calculate average SSIM
        char_count[lpt_num_seq[i]] = char_count.get(lpt_num_seq[i], 0) + 1
        ssim_char_sum[lpt_num_seq[i]] = ssim_char_sum.get(lpt_num_seq[i], 0) + ssim

        # Count ssim > ssimT
        if ssim > opt.ssimT:
            ssimT_char_count[lpt_num_seq[i]] = ssimT_char_count.get(lpt_num_seq[i], 0) + 1
        else:
            ssimT_char_count[lpt_num_seq[i]] = ssimT_char_count.get(lpt_num_seq[i], 0)

        max_s = 0.0
        max_c = "?"
        if i == 2:
            std_chars = digits[10:] # alphabet
        elif len(lptnum[0]) == 4 and i == 3:
            std_chars = digits
        else:
            std_chars = digits[:10] # digit
        lstd = [] # debug
        for d in std_chars:
            s = compare_ssim(d[1],im)#,dynamic_range=max(d[1].max(),im.max())-min(d[1].min(),im.min()))            
            if max_s < s:
                max_s = s
                max_c = d[0]
            # debug
            lstd.append([d[0],round(s,2)])
        print(sorted(lstd,key = operator.itemgetter(1)))
            # end debug
        lpt_recog_seq += max_c        

        # Count correct recognized char
        if lpt_num_seq[i] == max_c:
            corr_recog_char_count[lpt_num_seq[i]] = corr_recog_char_count.get(lpt_num_seq[i], 0) + 1
        else:
            corr_recog_char_count[lpt_num_seq[i]] = corr_recog_char_count.get(lpt_num_seq[i], 0)                   
    continue # debug
    
    dot = "." if len(lptnum[1]) > 4 else ""
    lpt_recog = lpt_recog_seq[:len(lptnum[0])] + "_" + lpt_recog_seq[len(lptnum[0]):-2] + dot + lpt_recog_seq[-2:]    

    # Open the test image to know size
    lr = Image.open(path.join(opt.testdir,"LR",path.basename(filename.replace('_SR'+opt.srext,opt.lrext))))
    
    # Count correct recognized image
    if lpt_recog == lpt_num:
        total_corr_recog_img += 1
        # Count correct recognized image by size
        corr_recog_size_count[lr.size] = corr_recog_size_count.get(lr.size, 0) + 1        
    else:
        corr_recog_size_count[lr.size] = corr_recog_size_count.get(lr.size, 0)    

    # Count correct recognized lpt
    lpt_unique[lpt_num] = lpt_unique.get(lpt_num, 0) + 1
    if lpt_recog == lpt_num and lpt_unique[lpt_num] == 1:
        total_corr_recog_lpt += 1

    # Count images by the number of correct recognized char in a lpt    
    tmp = len([i for i,j in zip(lpt_recog_seq,lpt_num_seq) if i == j])    
    tmp = (tmp,len(lpt_recog_seq))    
    corr_num_char_count[tmp] = corr_num_char_count.get(tmp, 0) + 1
        
    # Count correct recognized char in each position    
    if len(lpt_num) == 8: # 50Y_1234
        corr_recog_char_pos_count[0][lpt_num[0]] += 1 if lpt_recog[0] == lpt_num[0] else 0
        corr_recog_char_pos_count[1][lpt_num[1]] += 1 if lpt_recog[1] == lpt_num[1] else 0
        corr_recog_char_pos_count[2][lpt_num[2]] += 1 if lpt_recog[2] == lpt_num[2] else 0
        # 3
        # 4
        corr_recog_char_pos_count[5][lpt_num[4]] += 1 if lpt_recog[4] == lpt_num[4] else 0
        corr_recog_char_pos_count[6][lpt_num[5]] += 1 if lpt_recog[5] == lpt_num[5] else 0
        corr_recog_char_pos_count[7][lpt_num[6]] += 1 if lpt_recog[6] == lpt_num[6] else 0
        corr_recog_char_pos_count[8][lpt_num[7]] += 1 if lpt_recog[7] == lpt_num[7] else 0

        char_pos_count[0][lpt_num[0]] += 1
        char_pos_count[1][lpt_num[1]] += 1
        char_pos_count[2][lpt_num[2]] += 1
        # 3
        # 4
        char_pos_count[5][lpt_num[4]] += 1
        char_pos_count[6][lpt_num[5]] += 1
        char_pos_count[7][lpt_num[6]] += 1
        char_pos_count[8][lpt_num[7]] += 1
    elif len(lpt_num) == 10: # 50Y_123.45
        corr_recog_char_pos_count[0][lpt_num[0]] += 1 if lpt_recog[0] == lpt_num[0] else 0
        corr_recog_char_pos_count[1][lpt_num[1]] += 1 if lpt_recog[1] == lpt_num[1] else 0
        corr_recog_char_pos_count[2][lpt_num[2]] += 1 if lpt_recog[2] == lpt_num[2] else 0
        # 3
        # 4
        corr_recog_char_pos_count[5][lpt_num[4]] += 1 if lpt_recog[4] == lpt_num[4] else 0
        corr_recog_char_pos_count[6][lpt_num[5]] += 1 if lpt_recog[5] == lpt_num[5] else 0
        corr_recog_char_pos_count[7][lpt_num[6]] += 1 if lpt_recog[6] == lpt_num[6] else 0
        corr_recog_char_pos_count[8][lpt_num[8]] += 1 if lpt_recog[8] == lpt_num[8] else 0
        corr_recog_char_pos_count[9][lpt_num[9]] += 1 if lpt_recog[9] == lpt_num[9] else 0
        
        char_pos_count[0][lpt_num[0]] += 1
        char_pos_count[1][lpt_num[1]] += 1
        char_pos_count[2][lpt_num[2]] += 1
        # 3
        # 4
        char_pos_count[5][lpt_num[4]] += 1
        char_pos_count[6][lpt_num[5]] += 1
        char_pos_count[7][lpt_num[6]] += 1
        char_pos_count[8][lpt_num[8]] += 1
        char_pos_count[9][lpt_num[9]] += 1        
    elif len(lpt_num) == 9: # 51Z1_1234
        corr_recog_char_pos_count[0][lpt_num[0]] += 1 if lpt_recog[0] == lpt_num[0] else 0
        corr_recog_char_pos_count[1][lpt_num[1]] += 1 if lpt_recog[1] == lpt_num[1] else 0
        corr_recog_char_pos_count[2][lpt_num[2]] += 1 if lpt_recog[2] == lpt_num[2] else 0
        corr_recog_char_pos_count[3][lpt_num[3]] += 1 if lpt_recog[3] == lpt_num[3] else 0
        # 4
        corr_recog_char_pos_count[5][lpt_num[5]] += 1 if lpt_recog[5] == lpt_num[5] else 0
        corr_recog_char_pos_count[6][lpt_num[6]] += 1 if lpt_recog[6] == lpt_num[6] else 0
        corr_recog_char_pos_count[7][lpt_num[7]] += 1 if lpt_recog[7] == lpt_num[7] else 0
        corr_recog_char_pos_count[8][lpt_num[8]] += 1 if lpt_recog[8] == lpt_num[8] else 0

        char_pos_count[0][lpt_num[0]] += 1
        char_pos_count[1][lpt_num[1]] += 1
        char_pos_count[2][lpt_num[2]] += 1
        char_pos_count[3][lpt_num[3]] += 1
        # 4
        char_pos_count[5][lpt_num[5]] += 1
        char_pos_count[6][lpt_num[6]] += 1
        char_pos_count[7][lpt_num[7]] += 1
        char_pos_count[8][lpt_num[8]] += 1
    elif len(lpt_num) == 11: # 51Z1_123.45
        corr_recog_char_pos_count[0][lpt_num[0]] += 1 if lpt_recog[0] == lpt_num[0] else 0
        corr_recog_char_pos_count[1][lpt_num[1]] += 1 if lpt_recog[1] == lpt_num[1] else 0
        corr_recog_char_pos_count[2][lpt_num[2]] += 1 if lpt_recog[2] == lpt_num[2] else 0
        corr_recog_char_pos_count[3][lpt_num[3]] += 1 if lpt_recog[3] == lpt_num[3] else 0
        # 4
        corr_recog_char_pos_count[5][lpt_num[5]] += 1  if lpt_recog[5] == lpt_num[5] else 0
        corr_recog_char_pos_count[6][lpt_num[6]] += 1  if lpt_recog[6] == lpt_num[6] else 0
        corr_recog_char_pos_count[7][lpt_num[7]] += 1  if lpt_recog[7] == lpt_num[7] else 0
        corr_recog_char_pos_count[8][lpt_num[9]] += 1  if lpt_recog[9] == lpt_num[9] else 0
        corr_recog_char_pos_count[9][lpt_num[10]] += 1 if lpt_recog[10] == lpt_num[10] else 0

        char_pos_count[0][lpt_num[0]] += 1 
        char_pos_count[1][lpt_num[1]] += 1 
        char_pos_count[2][lpt_num[2]] += 1 
        char_pos_count[3][lpt_num[3]] += 1 
        # 4
        char_pos_count[5][lpt_num[5]] += 1 
        char_pos_count[6][lpt_num[6]] += 1 
        char_pos_count[7][lpt_num[7]] += 1 
        char_pos_count[8][lpt_num[9]] += 1 
        char_pos_count[9][lpt_num[10]] += 1
    elif len(lpt_num) == 12: # 59MÐ1_123.45
        corr_recog_char_pos_count[0][lpt_num[0]] += 1 if lpt_recog[0] == lpt_num[0] else 0
        corr_recog_char_pos_count[1][lpt_num[1]] += 1 if lpt_recog[1] == lpt_num[1] else 0
        corr_recog_char_pos_count[2][lpt_num[2]] += 1 if lpt_recog[2] == lpt_num[2] else 0
        corr_recog_char_pos_count[3][lpt_num[3]] += 1 if lpt_recog[3] == lpt_num[3] else 0
        corr_recog_char_pos_count[4][lpt_num[4]] += 1 if lpt_recog[4] == lpt_num[4] else 0
        corr_recog_char_pos_count[5][lpt_num[6]] += 1 if lpt_recog[6] == lpt_num[6] else 0
        corr_recog_char_pos_count[6][lpt_num[7]] += 1 if lpt_recog[7] == lpt_num[7] else 0
        corr_recog_char_pos_count[7][lpt_num[8]] += 1 if lpt_recog[8] == lpt_num[8] else 0
        corr_recog_char_pos_count[8][lpt_num[10]] += 1 if lpt_recog[10] == lpt_num[10] else 0
        corr_recog_char_pos_count[9][lpt_num[11]] += 1 if lpt_recog[11] == lpt_num[11] else 0
        
        char_pos_count[0][lpt_num[0]] += 1 
        char_pos_count[1][lpt_num[1]] += 1 
        char_pos_count[2][lpt_num[2]] += 1 
        char_pos_count[3][lpt_num[3]] += 1 
        char_pos_count[4][lpt_num[4]] += 1 
        char_pos_count[5][lpt_num[6]] += 1 
        char_pos_count[6][lpt_num[7]] += 1 
        char_pos_count[7][lpt_num[8]] += 1 
        char_pos_count[8][lpt_num[10]] += 1
        char_pos_count[9][lpt_num[11]] += 1
    else:
        raise ValueError("lpt_num has wrong value: {}".format(lpt_num))
    
    # Calculate PSNR, SSIM for entire image
    psnr = compare_psnr(img,compareimg)
    avgpsnr += psnr
    ssim = compare_ssim(compareimg,img,dynamic_range=max(compareimg.max(),img.max())-min(compareimg.min(),img.min()))
    ssim = round(ssim,2)
    avgssim += ssim
    testresult.append("{0}: {1}: {2:.2f} / {3:.2f} : {4}\r\n".format(path.basename(comparename),path.basename(filename),psnr,ssim, lpt_recog))
    
    ssim_size_count[lr.size] = ssim_size_count.get(lr.size, 0) + 1;
    ssim_size_sum[lr.size] = ssim_size_sum.get(lr.size, 0) + ssim;

    ssim_count[ssim] = ssim_count.get(ssim, 0) + 1;
raise SystemExit # debug
testresult.append("avgpsnr / avgssim: {0:.2f} / {1:.2f}\r\n".format(avgpsnr/len(opt.filenames),avgssim/len(opt.filenames)))
testresult.append("Correct recognized images: {0}/{1} ({2:.2f}%)\r\n".format(total_corr_recog_img, len(opt.filenames), total_corr_recog_img*100/len(opt.filenames)))
testresult.append("Correct recognized lpt: {0}/{1} ({2:.2f}%)\n".format(total_corr_recog_lpt, len(lpt_unique.keys()), total_corr_recog_lpt*100/len(lpt_unique.keys())))

# Save test result to file
#
with open(path.join(opt.save_dir,"testlog.txt"), "w") as f:
    f.writelines(testresult)

# Calculate percent of correct recognized images for each size
for k in corr_recog_size_count.keys():
    corr_recog_size_count[k] = round(corr_recog_size_count[k]*100/ssim_size_count[k],1)

# Calculate percent of correct recognized for each character
for k in corr_recog_char_count.keys():
    corr_recog_char_count[k] = corr_recog_char_count[k]*100//char_count[k]

# Calculate percent of corr_num_char_count
for k in corr_num_char_count.keys():
    corr_num_char_count[k] = round(corr_num_char_count[k]*100/len(opt.filenames),1)

# Calculate percent of correct recognized for each character in each position
for pos in corr_recog_char_pos_count.keys():
    for c in corr_recog_char_pos_count[pos]:
        if char_pos_count[pos][c] == 0:
            corr_recog_char_pos_count[pos][c] = 0
        else:
            corr_recog_char_pos_count[pos][c] = corr_recog_char_pos_count[pos][c]*100//char_pos_count[pos][c]

# Calculate the average SSIM for each character
avg_ssim_char = {}
for k in char_count.keys():
    if char_count.get(k,0) != 0:
        avg_ssim_char[k] = round(ssim_char_sum[k]/char_count[k],2)

# Calculate percent of ssim > ssimT for each character
for k in ssimT_char_count.keys():
    ssimT_char_count[k] = ssimT_char_count[k]*100//char_count[k]

# Calculate the average SSIM for each image size
for k in ssim_size_count.keys():
    ssim_size_count[k] = ssim_size_sum[k]/ssim_size_count[k]

# Display ssim_size_count
#
_, axis, data = sort3list([w*h for w,h in ssim_size_count.keys()], ["{0} x {1}".format(w,h) for w,h in ssim_size_count.keys()], list(ssim_size_count.values()))
figwidth = 22
fig = plt.figure(figsize=(figwidth,figwidth*9/16))
plt.title("Chỉ số SSIM trung bình theo kích thước", y=1.01, fontsize=24)
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
plt.title("Số lượng ảnh theo chỉ số SSIM", y=1.01, fontsize=24)
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

# Display avg_ssim_char
#
axis, data = sort2list(list(avg_ssim_char.keys()), list(avg_ssim_char.values()))
figwidth = 20
fig = plt.figure(figsize=(figwidth,figwidth*9/16))
plt.title("Chỉ số SSIM trung bình theo mỗi ký tự", y=1.01, fontsize=24)
plt.xlabel("Ký tự", fontsize=20)
plt.ylabel("Chỉ số SSIM trung bình", fontsize=20)
plt.grid(True,axis='y')
ax0 = fig.axes[0]
barcont = plt.bar(np.arange(len(axis)),data,tick_label=axis)
autolabel(barcont, ax0, data, fontsize=14)
for l in ax0.xaxis.get_ticklabels():
    l.set_fontsize(16)
for l in ax0.yaxis.get_ticklabels():
    l.set_fontsize(16)
ax0.autoscale(True,'x',True)
plt.savefig(path.join(opt.save_dir,"avg_ssim_char.jpg"))
plt.close(fig)

save_stc(path.join(opt.save_dir,"avg_ssim_char.txt"), axis, data)

# Display ssimT_char_count
#
axis, data = sort2list(list(ssimT_char_count.keys()), list(ssimT_char_count.values()))
figwidth = 20
fig = plt.figure(figsize=(figwidth,figwidth*9/16))
plt.title("Tỉ lệ ảnh có SSIM lớn hơn {} theo mỗi ký tự".format(opt.ssimT), y=1.01, fontsize=24)
plt.xlabel("Ký tự", fontsize=20)
plt.ylabel("%", fontsize=20)
plt.grid(True,axis='y')
ax0 = fig.axes[0]
barcont = plt.bar(np.arange(len(axis)),data,tick_label=axis)
autolabel(barcont, ax0, data, fontsize=14)
for l in ax0.xaxis.get_ticklabels():
    l.set_fontsize(16)
for l in ax0.yaxis.get_ticklabels():
    l.set_fontsize(16)
ax0.autoscale(True,'x',True)
plt.savefig(path.join(opt.save_dir,"ssimT_char_count.jpg"))
plt.close(fig)

save_stc(path.join(opt.save_dir,"ssimT_char_count.txt"), axis, data)

# Display corr_recog_char_count
#
axis, data = sort2list(list(corr_recog_char_count.keys()), list(corr_recog_char_count.values()))
figwidth = 20
fig = plt.figure(figsize=(figwidth,figwidth*9/16))
plt.title("Tỉ lệ nhận dạng đúng theo mỗi ký tự", y=1.01, fontsize=24)
plt.xlabel("Ký tự", fontsize=20)
plt.ylabel("%", fontsize=20)
plt.grid(True,axis='y')
ax0 = fig.axes[0]
barcont = plt.bar(np.arange(len(axis)),data,tick_label=axis)
autolabel(barcont, ax0, data, fontsize=16)
for l in ax0.xaxis.get_ticklabels():
    l.set_fontsize(16)
for l in ax0.yaxis.get_ticklabels():
    l.set_fontsize(16)
ax0.autoscale(True,'x',True)
plt.savefig(path.join(opt.save_dir,"corr_recog_char_count.jpg"))
plt.close(fig)

save_stc(path.join(opt.save_dir,"corr_recog_char_count.txt"), axis, data)

# Display corr_recog_char_pos_count
#
row = len(corr_recog_char_pos_count)

figwidth = 22
f, _axes = plt.subplots(row, sharey=True, figsize=(figwidth*9/16, figwidth))
f.suptitle("Tỉ lệ nhận dạng đúng của mỗi ký tự theo vị trí của ký tự", y=0.94, fontsize=24)
for i,pos in enumerate(corr_recog_char_pos_count.keys()):    
    axis, data = sort2list(list(corr_recog_char_pos_count[pos].keys()), list(corr_recog_char_pos_count[pos].values()))
    axis, data = np.array(axis), np.array(data)
    _axes[i].plot(axis, data)
    _axes[i].grid(True)
    _axes[i].set_title("Vị trí số {}".format(pos), fontsize=18)
    save_stc(path.join(opt.save_dir,"Corr_Position {}.txt".format(pos)), axis.tolist(), data.tolist())
f.subplots_adjust(hspace=0.5)
plt.setp([a.get_xticklabels() for a in f.axes], fontsize=14)
plt.setp([a.get_yticklabels() for a in f.axes], fontsize=14)
#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

plt.savefig(path.join(opt.save_dir,"corr_recog_char_pos_count.jpg"))

# Display corr_num_char_count
#
_, axis, data = sort3list([c*t for c,t in corr_num_char_count.keys()], ["{0}/{1}".format(c,t) for c,t in corr_num_char_count.keys()], list(corr_num_char_count.values()))
figwidth = 20
fig = plt.figure(figsize=(figwidth,figwidth*9/16))
plt.title("Tỉ lệ số lượng ký tự nhận dạng đúng của mỗi biển số xe", y=1.01, fontsize=24)
plt.xlabel("Số lượng ký tự nhận dạng đúng", fontsize=20)
plt.ylabel("%", fontsize=20)
plt.grid(True,axis='y')
ax0 = fig.axes[0]
barcont = plt.bar(np.arange(len(axis)),data,tick_label=axis)
autolabel(barcont, ax0, data, fontsize=16)
for l in ax0.xaxis.get_ticklabels():
    l.set_fontsize(16)
for l in ax0.yaxis.get_ticklabels():
    l.set_fontsize(16)
ax0.autoscale(True,'x',True)
plt.savefig(path.join(opt.save_dir,"corr_num_char_count.jpg"))
plt.close(fig)

save_stc(path.join(opt.save_dir,"corr_num_char_count.txt"), axis, data)

# Display corr_recog_size_count
#
_, axis, data = sort3list([w*h for w,h in corr_recog_size_count.keys()], ["{0} x {1}".format(w,h) for w,h in corr_recog_size_count.keys()], list(corr_recog_size_count.values()))
figwidth = 22
fig = plt.figure(figsize=(figwidth,figwidth*9/16))
plt.title("Tỉ lệ số ảnh nhận dạng đúng trên số ảnh theo kích thước", y=1.01, fontsize=24)
plt.xlabel("Kích thước của ảnh (pixel)", fontsize=20)
plt.ylabel("%", fontsize=20)
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
plt.savefig(path.join(opt.save_dir,"corr_recog_size_count.jpg"))
plt.close(fig)

save_stc(path.join(opt.save_dir,"corr_recog_size_count.txt"), axis, data)
