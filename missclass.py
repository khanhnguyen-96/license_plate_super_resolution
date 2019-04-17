import numpy as np
from os import path
import argparse
from plot_utils import autolabel, sort2list, sort3list
from lpt_utils import lptnum_of

import matplotlib.pyplot as plt
plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Miss classify counting')
parser.add_argument('--textfile', type=str,  required=True, help='testlog.txt')
parser.add_argument('--save-dir', type=str, default=".", help='Directory to save result. If file exists, it will be overwritten.')
opt = parser.parse_args()

with open(opt.textfile) as f:
    lines = f.read().splitlines()

alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']

char_count = {}
hist = {}
for line in lines:
    linesplt = line.split(":")    
    try:  
        truth = lptnum_of(linesplt[0])
    except IndexError:
        truth = None
    if truth == None:
        print("The string \"{}\" is wrong lptnum format.".format(linesplt[0]))
        continue
    recover = linesplt[-1].strip(' ')
    for t,r in zip(truth,recover):
        if t == "_" or t == ".":
            continue        
        if t != r:
            hist[(t,r)] = hist.get((t,r), 0) + 1
            char_count[t] = char_count.get(t,0) + 1

for k in hist.keys():
    hist[k] /= char_count[k[0]]

# Normalize by row, i.e. ground truth char
# max_n = 1
# max_c = 1
# for k in hist.keys():
    # if k[0].isdigit():
        # if max_n < hist[k]:
            # max_n = hist[k]
    # else:
        # if max_c < hist[k]:
            # max_c = hist[k]
# for k in hist.keys():
    # if k[0].isdigit():
        # hist[k] /= max_n
    # else:
        # hist[k] /= max_c

# Plot
figwidth = 7.5
fig = plt.figure(figsize=(figwidth,figwidth))
plt.title("Tỉ lệ khôi phục sai theo ký tự", y=1.02, fontsize=20)
plt.xlabel("Ký tự khôi phục", fontsize=14)
plt.ylabel("Ký tự thực tế", fontsize=14)


x = alphabet

for i in alphabet:
    y = [i for j in range(len(x))]
    d = [hist.get((i,j), 0) for j in alphabet]    
    maxd = max(d)
    cl = [ (0.70,0.20,0.20) if a == maxd else (0.17, 0.36, 0.49) for a in d]    
    area = [(20 * di)**2 for di in d]  # 0 to 15 point radii
    plt.scatter(x, y, s=area, c=cl, alpha=1.) # c=[(0.47, 0.66, 0.79)],

plt.grid(True)
ax0 = fig.axes[0]
for l in ax0.xaxis.get_ticklabels():    
    l.set_fontsize(14)
for l in ax0.yaxis.get_ticklabels():
    l.set_fontsize(14)
# ax0.autoscale(True,'x',True)
# ax0.autoscale(True,'y',True)

plt.tight_layout()
plt.savefig(path.join(opt.save_dir,"miss_class_count.jpg"))
plt.close(fig)
