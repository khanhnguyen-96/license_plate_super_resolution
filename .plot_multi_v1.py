import numpy as np
import argparse
from plot_utils import autolabel, sort2list, sort3list, save_stc

import matplotlib.pyplot as plt
plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Read data from multi text files and plot in 1 chart.')
parser.add_argument('--type', type=str,  default="line", help='type of plot ("line" or "bar")')
parser.add_argument('--num_axis', type=str,  default="", help='x axis is number ("float" or "int")')
parser.add_argument('--num_data', type=str,  default="float", help='y axis is number ("float" or "int")')
parser.add_argument('--suptitle', type=str,  default="", help='Suptitle for plot')
parser.add_argument('--title', type=str, required=True, help='title')
parser.add_argument('--xlabel', type=str, required=True, help='x label')
parser.add_argument('--ylabel', type=str, required=True, help='y label')
parser.add_argument('--vertical_x', action='store_true', help='Vertical x label')
parser.add_argument('--savename', type=str, required=True, help='Filename of output file')
parser.add_argument('--legend', nargs='+', help='Legend for each plot data.')
parser.add_argument('--txt', nargs='+', help='Input text files.')
opt = parser.parse_args()

def get_data_from_txt(filename, num_axis=False, num_data=False):
    with open(filename, 'r') as f:
        text = f.read().split("\n\n")[1].strip("\n")
    text = text.split("\n")
    
    if num_axis == "float":
        axis = [float(t.split(":")[0]) for t in text]
    elif num_axis == "int":
        axis = [int(t.split(":")[0]) for t in text]
    else:
        axis = [t.split(":")[0] for t in text]
        
    if num_data == "float":
        data = [float(t.split(":")[1]) for t in text]
    elif num_data == "int":
        data = [int(t.split(":")[1]) for t in text]
    else:
        data = [t.split(":")[1] for t in text]
    return axis, data

def plot_lines(list_axis_data):
    figwidth = 20
    fig = plt.figure(figsize=(figwidth,figwidth*9/16))
    plt.title(opt.title, fontsize=24)
    plt.xlabel(opt.xlabel, fontsize=20)
    plt.ylabel(opt.ylabel, fontsize=20)
    plt.grid(True)
    ax0 = fig.axes[0]
    for i, (axis, data) in enumerate(list_axis_data):
        axis, data = sort2list(axis, data)
        plt.scatter(axis, data)
        plt.plot(axis, data, label=opt.legend[i])
    for l in ax0.xaxis.get_ticklabels():
        if opt.vertical_x:
            l.set_rotation('vertical')
        l.set_fontsize(16)
    for l in ax0.yaxis.get_ticklabels():
        l.set_fontsize(16)
    ax0.autoscale(True,'x',True)
    ax0.autoscale(True,'y',True)
    plt.legend(prop={'size':16})
    plt.savefig(opt.savename)
    plt.close(fig)

def plot_bars(list_axis_data):
    figwidth = 20
    fig = plt.figure(figsize=(figwidth,figwidth*9/16))
    plt.title(opt.title, fontsize=24)
    plt.xlabel(opt.xlabel, fontsize=20)
    plt.ylabel(opt.ylabel, fontsize=20)
    plt.grid(True,axis='y')
    ax0 = fig.axes[0]
    width = 0.45
    for i, (axis, data) in enumerate(list_axis_data):
        axis, data = sort2list(axis, data)        
        barcont = plt.bar(np.arange(len(axis))+width*i,data,width,tick_label=axis, label=opt.legend[i])
        autolabel(barcont, ax0, data, fontsize=14)        
    for l in ax0.xaxis.get_ticklabels():
        if opt.vertical_x:
            l.set_rotation('vertical')
        l.set_fontsize(16)
    for l in ax0.yaxis.get_ticklabels():
        l.set_fontsize(16)
    ax0.set_xticks(np.arange(len(axis)) + width / 2)
    ax0.autoscale(True,'x',True)
    plt.legend(prop={'size':16})
    plt.savefig(opt.savename)
    plt.close(fig)

# Read data and plot
list_axis_data = [get_data_from_txt(t, opt.num_axis, opt.num_data) for t in opt.txt]
if opt.type == 'line':
    plot_lines(list_axis_data)
elif opt.type == 'bar':
    plot_bars(list_axis_data)
else:
    raise ValueError('opt.type has wrong value')