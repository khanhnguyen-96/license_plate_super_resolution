import os

# from data import common
from data import srdata

import random
import numpy as np
# import scipy.misc as misc

# import torch
# import torch.utils.data as data

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

class LPT(srdata.SRData):
    def __init__(self, args, train=True):
        self.repeat = args.test_every // (args.n_train // args.batch_size)
        self.datatype = args.datatype
        super(LPT, self).__init__(args, train)

    def lptnum_of(self, filename, has_prefix=True):
        idx = [1,2] if has_prefix else [0,1]
        lptnum = ('.').join(filename.split(".")[:-1]).split("_")
        return lptnum[idx[0]] + "_" + lptnum[idx[1]]

    # datatype = 'realreal' or 'simureal' or 'realsimu' or 'simusimu'
    def get_list_hr_lr(self,datatype):
        list_hr = []
        list_lr = [[]]

        hr_ext = ".bmp"
        lr_ext = ".bmp"

        id_file = "id_train.txt"
        if datatype == 'realreal' or datatype == 'simureal':
            lrprefix = "LR_"
            hrprefix = "OR_"
            if datatype == 'simureal':
                hrprefix = "HRS_"            
        elif datatype == 'realsimu' or datatype == 'simusimu':
            lrprefix = "LRS_"
            hrprefix = "OR_"
            if datatype == 'simusimu':
                hrprefix = "HRS_"            
        else:
            raise ValueError('datatype is not in "realreal" or "simureal" or "realsimu" or "simusimu"')
        if self.train == False:
            id_file = "id_validate.txt"

        # Read license plate number from id_file
        lcplt_num = open(os.path.join(self.apath,id_file), "r").readlines()

        # Read all LR filenames
        LRfilenames = [f for f in os.listdir(self.apath) if f[0:2] == "LR" and is_image_file(os.path.join(self.apath, f))]

        for x in lcplt_num:
            x = x.strip("\n")
            hrname = os.path.join(self.apath,hrprefix + x + hr_ext)

            for n in [lrname for lrname in LRfilenames if self.lptnum_of(lrname) == x]:
                list_hr.append(hrname)
                list_lr[0].append(os.path.join(self.apath,n))

        return list_hr, list_lr

    def shuffle_a_pair_of_list(self,a,b):
        c = []
        for i in zip(a,b):
            c.append(i)

        random.shuffle(c)

        d,e= [],[]
        for i,j in c:
            d.append(i)
            e.append(j)
        return d,e

    def _scan(self):
        list_hr, list_lr = [],[[]]
        if self.datatype == 'all':
            h,l = self.get_list_hr_lr('realreal')
            list_hr.extend(h)
            list_lr[0].extend(l[0])
            h,l = self.get_list_hr_lr('realsimu')
            list_hr.extend(h)
            list_lr[0].extend(l[0])
            h,l = self.get_list_hr_lr('simureal')
            list_hr.extend(h)
            list_lr[0].extend(l[0])
            h,l = self.get_list_hr_lr('simusimu')
            list_hr.extend(h)
            list_lr[0].extend(l[0])
            list_hr, list_lr[0] = self.shuffle_a_pair_of_list(list_hr, list_lr[0])
        else:
            list_hr, list_lr = self.get_list_hr_lr(self.datatype)

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        # self.dir_hr = os.path.join(self.apath, '')
        # self.dir_lr = os.path.join(self.apath, '')
        # self.ext = '.bmp'

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx


