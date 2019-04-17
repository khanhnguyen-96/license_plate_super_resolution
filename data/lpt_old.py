import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class LPT(srdata.SRData):
    def __init__(self, args, train=True):
        self.repeat = args.test_every // (args.n_train // args.batch_size)
        super(LPT, self).__init__(args, train)

    def _scan(self):
        list_hr = []
        list_lr = [[]]

        hr_ext = ".bmp"
        lr_ext = ".bmp"

        if self.train == True:
            #id_file = "id_train_all.txt"
            id_file = "id_train_real.txt"
        else:
            #id_file = "id_validate.txt"
            id_file = "id_validate_real.txt"

        # Read license plate number from id_file
        lcplt_num = open(os.path.join(self.apath,id_file), "r").readlines()

        for x in lcplt_num:
            hrname = os.path.join(self.apath,"HRS_"+x.strip("\n")+hr_ext)
            lrname = os.path.join(self.apath,"LR_"+x.strip("\n"))
            if (os.path.exists(lrname + lr_ext)):
                list_hr.append(hrname)
                list_lr[0].append(lrname + lr_ext)
            lrname += '-'
            for lr in [lrname + str(x) + lr_ext for x in range(5)]:
                if (os.path.exists(lr)):
                    list_hr.append(hrname)
                    list_lr[0].append(lr)

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

