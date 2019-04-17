import os

# from data import common
from data import srdata

import random
import numpy as np
# import scipy.misc as misc

# import torch
# import torch.utils.data as data

class LPT(srdata.SRData):
    def __init__(self, args, train=True):
        self.repeat = args.test_every // (args.n_train // args.batch_size)
        self.datatype = args.datatype
        super(LPT, self).__init__(args, train)

    # datatype = 'realreal' or 'simureal' or 'realsimu' or 'simusimu'
    def get_list_hr_lr(self,datatype):
        list_hr = []
        list_lr = [[]]

        hr_ext = ".bmp"
        lr_ext = ".bmp"
        
        if datatype == 'realreal' or datatype == 'simureal':
            lrprefix = "LR_"
            hrprefix = "OR_"
            if datatype == 'simureal':
                hrprefix = "HRS_"
            id_file = "id_train_real.txt"
        elif datatype == 'realsimu' or datatype == 'simusimu':
            lrprefix = "LRS_"
            hrprefix = "OR_"
            if datatype == 'simusimu':
                hrprefix = "HRS_"
            id_file = "id_train_all.txt"
        else:
            raise ValueError('datatype is not in "realreal" or "simureal" or "realsimu" or "simusimu"')
        if self.train == False:
            id_file = "id_validate.txt"

        # Read license plate number from id_file
        lcplt_num = open(os.path.join(self.apath,id_file), "r").readlines()

        for x in lcplt_num:
            hrname = os.path.join(self.apath,hrprefix+x.strip("\n")+hr_ext)
            lrname = os.path.join(self.apath,lrprefix+x.strip("\n"))
            if (os.path.exists(lrname + lr_ext)):
                list_hr.append(hrname)
                list_lr[0].append(lrname + lr_ext)
            lrname += '-'
            for lr in [lrname + str(x) + lr_ext for x in range(20)]:
                if (os.path.exists(lr)):
                    list_hr.append(hrname)
                    list_lr[0].append(lr)

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


