import torch

import utility
import datetime
from option import args
from data import data
from trainer import Trainer
from torchvision import utils

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

starttime = datetime.datetime.now()
checkpoint.write_log(starttime.strftime('%d-%m-%Y-%H:%M:%S'))

if checkpoint.ok:
    my_loader = data().get_loader(args)
    
    # Save a batch for visual purpose
    # train_loader,_ = my_loader
    # print("len of train_loader: {}".format(len(train_loader)))
    # for iteration, batch in enumerate(train_loader, 1):
       # print("{0}, {1}".format(len(batch),len(batch[0])))        
       # for i,b in enumerate(batch[0]):           
           # print(b)
           # raise SystemExit
           # utils.save_image(b[0],"i{0}_{1}.png".format(iteration,i),normalize=True,range=(0,255))
       # break
    # raise SystemExit
    
    t = Trainer(my_loader, checkpoint, args)
    # My code 12.6.2018
    sump = sum(p.numel() for p in t.model.parameters())
    checkpoint.write_log("Total parameters: {}".format(sump))
    # End my code
    while not t.terminate():
        t.train()
        t.test(starttime=starttime)
    now = datetime.datetime.now()
    checkpoint.write_log(now.strftime('%d-%m-%Y-%H:%M:%S'))
    checkpoint.write_log("Time elapsed: {}".format(str(now - starttime)))
    checkpoint.done()

