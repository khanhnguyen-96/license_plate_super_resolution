import torch
# import torch.nn as nn
from torchvision import utils
from torch import optim
import cv2
import numpy as np
from ML1Loss import ML1Loss

npImg1 = cv2.imread("HRS_99H8_4801_s144.bmp")

img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)
img2 = torch.randn(img1.size(), requires_grad = True)

print(img1.size())
print(img2.size())
# device = torch.device('cuda')
# if torch.cuda.is_available():
    # img1 = img1.to(device)
    # img2 = img2.to(device)


criterion = ML1Loss(img1.size()[2], img1.size()[3])

optimizer = optim.Adam([img2], lr=0.01)

count = 0
loss = 1000
while loss > 100:
    optimizer.zero_grad()
    loss = criterion(img2, img1, ["HRS_99H8_4801_s144.bmp"])
    count += 1
    # print("{0}: {1}".format(count,loss))
    loss.backward()
    optimizer.step()    

utils.save_image(img2,"img2.png",normalize=True, range=(0,255))
# utils.save_image(img1,"img1.png",normalize=True, range=(0,255))
