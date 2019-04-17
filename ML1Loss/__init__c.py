import torch
# from torchvision import utils
import lpt_utils
from lpt_utils import lptnum_of

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

def product(list):
    p = 1
    for x in list:
        p *= x
    return p

class ML1Loss(torch.nn.Module):
    def __init__(self, target_height=48, target_width=69):
        super(ML1Loss, self).__init__()
        self.lpt_std_size = lpt_utils.lpt_std_size

        self.hb = 0 # begin
        self.he = target_height # end
        self.rect_b = 11

        square = target_height == target_width
        w = target_width
        h = target_height
        if square:
            self.rect_b = round(self.rect_b * h / self.lpt_std_size[0])
            self.hb = self.rect_b
            h = round(w * self.lpt_std_size[1] / self.lpt_std_size[0])
            self.he = self.hb + h

        # Change if target size is not standard
        if w != self.lpt_std_size[0] or h != self.lpt_std_size[1]:
            lpt_utils.char_size = (round(lpt_utils.char_size[0] * w / self.lpt_std_size[0]), round(lpt_utils.char_size[1] * h / self.lpt_std_size[1]))
            lpt_utils.Cx[0] = round(lpt_utils.Cx[0] * w / self.lpt_std_size[0])
            lpt_utils.Cx[1] = round(lpt_utils.Cx[1] * w / self.lpt_std_size[0])
            lpt_utils.Cx[2] = round(lpt_utils.Cx[2] * w / self.lpt_std_size[0])
            lpt_utils.Cx[3] = round(lpt_utils.Cx[3] * w / self.lpt_std_size[0])
            lpt_utils.Cy[0] = round(lpt_utils.Cy[0] * h / self.lpt_std_size[1])
            lpt_utils.Cy[1] = round(lpt_utils.Cy[1] * h / self.lpt_std_size[1])
            lpt_utils.Dx[0] = round(lpt_utils.Dx[0] * w / self.lpt_std_size[0])
            lpt_utils.Dx[1] = round(lpt_utils.Dx[1] * w / self.lpt_std_size[0])
            lpt_utils.Dx[2] = round(lpt_utils.Dx[2] * w / self.lpt_std_size[0])

        self.L1, self.L2 = lpt_utils.L1L2()
        
    def _get_img_chars(self, lpt_num, img):
        # Standard parameters of license plate in pixel
        char_size = lpt_utils.char_size

        # The upper-left corner of characters
        L1 = self.L1
        L2 = self.L2

        lptnum = lpt_num.split("_")
        lptnum[1] = lptnum[1].replace(".","")

        img_chars = [img[L1[i][1]:char_size[1]+L1[i][1], L1[i][0]:char_size[0]+L1[i][0]] for i in range(len(lptnum[0])) if lptnum[0][i].isalpha()]
        lptnum = [c for c in lptnum[0] if c.isalpha()]

        return img_chars, ''.join(lptnum)

    def forward(self, input, target, labels):
        _assert_no_grad(target)

        loss = torch.abs(input - target)

        lptnums = [lptnum_of(label) for label in labels]

        img_chars_l = []
        for i, lptnum in enumerate(lptnums):
            for j, _ in enumerate(loss[i,:,self.hb:self.he,:]):
                img_chars_l.append(self._get_img_chars(lptnum, loss[i,j,self.hb:self.he,:]))

        # test only
        # for i in range(len(img_chars_l)):
            # k=0
            # for im, c in zip(img_chars_l[i][0],img_chars_l[i][1]):
                # utils.save_image(im,"{}_{}_{}.png".format(i,c,k),normalize=True, range=(0,255))
                # k += 1
        # raise SystemExit

        # Split img_char by char
        splited_chars_l = {}
        for i in range(len(img_chars_l)):
            for im, c in zip(img_chars_l[i][0],img_chars_l[i][1]):
                splited_chars_l[c] = splited_chars_l.get(c,[])
                splited_chars_l[c].append(im)

        # Calculate mean loss
        mean_char_loss = []
        total_char_area = 0
        for k in splited_chars_l.keys():
            sum_k = 0
            sum_s = 0
            for i in range(len(splited_chars_l[k])):
                sum_k += splited_chars_l[k][i].sum()
                sum_s += product(splited_chars_l[k][i].size())
                splited_chars_l[k][i] = 0
            mean_char_loss.append(sum_k/sum_s)
            total_char_area += sum_s

        bg_loss = loss.sum() / (product(loss.size()) - total_char_area)
        char_loss = sum(mean_char_loss)/len(mean_char_loss)

        return (bg_loss + char_loss)/2
