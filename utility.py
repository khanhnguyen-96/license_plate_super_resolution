import os
import math
import time
import datetime
# from functools import reduce

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
# import skimage.io as sio
# import skimage.color as sc

from model import model
from loss import loss

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision.utils as tu
from torch.autograd import Variable

matplotlib.use("Agg")


class timer:
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint:
    def __init__(self, args):
        print("args in init: " + str(args.load))
        self.args = args
        self.ok = True
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

        if args.load == ".":
            if args.save == ".":
                args.save = now
            self.dir = "/content/experiment/" + args.save
        else:
            self.dir = "/content/experiment/" + args.load
            print("Is exist load path: " + str(os.path.exists(self.dir)))
            if not os.path.exists(self.dir):
                args.load = "."

        if args.reset:
            os.system("rm -rf " + self.dir)
            args.load = "."

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + "/model")
        _make_dir(self.dir + "/results")

        open_type = "a" if os.path.exists(self.dir + "/log.txt") else "w"
        self.log_file = open(self.dir + "/log.txt", open_type)
        with open(self.dir + "/config.txt", open_type) as f:
            f.write(now + "\n\n")
            for arg in vars(args):
                f.write("{}: {}\n".format(arg, getattr(args, arg)))
            f.write("\n")

    def load(self):
        my_model = model().get_model(self.args)
        trainable = filter(lambda x: x.requires_grad, my_model.parameters())

        if self.args.optimizer == "SGD":
            optimizer_function = optim.SGD
            kwargs = {"momentum": self.args.momentum}
        elif self.args.optimizer == "ADAM":
            optimizer_function = optim.Adam
            kwargs = {
                "betas": (self.args.beta1, self.args.beta2),
                "eps": self.args.epsilon,
            }
        elif self.args.optimizer == "RMSprop":
            optimizer_function = optim.RMSprop
            kwargs = {"eps": self.args.epsilon}

        kwargs["lr"] = self.args.lr
        kwargs["weight_decay"] = 0
        my_optimizer = optimizer_function(trainable, **kwargs)

        self.log_training = torch.Tensor()
        self.log_test = torch.Tensor()
        resume = self.args.resume
        if self.args.load == ".":
            my_loss = loss(self.args).get_loss()
        else:
            if not self.args.test_only:
                self.log_training = torch.load(self.dir + "/log_training.pt")
                self.log_test = torch.load(self.dir + "/log_test.pt")

            if resume == -1:
                my_model.load_state_dict(
                    torch.load(self.dir + "/model/model_latest.pt")
                )
                resume = len(self.log_test)
            else:
                my_model.load_state_dict(
                    torch.load(self.dir + "/model/model_{}.pt".format(resume))
                )

            my_loss = torch.load(self.dir + "/loss.pt")
            my_optimizer.load_state_dict(torch.load(self.dir + "/optimizer.pt"))

            print("Load loss function from checkpoint...")
            print("Continue from epoch {}...".format(resume))

        if resume > 1:
            resume -= 1
        if self.args.decay_type == "step":
            my_scheduler = lrs.StepLR(
                my_optimizer,
                step_size=self.args.lr_decay,
                gamma=self.args.gamma,
                last_epoch=resume,
            )
        elif self.args.decay_type.find("step") >= 0:
            milestones = self.args.decay_type.split("_")
            milestones.pop(0)
            milestones = list(map(lambda x: int(x), milestones))
            my_scheduler = lrs.MultiStepLR(
                my_optimizer, milestones=milestones, gamma=self.args.gamma
            )

        return my_model, my_loss, my_optimizer, my_scheduler

    def add_log(self, log, train=True):
        if train:
            self.log_training = torch.cat([self.log_training, log])
        else:
            self.log_test = torch.cat([self.log_test, log])

    def save(self, trainer, epoch, is_best=False):
        if self.args.n_GPUs > 1:
            state = trainer.model.module.state_dict()
        else:
            state = trainer.model.state_dict()

        save_list = [(state, "model/model_latest.pt")]
        if not self.args.test_only:
            if is_best:
                if trainer.tmp_loss < self.args.model_best_loss:
                    save_list.append((state, "model/model_best_e{}.pt".format(epoch)))
                else:
                    save_list.append((state, "model/model_best.pt"))
            if self.args.save_models:
                save_list.append((state, "model/model_{}.pt".format(epoch)))

            save_list.append((trainer.loss, "loss.pt"))
            save_list.append((trainer.optimizer.state_dict(), "optimizer.pt"))
            save_list.append((self.log_training, "log_training.pt"))
            save_list.append((self.log_test, "log_test.pt"))
            # self.plot(trainer, epoch, self.log_training, self.log_test)

        for o, p in save_list:
            torch.save(o, os.path.join(self.dir, p))

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + "\n")
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + "/log.txt", "a")

    def done(self):
        self.log_file.close()

    def plot(self, trainer, epoch, training, test):
        axis = np.linspace(1, epoch, epoch)

        def _init_figure(label):
            fig = plt.figure()
            plt.title(label)
            plt.xlabel("Epochs")
            plt.grid(True)

            return fig

        def _close_figure(fig, filename):
            plt.savefig(filename)
            plt.close(fig)

        for i, trainerLoss in enumerate(trainer.loss):
            label = "{} Loss".format(trainerLoss["type"])
            fig = _init_figure(label)
            plt.plot(axis, training[:, i].numpy(), label=label)
            plt.legend()
            _close_figure(fig, "{}/loss_{}.pdf".format(self.dir, trainerLoss["type"]))

        set_name = type(trainer.loader_test.dataset).__name__
        fig = _init_figure("SR on {}".format(set_name))
        for idx_scale, scale in enumerate(self.args.scale):
            legend = "Scale {}".format(scale)
            plt.plot(axis, test[:, idx_scale].numpy(), label=legend)
            plt.legend()

        _close_figure(fig, "{}/test_{}.pdf".format(self.dir, set_name))

    def save_results(self, filename, save_list, scale):
        # filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        filename = "{}/results/{}_".format(self.dir, filename)
        postfix = ("SR", "LR", "HR")
        for v, p in zip(save_list, postfix):
            tu.save_image(v.data[0], "{}{}.png".format(filename, p), padding=0)


def chop_forward(x, model, scale, shave=10, min_size=80000, n_GPUs=1):
    n_GPUs = min(n_GPUs, 4)
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size) : w],
        x[:, :, (h - h_size) : h, 0:w_size],
        x[:, :, (h - h_size) : h, (w - w_size) : w],
    ]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i : (i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            chop_forward(patch, model, scale, shave, min_size, n_GPUs)
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = Variable(x.data.new(b, c, h, w), volatile=True)
    output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] = sr_list[1][
        :, :, 0:h_half, (w_size - w + w_half) : w_size
    ]
    output[:, :, h_half:h, 0:w_half] = sr_list[2][
        :, :, (h_size - h + h_half) : h_size, 0:w_half
    ]
    output[:, :, h_half:h, w_half:w] = sr_list[3][
        :, :, (h_size - h + h_half) : h_size, (w_size - w + w_half) : w_size
    ]

    return output


def x8_forward(img, model, precision="single"):
    def _transform(v, op):
        if precision != "single":
            v = v.float()

        v2np = v.data.cpu().numpy()
        if op == "vflip":
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == "hflip":
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == "transpose":
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).cuda()

        if precision == "half":
            ret = ret.half()
        elif precision == "double":
            ret = ret.double()

        return Variable(ret, volatile=True)

    lr_list = [img]
    for tf in "vflip", "hflip", "transpose":
        lr_list.extend([_transform(t, tf) for t in lr_list])

    sr_list = [model(aug) for aug in lr_list]
    for i in range(len(sr_list)):
        if i > 3:
            sr_list[i] = _transform(sr_list[i], "transpose")
        if i % 4 > 1:
            sr_list[i] = _transform(sr_list[i], "hflip")
        if (i % 4) % 2 == 1:
            sr_list[i] = _transform(sr_list[i], "vflip")

    output_cat = torch.cat(sr_list, dim=0)
    output = output_cat.mean(dim=0, keepdim=True)
    # output = output_cat.median(dim=0, keepdim=True)[0]

    return output


def quantize(img, rgb_range):
    return img.mul(255 / rgb_range).clamp(0, 255).round().div(255)


def calc_PSNR(sr, hr, set_name, scale):
    """
        Here we assume normalized(0~1) and quantized arguments.
        For Set5, Set14, B100, Urban100 dataset,
        we measure PSNR on luminance channel only
    """
    diff = (sr - hr).data

    # We will evaluate these datasets in y channel only
    luminance_only = ["Set5", "Set14", "B100", "Urban100"]
    if set_name in luminance_only:
        _, c, h, w = diff.size()
        shave = scale
        if c > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
        valid = diff[:, :, shave : (h - shave), shave : (w - shave)]
    else:
        # shave = scale + 6
        valid = diff

    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)
