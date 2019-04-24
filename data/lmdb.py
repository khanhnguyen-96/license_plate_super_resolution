import lmdb
import pickle
import os
import logging
import torch.utils.data as data
import numpy as np
import torch

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]

####################
# Files & IO
####################


class LMDB(data.Dataset):
    def __init__(self, opt, train=True):
        super(LMDB, self).__init__()

        print("train boolean: " + train)

        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.LR_env = None
        self.HR_env = None

        self.LR_env, self.paths_LR = get_image_paths("lmdb", opt.dir_dataLR)
        self.HR_env, self.paths_HR = get_image_paths("lmdb", opt.dir_dataHR)

        assert self.paths_HR, "Error: HR path is empty."

        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(
                self.paths_HR
            ), "HR and LR datasets have different number of images - {}, {}.".format(
                len(self.paths_LR), len(self.paths_HR)
            )

        self.random_scale_list = [1]

    def __getitem__(self, index):
        HR_path, LR_path = None, None

        # get HR image
        HR_path = self.paths_HR[index]
        img_HR = read_img(self.HR_env, HR_path)

        # get LR image
        if self.paths_LR:
            LR_path = self.paths_LR[index]
            img_LR = read_img(self.LR_env, LR_path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))
        ).float()
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()

        if LR_path is None:
            LR_path = HR_path
        return {"LR": img_LR, "HR": img_HR, "LR_path": LR_path, "HR_path": HR_path}

    def __len__(self):
        return len(self.paths_HR)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_lmdb(dataroot):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    keys_cache_file = os.path.join(dataroot, "_keys_cache.p")
    logger = logging.getLogger("base")
    if os.path.isfile(keys_cache_file):
        logger.info("Read lmdb keys from cache: {}".format(keys_cache_file))
        keys = pickle.load(open(keys_cache_file, "rb"))
    else:
        with env.begin(write=False) as txn:
            logger.info("Creating lmdb keys cache: {}".format(keys_cache_file))
            keys = [key.decode("ascii") for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, "wb"))
    paths = sorted([key for key in keys if not key.endswith(".meta")])
    return env, paths


def get_image_paths(data_type, dataroot):
    env, paths = None, None
    if dataroot is not None:
        if data_type == "lmdb":
            env, paths = _get_paths_from_lmdb(dataroot)
        elif data_type == "img":
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError(
                "data_type [{:s}] is not recognized.".format(data_type)
            )
    return env, paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), "{:s} is not a valid directory".format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, "{:s} has no valid image file".format(path)
    return images


def _read_lmdb_img(env, path):
    with env.begin(write=False) as txn:
        buf = txn.get(path.encode("ascii"))
        buf_meta = txn.get((path + ".meta").encode("ascii")).decode("ascii")
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    H, W, C = [int(s) for s in buf_meta.split(",")]
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path):
    img = _read_lmdb_img(env, path)
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img
