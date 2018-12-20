import torch

from nise_lib.nise_config import nise_cfg, nise_logger
from plogs.logutils import Levels


def gen_rand_flow(batch_size, h, w):
    return torch.rand(*[batch_size, 2, h, w])


def gen_rand_img(batch_size, h, w):
    return torch.rand(*[batch_size, 3, h, w])


def gen_rand_img(batch_size, num_joints, h, w):
    return torch.rand(*[batch_size, num_joints, h, w])


def gen_rand_bboxes(num_people, h, w):
    x1 = torch.randint(low = 0, high = int(
        w / 2), size = [num_people, nise_cfg.DATA.num_joints])
    x2 = torch.randint(low = int(w / 2), high = w,
                       size = [num_people, nise_cfg.DATA.num_joints])
    
    y1 = torch.randint(low = 0, high = int(
        h / 2), size = [num_people, nise_cfg.DATA.num_joints])
    y2 = torch.randint(low = int(h / 2), high = h,
                       size = [num_people, nise_cfg.DATA.num_joints])
    bbox = torch.stack([x1, y1, x2, y2], 3)
    return bbox  # ng


def gen_rand_joints(num_people, h, w):
    x1 = torch.randint(low = 0, high = int(
        w / 2), size = [num_people, nise_cfg.DATA.num_joints])
    y1 = torch.randint(low = 0, high = int(
        h / 2), size = [num_people, nise_cfg.DATA.num_joints])
    joints = torch.stack([x1, y1], 2)
    return joints  # ng
