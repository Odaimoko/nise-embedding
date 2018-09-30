import argparse
import os
import os.path
import sys

import numpy as np


def get_arg_parser():
    parser = argparse.ArgumentParser(description = 'PyTorch CPN Training')
    parser.add_argument('-j', '--workers', default = 4, type = int, metavar = 'N',
                        help = 'number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default = 1, type = int, metavar = 'N',
                        help = 'number of GPU to use (default: 1)')
    parser.add_argument('--epochs', default = 32, type = int, metavar = 'N',
                        help = 'number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default = 0, type = int, metavar = 'N',
                        help = 'manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default = 'checkpoint', type = str, metavar = 'PATH',
                        help = 'path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default = '', type = str, metavar = 'PATH',
                        help = 'path to latest checkpoint')
    return parser


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


from easydict import EasyDict as ed

cfg = ed()

#
# ─── TRAINING ───────────────────────────────────────────────────────────────────
#
cfg.TRAIN = ed()
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.END_EPOCH = 40
cfg.TRAIN.batch_size = 32
cfg.TRAIN.lr = 2.5e-4
cfg.TRAIN.weight_decay = .05
# lr_gamma = 0.05
# lr_dec_epoch = list(range(6, 40, 6))

# tag loss weighted against heatmap loss, since heatmap's entries are less than 1
cfg.emb_tag_weight = 4

#
# ─── DATA ───────────────────────────────────────────────────────────────────────
#
cfg.DATA = ed()
cfg.DATA.num_joints = 16
cfg.DATA.image_path = 'data/mpii-video-pose'
cfg.DATA.data_shape = (512, 512)
cfg.DATA.output_shape = (128, 128)

cfg.DATA.gt_hmap_sigmas = [3, 7, 11]

cfg.DATA.symmetry = [(1, 2), (3, 4), (5, 6), (7, 8),
                     (9, 10), (11, 12), (13, 14), (15, 16)]
cfg.DATA.bbox_extend_factor = (0.075, 0.075)  # x, y

# data augmentation setting
cfg.DATA.scale_factor = (0.7, 1.35)
cfg.DATA.rot_factor = 45

cfg.DATA.pixel_means = np.array([122.7717, 115.9465, 102.9801])  # RGB

#  should ask author
cfg.DATA.flow_input_size = (640,384 )
cfg.DATA.human_bbox_size = (384, 384)

# debugging

cfg.DEBUG = ed()
cfg.DEBUG.DEVELOPING = True
cfg.DEBUG.FLOW = False
cfg.DEBUG.FRAME = True
cfg.DEBUG.HUMAN = True

# ─── ALGORITHM ──────────────────────────────────────────────────────────────────

cfg.ALG = ed()
cfg.ALG.DEQUE_CAPACITY = 3
