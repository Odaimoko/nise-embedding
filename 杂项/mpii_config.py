import argparse
import os
import os.path
import sys

import numpy as np


def get_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch CPN Training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')
    parser.add_argument('--epochs', default=32, type=int, metavar='N',
                        help='number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    return parser


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


class Config:

    #
    # ─── TRAINING ───────────────────────────────────────────────────────────────────
    #

    START_EPOCH = 0
    END_EPOCH = 40
    batch_size = 32
    lr = 2.5e-4
    weight_decay = .05
    # lr_gamma = 0.05
    # lr_dec_epoch = list(range(6, 40, 6))

    # tag loss weighted against heatmap loss, since heatmap's entries are less than 1
    emb_tag_weight = 4

    
    #
    # ─── DATA ───────────────────────────────────────────────────────────────────────
    #

    num_class = 16
    image_path = 'data/mpii-video-pose'
    data_shape = (512, 512)
    output_shape = (128, 128)

    gt_hmap_sigmas = [3, 7, 11]

    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8),
                (9, 10), (11, 12), (13, 14), (15, 16)]
    bbox_extend_factor = (0.1, 0.15)  # x, y

    # data augmentation setting
    scale_factor = (0.7, 1.35)
    rot_factor = 45

    pixel_means = np.array([122.7717, 115.9465, 102.9801])  # RGB
    


cfg = Config()
# add_pypath(cfg.root_dir)
