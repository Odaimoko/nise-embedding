#
# ──────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: U S E   P R E T R A I N E D   M O D E L S   T O   T E S T
#       R E - I M P L E M E N T   S I M P L E   B A S E L I N E :
# ────────────────────────────────────────────────────────────────────────────
#
# import sys
#
# sys.path.append('../flownet2-pytorch/')
# sys.path.append('../simple-baseline-pytorch/')
# sys.path.append('../Detectron.pytorch/')
# !/usr/bin/env python

import torch
import torch.nn as nn
import scipy
import argparse
import os
import subprocess
import setproctitle
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *
from collections import deque
import cv2
import visdom
import json

# local packages
import nise_lib._init_paths
from nise_lib.frameitem import FrameItem
import flow_models
import flow_losses
import flow_datasets
from flownet_utils import flow_utils, tools
from nise_utils.imutils import *
from nise_utils.visualize import *
from nise_config import cfg as nise_cfg
from nise_functions import *
from nise_debugging_func import *
import tron_lib.utils.misc as misc_utils

from tron_lib.core.config import cfg as tron_cfg

# fp32 copy of parameters for update
global param_copy

viz = visdom.Visdom(env = 'run-with-flownet')

""" - detect person: do we use all the results? - """

""" - Normalize person crop to some size to fit flow input

# data[0] torch.Size([8, 3, 2, 384, 1024]) ，bs x channels x num_images, h, w
# target torch.Size([8, 2, 384, 1024]) maybe offsets

"""

""" - Now we have some images Q and current img, calculate FLow using previous and current one - """

""" - choose prev joints pos, add flow to determine pos in the next frame; generate new bbox - """

""" - merge bbox; nms; detect current image's bboxes' joints pos using human-pose-estimation - """

""" -  - """

""" - Associate ids. question: how to associate using more than two frames?
 between each 2?
 - """

batch_size = 8

# ─── FROM FLOWNET 2.0 ───────────────────────────────────────────────────────────
if nise_cfg.DEBUG.load_flow_model:
    parser = argparse.ArgumentParser()
    flow_init_parser_and_tools(parser, tools)
    flow_args, rest = parser.parse_known_args()
    flow_model = load_flow_model(flow_args, parser, tools)

# ─── FROM SIMPLE BASELINE ───────────────────────────────────────────────────────
if nise_cfg.DEBUG.load_joint_est_model:
    simple_human_det_model = load_simple_model()
    debug_print('Simple pose detector loaded.')

# ─── HUMAN DETECT ───────────────────────────────────────────────────────────────
if nise_cfg.DEBUG.load_human_det_model:
    human_detect_args = human_detect_parse_args()
    
    maskRCNN, dataset = load_human_detect_model(human_detect_args, tron_cfg)
    
    maskRCNN.eval()

# init deque (why deque??)
Q = deque(maxlen = nise_cfg.ALG.DEQUE_CAPACITY)

if nise_cfg.DEBUG.FRAME and nise_cfg.DEBUG.load_flow_model \
        and nise_cfg.DEBUG.load_human_det_model and nise_cfg.DEBUG.load_joint_est_model:
    
    if nise_cfg.DEBUG.DEVELOPING:
        seq_dir = 'images'
        seq_dir = nise_cfg.DEBUG.SEQ_DIR
        imglist = misc_utils.get_imagelist_from_dir(seq_dir)
        imglist = sorted(imglist)
        k = 0
        
        p0 = FrameItem(imglist[k], True)
        p0.detect_human(maskRCNN)
        p0.unify_bbox()
        p0.est_joints(simple_human_det_model)
        p0.assign_id(Q)
        p0.visualize(dataset)
        Q.append(p0)
        
        dicts = [p0.to_dict()]
        debug_print()
        for k in range(1, len(imglist)):
            debug_print('img ', imglist[k])
            new_p = FrameItem(imglist[k])
            new_p.detect_human(maskRCNN)
            new_p.gen_flow(flow_model, Q[-1].bgr_img)
            new_p.joint_prop(Q)
            new_p.unify_bbox()
            new_p.est_joints(simple_human_det_model)
            new_p.assign_id(Q, computer_joints_oks_mtx)
            new_p.visualize(dataset)
            # 2018-10-4  can run before this
            Q.append(new_p)
            dicts.append(new_p.to_dict())
        
        to_json = {'annolist': dicts}
        
        with open('f.json', 'w') as f:
            json.dump(to_json, f)
        
        # p0.flow_to_current = gen_rand_flow(1, 384, 640)
        # p0.flow_to_current.squeeze_()
        # p0.joint_prop(hh)
        
        # p0.visualize(dataset)
        # p0.img = gen_rand_img(batch_size, 384, 640)
        #
        debug_print('ALL DONE!')
