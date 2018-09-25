#
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: U S E   P R E T R A I N E D   M O D E L S   T O   T E S T
#       R E - I M P L E M E N T   S I M P L E   B A S E L I N E :
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#


import sys

sys.path.append('../flownet2-pytorch/')
sys.path.append('../simple-baseline-pytorch/')
# !/usr/bin/env python

import torch
import torch.nn as nn

import argparse
import os
import sys
import subprocess
import setproctitle
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *
from collections import deque

# local packages
import flow_models
import flow_losses
import flow_datasets
from flownet_utils import flow_utils, tools
from nise_utils.imutils import *
from nise_config import cfg as nise_cfg

# fp32 copy of parameters for update
global param_copy


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
human_detection_bboxes = torch.zeros([1])
# ─── FROM FLOWNET 2.0 ───────────────────────────────────────────────────────────
if True:
    from nise_functions import flow_init_parser_and_tools, load_flow_model
    parser = argparse.ArgumentParser()
    flow_init_parser_and_tools(parser, tools)
    args, rest = parser.parse_known_args()
    flow_model = load_flow_model(args,parser,tools)
if nise_cfg.DEBUG.DEVELOPING and nise_cfg.DEBUG.FLOW:
    from nise_debugging_func import *
    from nise_functions import *
    im_dir = '_dev/000001_bonn'
    # img_files = os.listdir(im_dir)
    two_img_file_name = ['06115.jpg','06116.jpg']
    two_imgs = [load_image(os.path.join(im_dir,path)) for path in two_img_file_name]
    tis = [resize(i, 640, 384) for i in two_imgs]
    ti = torch.stack(tis,1)
    o = pred_flow(ti, flow_model)
    print()
# ────────────────────────────────────────────────────────────────────────────────


# ─── PREDICT NEXT FRAME'S BOXES FROM FLOW AND PREV FRAME'S JOINTS ───────────────
if False:
    prev_frame_joints = torch.zeros([1])
    next_frame_joints = get_flowed_pos(flow_field, prev_frame_joints)
    flowed_bboxes = get_flowed_bboxes(next_frame_joints, image_size)
    all_bboxes = torch.stack(human_detection_bboxes, flowed_bboxes)
# ────────────────────────────────────────────────────────────────────────────────
# ─── NMS ────────────────────────────────────────────────────────────────────────


if False:
    final_predicted_bboxes = nms(all_bboxes)
# ────────────────────────────────────────────────────────────────────────────────

# ─── FROM SIMPLE BASELINE ───────────────────────────────────────────────────────
if False:
    from nise_functions import load_simple_model
    # ─── DEFINE LOSS FUNCTION  AND OPTIMIZER ────────────────────────────────────────
    simple_human_det_model = load_simple_model()
    criterion = JointsMSELoss(
        use_target_weight = simple_cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    
    print('Simple pose detector loaded.')
    batch_size = 8
    person_inputs = torch.zeros([batch_size, 3, 384, 384])
    person_hmap_target = torch.zeros([batch_size, 16, 96, 96])
    person_hmap_target_weight = torch.zeros([batch_size, 16, 1])
    meta = dict()
    joint_heatmap = simple_human_det_model(person_inputs)
    
    # ─── FOR VALIDATION ─────────────────────────────────────────────────────────────
    
    if simple_cfg.TEST.FLIP_TEST:
        # this part is ugly, because pytorch has not supported negative index
        # input_flipped = model(input[:, :, :, ::-1])
        input_flipped = np.flip(person_inputs.cpu().numpy(), 3).copy()
        input_flipped = torch.from_numpy(input_flipped).cuda()
        output_flipped = simple_human_det_model(input_flipped)
        output_flipped = flip_back(output_flipped.cpu().numpy(),
                                   val_dataset.flip_pairs)
        output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
        
        # feature is not aligned, shift flipped heatmap for higher accuracy
        if simple_cfg.TEST.SHIFT_HEATMAP:
            output_flipped[:, :, :, 1:] = \
                output_flipped.clone()[:, :, :, 0:-1]
        
        joint_heatmap = (joint_heatmap + output_flipped) * 0.5
    
    person_hmap_target = person_hmap_target.cuda(non_blocking = True)
    person_hmap_target_weight = person_hmap_target_weight.cuda(
        non_blocking = True)
    
    loss = criterion(joint_heatmap, person_hmap_target,
                     person_hmap_target_weight)
    
    # ─── FOR TRAIN ──────────────────────────────────────────────────────────────────
    
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    
    c = meta['center'].numpy()
    s = meta['scale'].numpy()
    score = meta['score'].numpy()
    #     now we already have all predicted joints' position
    preds, maxvals = get_final_preds(
        simple_cfg, joint_heatmap.clone().cpu().numpy(), c, s)  # bs x 16 x 2,  bs x 16 x 1
    # maxval - 不超过一，应该都是评分大小

# ───  ───────────────────────────────────────────────────────────────────────────

# ─── ASSOCIATE IDS USING GREEDY MATCHING ────────────────────────────────────────
""" input: distance matrix; output: correspondence   """
# ────────────────────────────────────────────────────────────────────────────────
