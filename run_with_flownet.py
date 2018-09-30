#
# ──────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: U S E   P R E T R A I N E D   M O D E L S   T O   T E S T
#       R E - I M P L E M E N T   S I M P L E   B A S E L I N E :
# ────────────────────────────────────────────────────────────────────────────
#
import sys

sys.path.append('../flownet2-pytorch/')
sys.path.append('../simple-baseline-pytorch/')
sys.path.append('../Detectron.pytorch/')
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
import cv2

# local packages
import flow_models
import flow_losses
import flow_datasets
from flownet_utils import flow_utils, tools
from nise_utils.imutils import *
# from nise_lib.frameitem import FrameItem
from nise_config import cfg as nise_cfg
from nise_functions import *
from nise_debugging_func import *

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

batch_size = 8

# ─── FROM FLOWNET 2.0 ───────────────────────────────────────────────────────────
if False:
    parser = argparse.ArgumentParser()
    flow_init_parser_and_tools(parser, tools)
    flow_args, rest = parser.parse_known_args()
    flow_model = load_flow_model(flow_args, parser, tools)
if nise_cfg.DEBUG.DEVELOPING and nise_cfg.DEBUG.FLOW:
    im_dir = '_dev/000001_bonn'
    
    two_img_file_name = ['06115.jpg', '06116.jpg']
    two_imgs = [load_image(os.path.join(im_dir, path))
                for path in two_img_file_name]
    tis = [resize(i, 640, 384) for i in two_imgs]
    ti = torch.stack(tis, 1)
    o = pred_flow(ti, flow_model)
    print()
# ────────────────────────────────────────────────────────────────────────────────


# ─── FROM SIMPLE BASELINE ───────────────────────────────────────────────────────
if False:
    
    # ─── DEFINE LOSS FUNCTION AND OPTIMIZER ────────────────────────────────────────
    simple_human_det_model = load_simple_model()
    criterion = JointsMSELoss(
        use_target_weight = simple_cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    
    print('Simple pose detector loaded.')
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

# ─── HUMAN DETECT ───────────────────────────────────────────────────────────────
if nise_cfg.DEBUG.HUMAN:
    import tron_lib.utils.misc as misc_utils
    from tron_lib.core.config import cfg as tron_cfg
    from tron_lib.core.test_for_pt import im_detect_all
    from tron_lib.core.test_engine import initialize_model_from_cfg
    import tron_lib.utils.vis as vis_utils
    
    human_detect_args = human_detect_parse_args()
    
    maskRCNN, dataset = load_human_detect_model(human_detect_args, tron_cfg)
    
    out_dir = 'images_out'
    maskRCNN.eval()
    # imglist = misc_utils.get_imagelist_from_dir('images')
    # num_images = len(imglist)
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    #
    # for i in range(num_images):
    #     print('img', i)
    #     im = cv2.imread(imglist[i])
    #     assert im is not None
    #
    #     cls_boxes = im_detect_all(maskRCNN, im)
    # cls_boxes : list of size 81. each entry is an ndarray with size num_bboxes x 5[coor + score]. num_bboxes is undetermined.
    #
    # im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
    # vis_utils.vis_one_image(
    #     im[:, :, ::-1],  # BGR -> RGB for visualization
    #     im_name,
    #     out_dir,
    #     cls_boxes,
    #     None,
    #     None,
    #     dataset=dataset,
    #     box_alpha=0.3,
    #     show_class=True,
    #     thresh=0.7,
    #     kp_thresh=2
    # )

# init deque (why deque??)
Q = deque(maxlen = nise_cfg.ALG.DEQUE_CAPACITY)
human_detection_bboxes = torch.zeros([1])

if nise_cfg.DEBUG.FRAME:
    
    class FrameItem:
        '''
            Item in Deque, storing image (for flow generation), joints(for propagation),
             trackids.
             One image or a batch of images?
        '''
        
        def __init__(self, is_first = False):
            '''

            :param is_first: is this frame the first of the sequence
            '''
            self.is_first = is_first
            self.img_name = ''
            self.img = None  # with original size? YES   ~~no, resize to some fixed size~~
            # num_person x num_joints x 2
            self.joints = None  # should be LongTensor since we are going to use them as index
            self.new_joints = None  # for similarity
            self.flow_to_current = None  # 2, h, w
            self.human_bboxes = None
            self.joint_prop_bboxes = None
            self.unified_bboxes = None
            self.human_ids = None
            # ─── FLAGS FOR CORRENT ORDER ─────────────────────────────────────
            self.human_detected = False
            self.flow_calculated = False
            self.joints_proped = False
            self.bboxes_unified = False
            self.joints_detected = False
            self.id_assigned = False
        
        def detect_human(self, detector):
            '''

            :param detector:
            :return: human is represented as tensor of size num_people x 4. The result is NMSed.
            '''
            cls_boxes = im_detect_all(detector, self.img)
            self.human_bboxes = cls_boxes[1]  # person is the first class of coco， 0 for background
            self.human_detected = True
        
        def gen_flow(self, prev_frame_img = None):
            # No precedent functions, root
            if self.is_first:  # if this is first frame
                return
            resized_img = resize(self.img, *nise_cfg.DATA.flow_input_size)
            resized_prev = resize(
                prev_frame_img, *nise_cfg.DATA.flow_input_size)
            ti = torch.stack([resized_prev, resized_img])
            self.flow_to_current = pred_flow(ti, flow_model)
            self.flow_calculated = True
        
        def joint_prop(self, prev_frame_joints):
            '''

            :param prev_frame_joints: 2 x num_people x num_joints
            :return:
            '''
            # preprocess
            if not ((prev_frame_joints.shape[2] == nise_cfg.DATA.num_joints and prev_frame_joints.shape[0] == 2) or (
                    prev_frame_joints.shape[2] == 2 and prev_frame_joints.shape[1] == nise_cfg.DATA.num_joints)):
                raise ValueError(
                    'Size not matched, current size ' + str(prev_frame_joints.shape))
            if prev_frame_joints.shape[2] == nise_cfg.DATA.num_joints:
                # :param prev_frame_joints: 2x num_people x num_joints
                prev_frame_joints = torch.transpose(prev_frame_joints, 0, 1)
                prev_frame_joints = torch.transpose(prev_frame_joints, 1, 2)
            
            if not self.flow_calculated:
                raise ValueError('Flow not calculated yet.')
            
            new_joints = torch.zeros(prev_frame_joints.shape)
            for person in range(new_joints.shape[0]):
                for joint in range(new_joints.shape[1]):
                    joint_pos = prev_frame_joints[person, joint, :]  # x,y
                    # x, y
                    joint_flow = self.flow_to_current[:,
                                 joint_pos[1], joint_pos[0]]
                    new_joints[person, joint,
                    :] = prev_frame_joints[person, joint, :].float() + joint_flow
            # for similarity
            self.new_joints = new_joints
            # calc new bboxes from new joints
            min_xs, _ = torch.min(new_joints[:, :, 0], 1)
            min_ys, _ = torch.min(new_joints[:, :, 1], 1)
            max_xs, _ = torch.max(new_joints[:, :, 0], 1)
            max_ys, _ = torch.max(new_joints[:, :, 1], 1)
            # extend
            ws = max_xs - min_xs
            hs = max_ys - min_ys
            ws = ws * nise_cfg.DATA.bbox_extend_factor[0]
            hs = hs * nise_cfg.DATA.bbox_extend_factor[1]
            min_xs -= ws
            max_xs += ws
            min_ys -= hs
            max_xs += hs
            min_xs.clamp_(0, nise_cfg.DATA.flow_input_size[0])
            max_xs.clamp_(0, nise_cfg.DATA.flow_input_size[0])
            min_ys.clamp_(0, nise_cfg.DATA.flow_input_size[1])
            max_ys.clamp_(0, nise_cfg.DATA.flow_input_size[1])
            
            self.joint_prop_bboxes = torch.stack([
                min_xs, min_ys, max_xs, max_ys
            ], 1)
            # assert(self.joint_prop_bboxes.shape )
            self.joints_proped = True
        
        def unify_bbox(self):
            if not self.joints_proped or not self.human_detected:
                raise ValueError(
                    'Should run human detection and joints propagation first')
            all_bboxes = torch.stack(self.human_bboxes, self.joint_prop_bboxes)
            self.bboxes_unified = True
            raise NotImplementedError
        
        def assign_id(self, Q, dist_func = None):
            
            # ─── ASSOCIATE IDS USING GREEDY MATCHING ────────────────────────────────────────
            # ────────────────────────────────────────────────────────────────────────────────
            """ input: distance matrix; output: correspondence   """
            if not self.joints_detected:
                raise ValueError('Should detect joints first')
            self.human_ids = []
            if self.is_first:
                # if it's the first frame, just assign every
                for i in range(self.human_bboxes.shape[0]):
                    self.human_ids.append(i)
            else:
                if dist_func is None: raise NotImplementedError('Should pass a distance function in')
                prev_frame_joints = Q[-1].joints
                prev_ids = Q[-1].human_ids
                num_human_prev = prev_frame_joints.shape[0]
                num_human_cur = self.joints.shape[0]
                distance_matrix = torch.zeros(num_human_prev, num_human_cur)
                for prev in range(num_human_prev):
                    for cur in range(num_human_cur):
                        distance_matrix[prev, cur] = dist_func(prev_frame_joints[prev], self.joints[cur])
                raise NotImplementedError
            self.id_assigned = True
        
        def est_joints(self, joint_detector):
            if not self.bboxes_unified:
                raise ValueError('Should unify bboxes first')
            
            self.joints_detected = True
            
            raise NotImplementedError
        
        def visualize(self, dataset, out_dir = 'images_out'):
            classes = [[]] * 81
            classes[1] = self.human_bboxes
            vis_utils.vis_one_image(
                self.img[:, :, ::-1],  # BGR -> RGB for visualization
                self.img_name,
                out_dir,
                classes,
                None,
                None,
                dataset = dataset,
                box_alpha = 0.3,
                show_class = True,
                thresh = 0.7,
                kp_thresh = 2
            )
    
    
    p0 = FrameItem(True)
    
    if nise_cfg.DEBUG.DEVELOPING:
        hh = gen_rand_joints(3, 384, 640)
        hh = hh.int()
        print(hh.dtype)
        p0.img_name = 'images/000000.jpg'
        im = cv2.imread(p0.img_name)
        p0.img = im
        # p0.flow_to_current = gen_rand_flow(1, 384, 640)
        # p0.flow_to_current.squeeze_()
        # p0.joint_prop(hh)
        
        p0.detect_human(maskRCNN)
        p0.visualize(dataset)
        # p0.img = gen_rand_img(batch_size, 384, 640)
        #
        # Q.append(p0)
        #
        # while False:
        #     new_p = FrameItem()
        #     new_p.detect_human(None)
        #     new_p.gen_flow(Q[-1].img)
        #     new_p.joint_prop(Q[-1].joints)
        #     new_p.unify_bbox()
        #     new_p.est_joints(simple_human_det_model)
        #     new_p.assign_id(Q)
        #     Q.append(new_p)
        print()
