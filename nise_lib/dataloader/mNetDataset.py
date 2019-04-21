# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import cv2
import numpy as np
import torch
import pprint
from torch.utils.data import Dataset
from collections import OrderedDict
from nise_lib.nise_functions import *


class mNetDataset(Dataset):
    def __init__(self, _nise_cfg, gt_anno_dir, pred_anno_dir, uni_box_dir, is_train, maskRCNN):
        self.cfg = _nise_cfg
        self.gt_anno_dir = gt_anno_dir
        self.pred_anno_dir = pred_anno_dir
        self.uni_box_dir = uni_box_dir
        self.is_train = is_train
        self.maskRCNN = maskRCNN
        
        dataset_path = self.cfg.PATH.PRE_COMPUTED_TRAIN_DATASET if is_train \
            else self.cfg.PATH.PRE_COMPUTED_VAL_DATASET
        if os.path.exists(dataset_path):
            debug_print("Loading cached dataset", dataset_path)
            self.db = torch.load(dataset_path)
        else:
            self.db = self._get_db()
        debug_print("Loaded %d entries." % len(self), lvl = Levels.SUCCESS)
    
    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError
    
    def __len__(self, ):
        return len(self.db)
    
    @log_time("Load dataset...")
    def _get_db(self):
        np.set_printoptions(suppress = True)
        
        anno_file_names = get_type_from_dir(self.gt_anno_dir, ['.json'])
        anno_file_names = sorted(anno_file_names)
        db = []
        total_num_pos = 0
        total_num_neg = 0
        for vid, file_name in enumerate(anno_file_names):
            if is_skip_video(nise_cfg, vid, file_name):
                debug_print('Skip', vid, file_name)
                continue
            debug_print(vid, file_name)
            with open(file_name, 'r') as f:
                gt = json.load(f)['annolist']
            
            ppp = PurePosixPath(file_name)
            pred_json_path = os.path.join(self.pred_anno_dir, ppp.stem + '.json')
            with open(pred_json_path, 'r') as f:
                pred = json.load(f)['annolist']
            
            prev_id = None
            prev_j = -1
            for j, frame in enumerate(gt):
                
                # debug_print(cur_id, lvl = Levels.STATUS)
                
                if frame['is_labeled'][0]:
                    img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, frame['image'][0]['name'])
                    # debug_print(j, img_file_path)
                    gt_annorects = frame['annorect']
                    pred_annorects = pred[j]['annorect']
                    gt_annorects = removeRectsWithoutPoints(gt_annorects)
                    pred_annorects = removeRectsWithoutPoints(pred_annorects)
                    
                    cur_id = find_gt_for_det_and_assignID(gt_annorects, pred_annorects, nise_cfg)
                    if prev_j != -1:
                        matched = {
                            int(i): [
                                np.where(prev_id == i)[0].tolist(),
                                np.where(cur_id == i)[0].tolist()  # may be empty
                            ] for i in prev_id if i != -1
                        }
                        pos_entries = []
                        for k, v in matched.items():
                            # positive samples
                            p, c = v
                            for p_idx in p:
                                for c_idx in c:
                                    entry = OrderedDict({
                                        'video_file': vid,
                                        'prev_frame': prev_j, 'cur_frame': j,  # index of images in video seq
                                        'p_box_idx': p_idx, 'c_box_idx': c_idx,  # index of box in annotations
                                        'is_same': True
                                    })
                                    pos_entries.append(entry)
                        db.extend(pos_entries)
                        
                        num_pos = len(pos_entries)
                        num_neg = self.cfg.TRAIN.neg_pos_ratio * num_pos
                        neg_entries = []
                        
                        # use certain rules to sample negative entries
                        skip = 1
                        while len(matched) > 1 and len(neg_entries) < num_neg and skip <= len(matched) - 1:
                            key_list = list(matched.keys())
                            for k1 in range(len(key_list[:-skip])):
                                k2 = k1 + skip
                                k1 = key_list[k1]
                                k2 = key_list[k2]
                                p1, c1 = matched[k1]
                                p2, c2 = matched[k2]
                                for p_idx in p1:
                                    for c_idx in c2:
                                        entry = OrderedDict({
                                            'video_file': vid,
                                            'prev_frame': prev_j, 'cur_frame': j,  # index of images in video seq
                                            'p_box_idx': p_idx, 'c_box_idx': c_idx,  # index of box in annotations
                                            'is_same': False
                                        })
                                        neg_entries.append(entry)
                                for p_idx in p2:
                                    for c_idx in c1:
                                        entry = OrderedDict({
                                            'video_file': vid,
                                            'prev_frame': prev_j, 'cur_frame': j,  # index of images in video seq
                                            'p_box_idx': p_idx, 'c_box_idx': c_idx,  # index of box in annotations
                                            'is_same': False
                                        })
                                        neg_entries.append(entry)
                            skip += 1
                        
                        db.extend(neg_entries)
                        # pprint.pprint(neg_entries)
                        num_neg = len(neg_entries)
                        # print('num_pos vs num_neg', num_pos, num_neg)
                        total_num_pos += num_pos
                        total_num_neg += num_neg
                    prev_j = j
                    prev_id = cur_id
        debug_print("Total num_pos vs num_neg", total_num_pos, total_num_neg, lvl = Levels.STATUS)
        return db
    
    def gen_joints_hmap(self, fmap_size, hmap_size, joints, joint_scores):
        '''
        All op is writen in numpy, and converted to tensor at the end
            # copy from simple-baseline

        :param fmap_size: H,W,fmap_scale,im_scale
        :param hmap_size: (h,w)
        :param joints: 15x2
        :param joint_scores: (15,)
        :return:
        '''
        # 暂时不对，这个是对应到原图，应该对应到 union box大小
        num_joints = joints.shape[0]
        H, W, fmap_scale, im_scale = fmap_size
        ori_img_size = np.array([H, W]) / fmap_scale / im_scale
        
        target = np.zeros((num_joints,
                           hmap_size[1],
                           hmap_size[0]),
                          dtype = np.float32)  # nj x w x h ???
        target_weight = joint_scores  # not visibility since all is visible
        
        feat_stride = ori_img_size / hmap_size
        tmp_size = self.cfg.MODEL.JOINT_MAP_SIGMA * 3
        for joint_id in range(num_joints):
            # joint coord is [x,y]
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            
            if ul[0] >= hmap_size[0] or ul[1] >= hmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue
            
            size = 2 * tmp_size + 1  # 13
            x = np.arange(0, size, 1, np.float32)  # (13,)
            y = x[:, np.newaxis]  # (1,13)
            x0 = y0 = size // 2  # 6
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.cfg.MODEL.JOINT_MAP_SIGMA ** 2))
            
            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], hmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], hmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], hmap_size[0])
            img_y = max(0, ul[1]), min(br[1], hmap_size[1])
            
            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return to_torch(target)
    
    def __getitem__(self, idx):
        assert self.maskRCNN is not None, "maskRCNN must be instantiated."
        anno_file_names = sorted(get_type_from_dir(self.gt_anno_dir, ['.json']))
        db_rec = copy.deepcopy(self.db[idx])
        file_name = anno_file_names[db_rec['video_file']]
        prev_j = db_rec['prev_frame']
        cur_j = db_rec['cur_frame']
        p_box_idx = db_rec['p_box_idx']
        c_box_idx = db_rec['c_box_idx']
        is_same = db_rec['is_same']
        
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        p = PurePosixPath(file_name)
        # load boxes
        prev_img_file_path = os.path.join(self.cfg.PATH.POSETRACK_ROOT, gt[prev_j]['image'][0]['name'])
        cur_img_file_path = os.path.join(self.cfg.PATH.POSETRACK_ROOT, gt[cur_j]['image'][0]['name'])
        
        uni_boxes = torch.load(os.path.join(self.uni_box_dir, p.stem + '.pkl'))
        p_box = uni_boxes[prev_j][prev_img_file_path][p_box_idx]
        c_box = uni_boxes[cur_j][cur_img_file_path][c_box_idx]
        u = torch.tensor([[
            min(p_box[0], c_box[0]),
            min(p_box[1], c_box[1]),
            max(p_box[2], c_box[2]),
            max(p_box[3], c_box[3]),
        ]])
        assert (u.shape[0] == 1 and u.shape[1] == 4)
        # get union feature map
        p_fmap_pkl = torch.load(os.path.join(self.cfg.PATH.FPN_PKL_DIR, p.stem + '-%03d' % (prev_j) + '.pkl'))
        c_fmap_pkl = torch.load(os.path.join(self.cfg.PATH.FPN_PKL_DIR, p.stem + '-%03d' % (cur_j) + '.pkl'))
        p_fmap = get_box_fmap(self.maskRCNN, p_fmap_pkl, u)
        c_fmap = get_box_fmap(self.maskRCNN, c_fmap_pkl, u)
        # load joints
        with open(os.path.join(self.pred_anno_dir, p.stem + '.json'), 'r') as f:
            pred = json.load(f)['annolist']
        pred_joints, pred_joints_scores = get_joints_from_annorects(pred[prev_j]['annorect'])
        p_joints, p_joints_scores = pred_joints[p_box_idx, :, :2], pred_joints_scores[p_box_idx, :]
        pred_joints, pred_joints_scores = get_joints_from_annorects(pred[cur_j]['annorect'])
        c_joints, c_joints_scores = pred_joints[c_box_idx, :, :2], pred_joints_scores[c_box_idx, :]
        # gen joints hmap
        fmaps, im_scale = p_fmap_pkl['fmap'], p_fmap_pkl['scale']
        highest_res_fmap = fmaps[3]
        C, H, W = highest_res_fmap.shape
        C, mH, mW = p_fmap.shape
        fmap_size = [H, W, 0.25, im_scale]
        
        p_joints_hmap = self.gen_joints_hmap(fmap_size,
                                             (mH, mW),
                                             p_joints, p_joints_scores)
        c_joints_hmap = self.gen_joints_hmap(fmap_size,
                                             (mH, mW),
                                             c_joints, c_joints_scores)
        # assemble
        inputs = torch.stack(p_fmap, c_fmap, p_joints_hmap, c_joints_hmap)
        
        return [inputs, is_same]
