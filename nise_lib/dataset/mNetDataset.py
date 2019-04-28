# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, roc_curve
import numpy as np
import torch
import pprint
from torch.utils.data import Dataset
from collections import OrderedDict
from nise_lib.nise_functions import *
from nise_lib.nise_config import nise_cfg, nise_logger
from collections import OrderedDict
from nise_lib.dataset.dataset_util import *
from tron_lib.core.test_for_pt import _get_blobs
from tron_lib.core.config import cfg as tron_cfg


class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()
    
    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()
    
    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                it = self.popitem(last = False)[0]
                # debug_print("Popped",)


class mNetDataset(Dataset):
    def __init__(self, _nise_cfg, gt_anno_dir, pred_anno_dir, uni_box_dir, is_train):
        self.cfg = _nise_cfg
        self.gt_anno_dir = gt_anno_dir
        self.anno_file_names = sorted(get_type_from_dir(self.gt_anno_dir, ['.json']))
        self.pred_anno_dir = pred_anno_dir
        self.uni_box_dir = uni_box_dir
        self.is_train = is_train
        self.num_files_to_load = 250
        debug_print('Loading prediction jsons')
        json_path = self.cfg.PATH.SAVED_PRED_JSON_TRAIN_DATASET
        if os.path.exists(json_path):
            debug_print("Loading cached json predictions", json_path)
            self.pred_jsons = torch.load(json_path)
        else:
            debug_print("Loading  json predictions", json_path)
            self.pred_jsons = [json_load(os.path.join(self.pred_anno_dir,
                                                      PurePosixPath(file_name).stem + '.json'))['annolist'] for
                               file_name in self.anno_file_names[:self.num_files_to_load]]
            torch.save(self.pred_jsons, json_path)
        
        debug_print("Done. Loaded {} jsons".format(len(self.pred_jsons)))
        dataset_path = self.cfg.PATH.PRE_COMPUTED_TRAIN_DATASET if is_train \
            else self.cfg.PATH.PRE_COMPUTED_VAL_DATASET
        if os.path.exists(dataset_path):
            debug_print("Loading cached dataset", dataset_path)
            self.db = torch.load(dataset_path)
        else:
            debug_print("Loading  dataset", dataset_path)
            self.db = self._get_db()
        # self.db = self._get_db()
        debug_print("Loaded %d entries." % len(self), lvl = Levels.SUCCESS)
        self.cached_pkl = LimitedSizeDict(size_limit = 15)
    
    def __len__(self, ):
        return len(self.db)
    
    @log_time("Load dataset...")
    def _get_db(self):
        def add_pairs(idx1, idx2, l):
            for p_idx in idx1:
                for c_idx in idx2:
                    p_box = uni_boxes[prev_j][prev_img_file_path][p_idx]
                    c_box = uni_boxes[j][cur_img_file_path][c_idx]
                    u = torch.tensor([
                        min(p_box[0], c_box[0]).int().type(torch.float32),
                        min(p_box[1], c_box[1]).int().type(torch.float32),
                        max(p_box[2], c_box[2]).int().type(torch.float32),
                        max(p_box[3], c_box[3]).int().type(torch.float32),
                    ])
                    p_box = expand_vector_to_tensor(p_box[:4])
                    c_box = expand_vector_to_tensor(c_box[:4])
                    iou = tf_iou(p_box.numpy(), c_box.numpy())
                    if iou[0, 0] > self.cfg.TRAIN.IOU_THERS_FOR_NEGATIVE:
                        entry = torch.zeros(6)
                        entry[0] = p_idx
                        entry[1] = c_idx
                        entry[2:] = u
                        l.append(entry)
        
        np.set_printoptions(suppress = True)
        
        db = []
        total_num_pos = 0
        total_num_neg = 0
        for vid, file_name in enumerate(self.anno_file_names[:self.num_files_to_load]):
            if is_skip_video(nise_cfg, vid, file_name):
                # debug_print('Skip', vid, file_name)
                continue
            debug_print(vid, file_name)
            with open(file_name, 'r') as f:
                gt = json.load(f)['annolist']
            
            ppp = PurePosixPath(file_name)
            pred = self.pred_jsons[vid]
            
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
                        uni_boxes = torch.load(os.path.join(self.uni_box_dir, ppp.stem + '.pkl'))
                        prev_img_file_path = os.path.join(self.cfg.PATH.POSETRACK_ROOT, gt[prev_j]['image'][0]['name'])
                        cur_img_file_path = os.path.join(self.cfg.PATH.POSETRACK_ROOT, gt[j]['image'][0]['name'])
                        for k, v in matched.items():
                            # positive samples
                            p, c = v
                            add_pairs(p, c, pos_entries)
                        
                        num_pos = len(pos_entries)
                        
                        neg_entries = []
                        # use certain rules to sample negative entries
                        skip = 1
                        key_list = list(matched.keys())
                        while len(matched) > 1 and skip <= len(matched) - 1:
                            for k1 in range(len(key_list[:-skip])):
                                k2 = k1 + skip
                                k1 = key_list[k1]
                                k2 = key_list[k2]
                                p1, c1 = matched[k1]
                                p2, c2 = matched[k2]
                                add_pairs(p1, c2, neg_entries)
                                add_pairs(p2, c1, neg_entries)
                            skip += 1
                        
                        num_neg = len(neg_entries)
                        # debug_print('num_pos vs num_neg', num_pos, num_neg)
                        if num_pos + num_neg == 0:
                            continue
                        ent = {
                            'video_file': vid,
                            'prev_frame': prev_j,
                            'cur_frame': j,  # index of images in video seq
                            'pos': pos_entries,
                            'neg': neg_entries
                        }
                        # debug_print(pprint.pformat(ent))
                        db.append(ent)
                        total_num_pos += num_pos
                        total_num_neg += num_neg
                    prev_j = j
                    prev_id = cur_id
        # debug_print("Total num_pos vs num_neg", total_num_pos, total_num_neg, lvl = Levels.STATUS)
        
        dataset_path = self.cfg.PATH.PRE_COMPUTED_TRAIN_DATASET if self.is_train \
            else self.cfg.PATH.PRE_COMPUTED_VAL_DATASET
        # if not os.path.exists(dataset_path):
        debug_print("Saving cache dataset...", dataset_path)
        torch.save(db, dataset_path)
        debug_print('Done.')
        return db
    
    # @log_time('Getting item...')
    def __getitem__(self, idx):
        
        # debug_print('Get', idx)
        db_rec = copy.deepcopy(self.db[idx])
        pred = self.pred_jsons[db_rec['video_file']]
        prev_j = db_rec['prev_frame']
        cur_j = db_rec['cur_frame']
        pos = db_rec['pos']  # list of [p_box_idx,c_box_idx,union box]
        neg = db_rec['neg']
        
        start = time.time()
        # debug_print('load pred jsond', time.time() - start)
        prev_img_file_path = os.path.join(self.cfg.PATH.POSETRACK_ROOT, pred[prev_j]['image'][0]['name'])
        cur_img_file_path = os.path.join(self.cfg.PATH.POSETRACK_ROOT, pred[cur_j]['image'][0]['name'])
        img_file_paths = [prev_img_file_path, cur_img_file_path]
        inp = []
        scales = []
        for img_file_path in img_file_paths:
            start = time.time()
            original_img = cv2.imread(img_file_path)  # with original size
            original_img = aug_img(original_img)
            original_img=rectify_img_size(original_img)
            inputs, im_scale = _get_blobs(original_img, None, tron_cfg.TEST.SCALE, tron_cfg.TEST.MAX_SIZE)
            inp.append(torch.from_numpy(inputs['data']))
            scales.append(im_scale[0])
        
        if self.is_train:
            # sample from pos
            if len(pos) >= nise_cfg.TRAIN.POS_SAMPLES_PER_IMAGE:  # have enough pos
                num_neg_to_sample = min(nise_cfg.TRAIN.NEG_SAMPLES_PER_IMAGE, len(neg))
                # if neg not enough, fill training data with pos
                difference = nise_cfg.TRAIN.NEG_SAMPLES_PER_IMAGE - num_neg_to_sample
                num_pos_to_sample = min(nise_cfg.TRAIN.POS_SAMPLES_PER_IMAGE + difference, len(pos))
            else:
                num_pos_to_sample = len(pos)
                # have enough neg to sample, and try to fill the training data
                difference = nise_cfg.TRAIN.POS_SAMPLES_PER_IMAGE - num_pos_to_sample
                num_neg_to_sample = min(nise_cfg.TRAIN.NEG_SAMPLES_PER_IMAGE + difference, len(neg))
            
            pos_rand_idx = torch.randperm(len(pos))
            pos_samples = [pos[i] for i in pos_rand_idx[:num_pos_to_sample]]
            neg_rand_idx = torch.randperm(len(neg))
            neg_samples = [neg[i] for i in neg_rand_idx[:num_neg_to_sample]]
            num_total_samples = nise_cfg.TRAIN.POS_SAMPLES_PER_IMAGE + nise_cfg.TRAIN.NEG_SAMPLES_PER_IMAGE
        else:
            num_pos_to_sample = len(pos)
            num_neg_to_sample = len(neg)
            pos_samples = pos
            neg_samples = neg
            
            num_total_samples = num_pos_to_sample + num_neg_to_sample
        
        all_samples = pos_samples + neg_samples
        all_samples_tensor = -torch.ones([num_total_samples, 6])
        if all_samples:
            all_samples_tensor[:len(all_samples), :] = torch.stack(all_samples)
        # debug_print(all_samples, lvl = Levels.ERROR)
        p_pred_joints, p_pred_joints_scores = get_joints_from_annorects(pred[prev_j]['annorect'])
        c_pred_joints, c_pred_joints_scores = get_joints_from_annorects(pred[cur_j]['annorect'])
        _, num_joints, _ = p_pred_joints.shape
        joints_heatmap = -torch.ones(
            [num_total_samples, num_joints * 2, nise_cfg.MODEL.FEATURE_MAP_RESOLUTION,
             nise_cfg.MODEL.FEATURE_MAP_RESOLUTION])
        for i, s in enumerate(all_samples):
            p_box_idx = all_samples_tensor[i, 0].int()
            c_box_idx = all_samples_tensor[i, 1].int()
            p_joints, p_joints_scores = p_pred_joints[p_box_idx, :, :2], p_pred_joints_scores[p_box_idx, :]
            c_joints, c_joints_scores = c_pred_joints[c_box_idx, :, :2], c_pred_joints_scores[c_box_idx, :]
            u = all_samples_tensor[i, 2:]
            u_box_size = torch.tensor([
                u[2] - u[0],  # W
                u[3] - u[1],  # H
            ]).numpy()
            p_joints = get_union_box_based_joints(u, p_joints)
            c_joints = get_union_box_based_joints(u, c_joints)
            
            p_joints_hmap = gen_joints_hmap(u_box_size,
                                            (nise_cfg.MODEL.FEATURE_MAP_RESOLUTION,
                                             nise_cfg.MODEL.FEATURE_MAP_RESOLUTION),
                                            p_joints, p_joints_scores)
            c_joints_hmap = gen_joints_hmap(u_box_size,
                                            (nise_cfg.MODEL.FEATURE_MAP_RESOLUTION,
                                             nise_cfg.MODEL.FEATURE_MAP_RESOLUTION),
                                            c_joints, c_joints_scores)
            joints_heatmap[i, :num_joints] = p_joints_hmap
            joints_heatmap[i, num_joints:] = c_joints_hmap
        
        labels = -torch.ones(num_total_samples)
        labels[:num_pos_to_sample] = 1
        labels[num_pos_to_sample:num_neg_to_sample + num_pos_to_sample] = 0
        # debug_print(db_rec)
        return torch.cat(inp), torch.tensor(scales), joints_heatmap, labels, {
            'all_samples': all_samples_tensor,
            'num_total_samples': num_total_samples,
            'db_entry': db_rec
        }
    
    @log_time('Evaluating model...')
    def eval(self, gt: np.ndarray, pred_scores: np.ndarray):
        '''
        :param gt: vector, (num_gt,)
        :param pred_scores: not ordered
        :return:
        '''
        
        # gt = np.array([i['is_same'] for i in self.db])
        assert len(gt) == len(pred_scores)
        
        prec, rec, pr_thres = precision_recall_curve(gt, pred_scores)
        ap = average_precision_score(gt, pred_scores)
        fpr, tpr, roc_thres = roc_curve(gt, pred_scores)
        auc = roc_auc_score(gt, pred_scores)
        return {
            'prec': prec, 'rec': rec, "pr_thres": pr_thres, 'ap': ap,
            'fpr': fpr, 'tpr': tpr, 'roc_thres': roc_thres, 'auc': auc
        }
