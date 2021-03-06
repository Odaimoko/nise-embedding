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
    def __init__(self, _nise_cfg, gt_anno_dir, pred_anno_dir, uni_box_dir, is_train, maskRCNN = None):
        self.cfg = _nise_cfg
        self.gt_anno_dir = gt_anno_dir
        self.anno_file_names = sorted(get_type_from_dir(self.gt_anno_dir, ['.json']))
        self.pred_anno_dir = pred_anno_dir
        self.uni_box_dir = uni_box_dir
        self.is_train = is_train
        
        if maskRCNN is not None:
            if isinstance(maskRCNN, mynn.DataParallel):
                maskRCNN = list(maskRCNN.children())[0]
            else:
                maskRCNN = maskRCNN
            self.conv_body = maskRCNN.Conv_Body
        else:
            self.conv_body = None
        
        self.dataset_path = self.cfg.PATH.PRE_COMPUTED_TRAIN_DATASET if is_train \
            else self.cfg.PATH.PRE_COMPUTED_VAL_DATASET
        self.dataset_path += '.pairwise'
        if os.path.exists(self.dataset_path):
            debug_print("Loading cached dataset", self.dataset_path)
            self.db = torch.load(self.dataset_path)
        else:
            self.db = self._get_db()
        
        debug_print("Loaded %d entries." % len(self), lvl = Levels.SUCCESS)
        self.dict_size = 20
        self.cached_pkl = LimitedSizeDict(size_limit = self.dict_size)
        self.cache_hit = 0
        self.cache_unhit = 0
        self.cached_boxes = LimitedSizeDict(size_limit = self.dict_size)
        self.cached_pred = LimitedSizeDict(size_limit = self.dict_size)
    
    def __len__(self, ):
        return len(self.db)
    
    @log_time("Load dataset...")
    def _get_db(self):
        np.set_printoptions(suppress = True)
        
        db = []
        total_num_pos = 0
        total_num_neg = 0
        for vid, file_name in enumerate(self.anno_file_names[:3]):
            if is_skip_video(nise_cfg, vid, file_name):
                # debug_print('Skip', vid, file_name)
                continue
            # debug_print(vid, file_name)
            with open(file_name, 'r') as f:
                gt = json.load(f)['annolist']
            
            ppp = PurePosixPath(file_name)
            pred_json_path = os.path.join(self.pred_anno_dir, ppp.stem + '.json')
            with open(pred_json_path, 'r') as f:
                pred = json.load(f)['annolist']
            
            prev_id = None
            prev_j = -1
            for j, frame in enumerate(gt):
                
                # # debug_print(cur_id, lvl = Levels.STATUS)
                
                if frame['is_labeled'][0]:
                    img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, frame['image'][0]['name'])
                    # # debug_print(j, img_file_path)
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
                            for p_idx in p:
                                for c_idx in c:
                                    p_box = uni_boxes[prev_j][prev_img_file_path][p_idx]
                                    c_box = uni_boxes[j][cur_img_file_path][c_idx]
                                    p_box = expand_vector_to_tensor(p_box[:4])
                                    c_box = expand_vector_to_tensor(c_box[:4])
                                    iou = tf_iou(p_box.numpy(), c_box.numpy())
                                    if iou[0, 0] > self.cfg.TRAIN.IOU_THERS_FOR_NEGATIVE:
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
                        while len(matched) > 1 and skip <= len(matched) - 1:
                            key_list = list(matched.keys())
                            for k1 in range(len(key_list[:-skip])):
                                k2 = k1 + skip
                                k1 = key_list[k1]
                                k2 = key_list[k2]
                                p1, c1 = matched[k1]
                                p2, c2 = matched[k2]
                                for p_idx in p1:
                                    for c_idx in c2:
                                        p_box = uni_boxes[prev_j][prev_img_file_path][p_idx]
                                        c_box = uni_boxes[j][cur_img_file_path][c_idx]
                                        p_box = expand_vector_to_tensor(p_box[:4])
                                        c_box = expand_vector_to_tensor(c_box[:4])
                                        iou = tf_iou(p_box.numpy(), c_box.numpy())
                                        if iou[0, 0] > self.cfg.TRAIN.IOU_THERS_FOR_NEGATIVE:
                                            entry = OrderedDict({
                                                'video_file': vid,
                                                'prev_frame': prev_j, 'cur_frame': j,  # index of images in video seq
                                                'p_box_idx': p_idx, 'c_box_idx': c_idx,  # index of box in annotations
                                                'is_same': False
                                            })
                                            neg_entries.append(entry)
                                for p_idx in p2:
                                    for c_idx in c1:
                                        p_box = uni_boxes[prev_j][prev_img_file_path][p_idx]
                                        c_box = uni_boxes[j][cur_img_file_path][c_idx]
                                        p_box = expand_vector_to_tensor(p_box[:4])
                                        c_box = expand_vector_to_tensor(c_box[:4])
                                        iou = tf_iou(p_box.numpy(), c_box.numpy())
                                        if iou[0, 0] > self.cfg.TRAIN.IOU_THERS_FOR_NEGATIVE:
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
        # debug_print("Total num_pos vs num_neg", total_num_pos, total_num_neg, lvl = Levels.STATUS)
        
        if not os.path.exists(self.dataset_path):
            # debug_print("Saving cached dataset...", self.dataset_path)
            self.db = torch.save(db, self.dataset_path)
            # debug_print('Done.')
        return db
    
    # @log_time('Getting item...')
    def __getitem__(self, idx):
        
        def load_from_cache(fn, d: LimitedSizeDict):
            if not fn in d.keys():
                self.cache_unhit += 1
                # debug_print("Not hit cache", self.cache_unhit)
                if 'json' in fn:
                    with open(fn, 'r') as f:
                        d[fn] = json.load(f)['annolist']
                else:
                    d[fn] = torch.load(fn)
            else:
                self.cache_hit += 1
                # debug_print("hit cache", self.cache_hit)
            return d[fn]
        
        # debug_print('Get', idx)
        db_rec = self.db[idx]
        file_name = self.anno_file_names[db_rec['video_file']]
        prev_j = db_rec['prev_frame']
        cur_j = db_rec['cur_frame']
        p_box_idx = db_rec['p_box_idx']
        c_box_idx = db_rec['c_box_idx']
        is_same = db_rec['is_same']
        start = time.time()
        p = PurePosixPath(file_name)
        pred_json_file_name = os.path.join(self.pred_anno_dir, p.stem + '.json')
        pred = load_from_cache(pred_json_file_name, self.cached_pred)
        # debug_print('load pred json', time.time() - start)
        
        # load boxes
        start = time.time()
        prev_img_file_path = pred[prev_j]['image'][0]['name']
        cur_img_file_path = pred[cur_j]['image'][0]['name']
        box_file_name = os.path.join(self.uni_box_dir, p.stem + '.pkl')
        uni_boxes = load_from_cache(box_file_name, self.cached_boxes)
        
        p_box = uni_boxes[prev_j][prev_img_file_path][p_box_idx]
        c_box = uni_boxes[cur_j][cur_img_file_path][c_box_idx]
        u = torch.tensor([[
            min(p_box[0], c_box[0]).int().type(torch.float32),
            min(p_box[1], c_box[1]).int().type(torch.float32),
            max(p_box[2], c_box[2]).int().type(torch.float32),
            max(p_box[3], c_box[3]).int().type(torch.float32),
        ]])
        assert (u.shape[0] == 1 and u.shape[1] == 4)
        # debug_print('load boxes', time.time() - start, lvl = Levels.ERROR)
        
        
        # get union feature map
        start = time.time()
        p_pkl_file_name = os.path.join(self.cfg.PATH.FPN_PKL_DIR, p.stem + '-%03d' % (prev_j) + '.pkl')
        c_pkl_file_name = os.path.join(self.cfg.PATH.FPN_PKL_DIR, p.stem + '-%03d' % (cur_j) + '.pkl')
        
        p_fmap_pkl = load_from_cache(p_pkl_file_name, self.cached_pkl)
        c_fmap_pkl = load_from_cache(c_pkl_file_name, self.cached_pkl)
        # debug_print('load fmap', time.time() - start, lvl = Levels.ERROR)
        
        start = time.time()
        p_fmap = get_box_fmap(p_fmap_pkl, u, 'align')
        c_fmap = get_box_fmap(c_fmap_pkl, u, 'align')
        # debug_print('get fmap', time.time() - start, lvl = Levels.ERROR)
        # load joints
        
        start = time.time()
        pred_joints, pred_joints_scores = get_joints_from_annorects(pred[prev_j]['annorect'])
        p_joints, p_joints_scores = pred_joints[p_box_idx, :, :2], pred_joints_scores[p_box_idx, :]
        pred_joints, pred_joints_scores = get_joints_from_annorects(pred[cur_j]['annorect'])
        c_joints, c_joints_scores = pred_joints[c_box_idx, :, :2], pred_joints_scores[c_box_idx, :]
        # gen joints hmap
        bs, C, mH, mW = p_fmap.shape
        u_box_size = torch.tensor([
            u[0, 2] - u[0, 0],  # W
            u[0, 3] - u[0, 1],  # H
        ]).numpy()
        
        # convert joint pos to be relative to union box, some may be less than 0
        def get_union_box_based_joints(u, joints):
            box_size = u_box_size.tolist()
            joints[:, 0] -= u[0, 0]
            joints[:, 1] -= u[0, 1]
            joints[:, 0].clamp_(0, box_size[0])
            joints[:, 1].clamp_(0, box_size[1])
            return joints  # no matter copy or not
        
        p_joints = get_union_box_based_joints(u, p_joints)
        c_joints = get_union_box_based_joints(u, c_joints)
        
        p_joints_hmap = self.gen_joints_hmap(u_box_size,
                                             (mH, mW),
                                             p_joints, p_joints_scores)
        c_joints_hmap = self.gen_joints_hmap(u_box_size,
                                             (mH, mW),
                                             c_joints, c_joints_scores)
        # debug_print('get joints hemap ', time.time() - start, lvl = Levels.ERROR)
        # assemble
        start = time.time()
        inputs = torch.cat([p_fmap.squeeze(), c_fmap.squeeze(), p_joints_hmap, c_joints_hmap])
        # debug_print('assemble ', time.time() - start, lvl = Levels.ERROR)
        
        return [inputs, torch.tensor([is_same]).float()]
    
    # @log_time("Getting joints heatmap...")
    def gen_joints_hmap(self, union_box_size, hmap_size, joints, joint_scores):
        '''
        All op is writen in numpy, and converted to tensor at the end
            # copy from simple-baseline

        :param union_box_size: W,H
        :param hmap_size: (h,w)
        :param joints: 15x2
        :param joint_scores: (15,)
        :return:
        '''
        # 2TODO：暂时不对，这个是对应到原图，应该对应到 union box大小
        num_joints = joints.shape[0]
        
        target = np.zeros((num_joints,
                           hmap_size[1],
                           hmap_size[0]),
                          dtype = np.float32)  # nj x w x h ???
        target_weight = joint_scores  # not visibility since all is visible
        
        feat_stride = union_box_size / hmap_size
        radius = self.cfg.TRAIN.JOINT_MAP_SIGMA * 3
        for joint_id in range(num_joints):
            # joint coord is [x,y]
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            ul = [int(mu_x - radius), int(mu_y - radius)]  # upper left
            br = [int(mu_x + radius + 1), int(mu_y + radius + 1)]  # bottom right
            
            size = 2 * radius + 1  # 13
            x = np.arange(0, size, 1, np.float32)  # (13,)
            y = x[:, np.newaxis]  # (1,13)
            x0 = y0 = size // 2  # 6
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * radius ** 2))
            
            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], hmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], hmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], hmap_size[0])
            img_y = max(0, ul[1]), min(br[1], hmap_size[1])
            
            v = target_weight[joint_id]
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]] * self.cfg.TRAIN.JOINT_MAP_SCALE \
                * v
            # cv2.imwrite('joint_%2d.jpg' % joint_id, target[joint_id])
        return to_torch(target)
    
    @log_time('Evaluating model...')
    def eval(self, pred_scores: np.ndarray):
        '''
        
        :param pred_scores: not ordered
        :return:
        '''
        
        gt = np.array([i['is_same'] for i in self.db])
        assert len(gt) == len(pred_scores)
        
        prec, rec, pr_thres = precision_recall_curve(gt, pred_scores)
        ap = average_precision_score(gt, pred_scores)
        fpr, tpr, roc_thres = roc_curve(gt, pred_scores)
        auc = roc_auc_score(gt, pred_scores)
        return {
            'prec': prec, 'rec': rec, "pr_thres": pr_thres, 'ap': ap,
            'fpr': fpr, 'tpr': tpr, 'roc_thres': roc_thres, 'auc': auc
        }
