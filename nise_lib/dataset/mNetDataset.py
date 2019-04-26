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
    def __init__(self, _nise_cfg, gt_anno_dir, pred_anno_dir, uni_box_dir, is_train, maskRCNN):
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
        dataset_path = self.cfg.PATH.PRE_COMPUTED_TRAIN_DATASET if is_train \
            else self.cfg.PATH.PRE_COMPUTED_VAL_DATASET
        if os.path.exists(dataset_path):
            debug_print("Loading cached dataset", dataset_path)
            self.db = torch.load(dataset_path)
        else:
            self.db = self._get_db()
        # self.db = self._get_db()
        
        debug_print("Loaded %d entries." % len(self), lvl = Levels.SUCCESS)
        self.cached_pkl = LimitedSizeDict(size_limit = 20)
    
    def __len__(self, ):
        return len(self.db)
    
    @log_time("Load dataset...")
    def _get_db(self):
        def add_pairs(idx1, idx2, l):
            for p_idx in idx1:
                for c_idx in idx2:
                    p_box = uni_boxes[prev_j][prev_img_file_path][p_idx]
                    c_box = uni_boxes[j][cur_img_file_path][c_idx]
                    u = unioned_box(p_box, c_box).int().float()
                    assert (u.shape[0] == 1 and u.shape[1] == 4)
                    p_box = expand_vector_to_tensor(p_box[:4])
                    c_box = expand_vector_to_tensor(c_box[:4])
                    iou = tf_iou(p_box.numpy(), c_box.numpy())
                    if iou[0, 0] > self.cfg.TRAIN.IOU_THERS_FOR_NEGATIVE:
                        entry = [p_idx, c_idx, u]
                        l.append(entry)
        
        np.set_printoptions(suppress = True)
        
        db = []
        total_num_pos = 0
        total_num_neg = 0
        for vid, file_name in enumerate(self.anno_file_names[:10]):
            if is_skip_video(nise_cfg, vid, file_name):
                # debug_print('Skip', vid, file_name)
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
                    # img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, frame['image'][0]['name'])
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
                        
                        ent = {
                            'video_file': vid,
                            'prev_frame': prev_j,
                            'cur_frame': j,  # index of images in video seq
                            'pos': pos_entries,
                            'neg': neg_entries
                        }
                        db.append(ent)
                        # debug_print('num_pos vs num_neg', num_pos, num_neg)
                        total_num_pos += num_pos
                        total_num_neg += num_neg
                    prev_j = j
                    prev_id = cur_id
        # debug_print("Total num_pos vs num_neg", total_num_pos, total_num_neg, lvl = Levels.STATUS)
        
        dataset_path = self.cfg.PATH.PRE_COMPUTED_TRAIN_DATASET if self.is_train \
            else self.cfg.PATH.PRE_COMPUTED_VAL_DATASET
        if not os.path.exists(dataset_path):
            # debug_print("Saving cached dataset...", dataset_path)
            self.db = torch.save(db, dataset_path)
            # debug_print('Done.')
        return db
    
    # @log_time('Getting item...')
    def __getitem__(self, idx):
        def load_pkl(img_idx):
            if self.conv_body:
                prev_img_file_path = os.path.join(self.cfg.PATH.POSETRACK_ROOT, pred[img_idx]['image'][0]['name'])
                fmap = gen_fmap_by_maskRCNN(prev_img_file_path)
            else:
                p_pkl_file_name = os.path.join(self.cfg.PATH.FPN_PKL_DIR, p.stem + '-%03d' % (img_idx) + '.pkl')
                fmap = torch.load(p_pkl_file_name)
            key = file_name + str(img_idx)
            if not (key in self.cached_pkl.keys()):
                self.cached_pkl[key] = fmap
            return self.cached_pkl[key]
        
        # @log_time("GEN fmap by mask")
        def gen_fmap_by_maskRCNN(img_file_path):
            
            original_img = cv2.imread(img_file_path)  # with original size
            
            inputs, im_scale = _get_blobs(original_img, None, tron_cfg.TEST.SCALE, tron_cfg.TEST.MAX_SIZE)
            
            if tron_cfg.DEDUP_BOXES > 0 and not tron_cfg.MODEL.FASTER_RCNN:
                # No use but serves to check whether the yaml file is loaded to cfg
                v = inputs['rois']
            
            with torch.no_grad():
                hai = self.conv_body(torch.from_numpy(inputs['data']).cuda())
            
            return {
                'fmap': hai[3].cpu(),
                'scale': im_scale
            }
        
        # debug_print('Get', idx)
        db_rec = copy.deepcopy(self.db[idx])
        file_name = self.anno_file_names[db_rec['video_file']]
        prev_j = db_rec['prev_frame']
        cur_j = db_rec['cur_frame']
        pos = db_rec['pos']  # list of [p_box_idx,c_box_idx,union box]
        neg = db_rec['neg']
        
        start = time.time()
        
        p = PurePosixPath(file_name)
        # get union feature map
        with open(os.path.join(self.pred_anno_dir, p.stem + '.json'), 'r') as f:
            pred = json.load(f)['annolist']
        # debug_print('load pred jsond', time.time() - start)
        p_pred_joints, p_pred_joints_scores = get_joints_from_annorects(pred[prev_j]['annorect'])
        c_pred_joints, c_pred_joints_scores = get_joints_from_annorects(pred[cur_j]['annorect'])
        
        start = time.time()
        
        p_fmap_pkl = load_pkl(prev_j)
        c_fmap_pkl = load_pkl(cur_j)
        # debug_print('load fmap files', time.time() - start)
        
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
        all_inputs = torch.zeros([num_total_samples,
                                  nise_cfg.MODEL.INPUTS_CHANNELS,
                                  nise_cfg.MODEL.FEATURE_MAP_RESOLUTION,
                                  nise_cfg.MODEL.FEATURE_MAP_RESOLUTION])
        start = time.time()
        
        all_inputs = gen_all_inputs(all_inputs, all_samples, p_fmap_pkl, c_fmap_pkl,
                                    p_pred_joints, p_pred_joints_scores, c_pred_joints, c_pred_joints_scores)
        # for i, s in enumerate(all_samples):
        #     p_box_idx, c_box_idx, union_box = s
        #     all_inputs[i] = get_one_sample(p_fmap_pkl, c_fmap_pkl, union_box,
        #                                          p_box_idx, c_box_idx,
        #                                          p_pred_joints, p_pred_joints_scores,
        #                                          c_pred_joints, c_pred_joints_scores)
        labels = -torch.ones(num_total_samples)
        labels[:num_pos_to_sample] = 1
        labels[num_pos_to_sample:num_neg_to_sample + num_pos_to_sample] = 0
        # debug_print('ASSEMBLE ASSEMBLE', time.time() - start)
        return all_inputs, labels
    
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
