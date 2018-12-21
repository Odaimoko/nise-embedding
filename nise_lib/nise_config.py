import argparse
import copy
import os
import os.path
import time
from pathlib import Path

import numpy as np
import yaml
from easydict import EasyDict as edict
from munkres import Munkres

from plogs.plogs import get_logger


def get_nise_arg_parser():
    parser = argparse.ArgumentParser(description = 'NISE PT')
    parser.add_argument('--nise_config', type = str, metavar = 'nise config file',
                        help = 'path to yaml format config file', default = 'exp_config/t.yaml')
    parser.add_argument('--simple-model-file', type = str, )
    args, rest = parser.parse_known_args()
    return args


def update_nise_config(_config, config_file):
    if config_file is None: return
    
    def update_dict(_config, k, v):
        
        if isinstance(v, dict):
            for k1, v1 in v.items():
                update_dict(_config[k], k1, v1)
        else:
            _config[k] = v
    
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            update_dict(_config, k, v)
            # if k in _config:
            #     else:
            # else:
            #     raise ValueError("{} not exist in _config.py".format(k))


def get_edcfg_from_nisecfg(nise_cfg):
    '''

    :param nise_cfg: 2-level class
    :return:
    '''
    new_cfg = copy.copy(nise_cfg)
    new_cfg = new_cfg.__dict__
    for k, v in new_cfg.items():
        new_cfg[k] = edict(v.__dict__)
    new_cfg = edict(new_cfg)
    return new_cfg


def set_path_from_nise_cfg(nise_cfg):
    detect_part = [
        'GTbox' if nise_cfg.TEST.USE_GT_PEOPLE_BOX else 'DETbox',
        'filterBox' if nise_cfg.ALG.FILTER_HUMAN_WHEN_DETECT else 'allBox'
    ]
    if nise_cfg.ALG.FILTER_HUMAN_WHEN_DETECT:
        detect_part.append('boxThres')
        detect_part.append(str(nise_cfg.ALG._HUMAN_THRES))
    
    prop_part = []
    
    if nise_cfg.TEST.USE_ALL_GT_JOINTS_TO_PROP:
        prop_part.append('propALLGT')
        prop_part.append('gtScore')
        prop_part.append(str(nise_cfg.TEST.GT_BOX_SCORE))
    # else:
    if nise_cfg.ALG.JOINT_PROP_WITH_FILTERED_HUMAN:
        prop_part.append('propFiltered')
        prop_part.append('propThres')
        prop_part.append(str(nise_cfg.ALG.PROP_HUMAN_THRES))
    else:
        prop_part.append('propAll')
    if nise_cfg.TEST.USE_GT_JOINTS_TO_PROP:
        prop_part.append('propGT')
    else:
        prop_part.append('propDET')
    
    if not nise_cfg.DEBUG.NO_NMS:
        if nise_cfg.ALG.USE_COCO_IOU_IN_NMS:
            unify_part = ['cocoIoU']
        else:
            unify_part = ['tfIoU']
        unify_part.extend([
            'nmsThres',
            '%.2f' % (nise_cfg.ALG.UNIFY_NMS_THRES_1),
            '%.2f' % (nise_cfg.ALG.UNIFY_NMS_THRES_2)
        ])
    
    else:
        unify_part = ['noNMS']
    
    detect_part = '_'.join(detect_part)
    prop_part = '_'.join(prop_part) if nise_cfg.TEST.TASK == 2 or nise_cfg.TEST.TASK == -1 else ''
    unify_part = '_'.join(unify_part)
    
    suffix_list = [
        nise_cfg.TEST.MODE,
        'task',
        str(nise_cfg.TEST.TASK), ]
    if detect_part: suffix_list.append(detect_part)
    if prop_part: suffix_list.append(prop_part)
    if unify_part: suffix_list.append(unify_part)
    suffix = '_'.join(suffix_list)
    suffix_range = '_'.join(['RANGE',
                             str(nise_cfg.TEST.FROM),
                             str(nise_cfg.TEST.TO)] if not nise_cfg.TEST.ONLY_TEST else nise_cfg.TEST.ONLY_TEST)
    suffix_with_range = '_'.join([suffix, suffix_range])
    nise_cfg.PATH.JSON_SAVE_DIR = os.path.join(nise_cfg.PATH._JSON_SAVE_DIR, suffix)
    nise_cfg.PATH.UNIFIED_JSON_DIR = os.path.join(nise_cfg.PATH._UNIFIED_JSON_DIR, suffix)
    nise_cfg.PATH.JOINTS_DIR = os.path.join(nise_cfg.PATH._JOINTS_DIR, suffix_with_range)
    nise_cfg.PATH.IMAGES_OUT_DIR = os.path.join(nise_cfg.PATH._IMAGES_OUT_DIR, suffix_with_range)
    return suffix, suffix_with_range


def create_nise_logger(nise_cfg, cfg_name, phase = 'train'):
    ploger = get_logger()
    update_nise_logger(ploger, nise_cfg, cfg_name)
    
    return ploger


def update_nise_logger(ploger, nise_cfg, cfg_name, phase = 'train'):
    root_output_dir = Path(nise_cfg.PATH.LOG_SAVE_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    
    cfg_name = os.path.basename(cfg_name)
    time_str = time.strftime("%m_%d-%H_%M", time.localtime())
    
    log_file = '{}_{}.log'.format(time_str, cfg_name)
    ploger.config(to_file = True,
                  file_location = nise_cfg.PATH.LOG_SAVE_DIR,
                  filename = log_file, show_levels = True,
                  show_time = True)
    ploger.format('[{level}] - {msg}')


class NiseConfig:
    class _TRAIN:
        def __init__(self):
            self.START_EPOCH = 0
            self.END_EPOCH = 40
            self.batch_size = 32
            self.lr = 2.5e-4
            self.weight_decay = .05
            # lr_gamma = 0.05
            # lr_dec_epoch = list(range(6, 40, 6))
    
    class _DATA:
        def __init__(self):
            self.num_joints = 15
            self.image_path = 'data/mpii-video-pose'
            self.data_shape = (512, 512)
            self.output_shape = (128, 128)
            
            self.gt_hmap_sigmas = [3, 7, 11]
            
            self.symmetry = [(1, 2), (3, 4), (5, 6), (7, 8),
                             (9, 10), (11, 12), (13, 14), (15, 16)]
            self.bbox_extend_factor = (0.075, 0.075)  # x, y
            
            # ─── DATA AUGMENTATION SETTING ──────────────────────────────────────────────────
            
            self.scale_factor = (0.7, 1.35)
            self.rot_factor = 45
            
            self.pixel_means = np.array([122.7717, 115.9465, 102.9801])  # RGB
            
            #  should ask author
            self.flow_input_size = (1024, 576)
            # should initialize from simple_cfg
    
    class _DEBUG:
        def __init__(self):
            self.PRINT = True
            self.DEVELOPING = True
            self.load_flow_model = False
            self.FLOW = False
            self.load_human_det_model = True
            self.HUMAN = False
            self.load_joint_est_model = True
            self.SIMPLE = False
            
            self.FRAME = True
            self.VISUALIZE = True
            self.VIS_HUMAN_THRES = .5
            self.VIS_SINGLE_NO_JOINTS = False
            self.VIS_BOX = False
            self.VIS_EST_SINGLE = False
            self.VIS_PROPED_JOINTS = False
            self.VIS_JOINTS_FULL = True
            
            self.SAVE_DETECTION_TENSOR = False
            self.USE_DETECTION_RESULT = True
    
    class _ALG:
        
        def __init__(self):
            self._DEQUE_CAPACITY = 3
            self._OKS_MULTIPLIER = 1e4
            # only bbox score over this are recognized as human
            self._HUMAN_THRES = .5
            self.PROP_HUMAN_THRES = .5
            # only bbox area over this are recognized as human
            self._AREA_THRES = 32
            # only bbox ratio not over this are recognized as human
            self._ASPECT_RATIO_THRES = 0.75
            # if want more joint prop boxes, set this to false
            self.FILTER_HUMAN_WHEN_DETECT = True
            # if not filtered when detected, filter when prop??
            self.JOINT_PROP_WITH_FILTERED_HUMAN = True
            self.FILTER_BBOX_WITH_SMALL_AREA = False
            self.UNIFY_NMS_THRES_1 = .3
            self.UNIFY_NMS_THRES_2 = .5
            
            self.ASSGIN_ID_TO_FILTERED_BOX = False  # self.JOINT_PROP_WITH_FILTERED_HUMAN
            self.USE_ALL_PROPED_BOX_TO_ASSIGN_ID = True
            # padding image s.t. w/h to be multiple of 32
            self.FLOW_MULTIPLE = 2 ** 6
            self.FLOW_PADDING_END = 0
            self.FLOW_MODE = self.FLOW_PADDING_END
            self.SIMILARITY_TYPE = 0
    
    class _PATH:
        def __init__(self):
            self.SEQ_DIR = 'data/pt17/images/bonn/000001_bonn/'
            self.LOG_SAVE_DIR = 'logs/'
            self.POSETRACK_ROOT = 'data/pt17/'
            self.GT_TRAIN_ANNOTATION_DIR = os.path.join(self.POSETRACK_ROOT, 'train_anno_json/')
            self.GT_VAL_ANNOTATION_DIR = os.path.join(self.POSETRACK_ROOT, 'valid_anno_json/')
            
            self._JOINTS_DIR = 'images_joint/'
            self._IMAGES_OUT_DIR = 'images_out/'
            self._JSON_SAVE_DIR = 'pred_json-debug/'
            self._UNIFIED_JSON_DIR = 'unifed_boxes-debug/'
            
            self.JOINTS_DIR = ''
            self.IMAGES_OUT_DIR = ''
            self.JSON_SAVE_DIR = ''
            self.UNIFIED_JSON_DIR = ''
            
            self.DETECT_JSON_DIR = 'pre_com/det_json/'
            self.FLOW_JSON_DIR = 'pre_com/flow/'
            self.DET_EST_JSON_DIR = 'pre_com/det_est/'
    
    class _TEST:
        def __init__(self):
            self.USE_GT_PEOPLE_BOX = False
            
            self.USE_GT_JOINTS_TO_PROP = True
            self.USE_ALL_GT_JOINTS_TO_PROP = True
            self.GT_BOX_SCORE = 1
            
            self.GT_JOINTS_PROP_IOU_THRES = .5
            
            self.ONLY_TEST = []
            
            self.MAP_TP_IOU_THRES = .5
    
    def __init__(self):
        #
        # ─── TRAINING ───────────────────────────────────────────────────────────────────
        #
        self.TRAIN = NiseConfig._TRAIN()
        
        #
        # ─── DATA ───────────────────────────────────────────────────────────────────────
        #
        self.DATA = NiseConfig._DATA()
        
        # ─── DEBUGGING ──────────────────────────────────────────────────────────────────
        self.DEBUG = NiseConfig._DEBUG()
        
        # ─── ALGORITHM ──────────────────────────────────────────────────────────────────
        
        self.ALG = NiseConfig._ALG()
        
        self.PATH = NiseConfig._PATH()
        
        self.TEST = NiseConfig._TEST()


cfg = NiseConfig()
nise_cfg = get_edcfg_from_nisecfg(cfg)
nise_args = get_nise_arg_parser()
update_nise_config(nise_cfg, nise_args.nise_config)

suffix, suffix_with_range = set_path_from_nise_cfg(nise_cfg)
print('SUFFIX', suffix_with_range)
nise_logger = get_logger()
update_nise_logger(nise_logger, nise_cfg, suffix)
# print('original cfg and logger', id(nise_cfg), id(nise_logger))
mkrs = Munkres()
nise_cfg_pack = {
    'cfg': nise_cfg,
    'logger': nise_logger
}
