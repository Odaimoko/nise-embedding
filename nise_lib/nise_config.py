import argparse
import copy
import os
import os.path
import time
from pathlib import Path
from munkres import Munkres

import numpy as np
import yaml
from easydict import EasyDict as edict

from plogs.plogs import get_logger


def get_nise_arg_parser():
    parser = argparse.ArgumentParser(description = 'NISE PT')
    # parser.add_argument('-j', '--workers', default = 4, type = int, metavar = 'N',
    #                     help = 'number of data loading workers (default: 12)')
    # parser.add_argument('-g', '--num_gpus', default = 1, type = int, metavar = 'N',
    #                     help = 'number of GPU to use (default: 1)')
    
    parser.add_argument('--nise_config', type = str, metavar = 'nise config file',
                        help = 'path to yaml format config file')
    parser.add_argument('--simple-model-file', type = str, )
    args, rest = parser.parse_known_args()
    return args


def update_config(_config, config_file):
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
    suffix = '_'.join([
        nise_cfg.TEST.MODE,
        'task',
        str(nise_cfg.TEST.TASK),
        'gt' if nise_cfg.TEST.USE_GT_VALID_BOX else 'detect',
        'propthres',
        str(nise_cfg.ALG.PROP_HUMAN_THRES),
        'propfiltered' if nise_cfg.ALG.JOINT_PROP_WITH_FILTERED_HUMAN else 'propall',
    ])
    suffix_range = '_'.join(['RANGE',
                             str(nise_cfg.TEST.FROM),
                             str(nise_cfg.TEST.TO)] if not nise_cfg.TEST.ONLY_TEST else nise_cfg.TEST.ONLY_TEST)
    suffix_with_range = '_'.join([suffix, suffix_range])
    nise_cfg.PATH.JSON_SAVE_DIR = os.path.join(nise_cfg.PATH.JSON_SAVE_DIR, suffix)
    nise_cfg.PATH.JOINTS_DIR = os.path.join(nise_cfg.PATH.JOINTS_DIR, suffix_with_range)
    nise_cfg.PATH.IMAGES_OUT_DIR = os.path.join(nise_cfg.PATH.IMAGES_OUT_DIR, suffix_with_range)
    return suffix_with_range


def create_nise_logger(nise_cfg, cfg_name, phase = 'train'):
    root_output_dir = Path(nise_cfg.PATH.LOG_SAVE_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    
    model = '_'.join([
        nise_cfg.TEST.MODE,
        'task',
        str(nise_cfg.TEST.TASK),
        'gt' if nise_cfg.TEST.USE_GT_VALID_BOX else '',
        'propthres',
        str(nise_cfg.ALG.PROP_HUMAN_THRES),
        '_'.join(['RANGE',
                  str(nise_cfg.TEST.FROM),
                  str(nise_cfg.TEST.TO)] if not nise_cfg.TEST.ONLY_TEST else nise_cfg.TEST.ONLY_TEST),
    ])
    cfg_name = os.path.basename(cfg_name)
    
    final_output_dir = root_output_dir / cfg_name
    
    print('creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents = True, exist_ok = True)
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    ploger = get_logger()
    ploger.config(to_file = True,
                  file_location = str(final_output_dir) + '/',
                  filename = log_file, show_levels = True,
                  show_time = True)
    ploger.format('[{level}] - {msg}')
    
    return ploger, str(final_output_dir)


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
            self.load_flow_model = True
            self.FLOW = False
            self.load_human_det_model = True
            self.HUMAN = False
            self.load_joint_est_model = True
            self.SIMPLE = False
            
            self.FRAME = True
            self.EST_IN_FRAME = False
            self.VIS_HUMAN_THRES = .5
            
            self.VISUALIZE = False
    
    class _ALG:
        
        def __init__(self):
            self._DEQUE_CAPACITY = 3
            self._OKS_MULTIPLIER = 1e4
            # only bbox score over this are recognized as human
            self._HUMAN_THRES = .5
            self.PROP_HUMAN_THRES = .5
            # only bbox area over this are recognized as human
            self._AREA_THRES = 32 * 32
            # only bbox ratio not over this are recognized as human
            self._ASPECT_RATIO_THRES = 0.75
            # if want more joint prop boxes, set this to false
            self.FILTER_HUMAN_WHEN_DETECT = False
            # if not filtered when detected, filter when prop??
            self.JOINT_PROP_WITH_FILTERED_HUMAN = True
            self.FILTER_BBOX_WITH_SMALL_AREA = False
            self.ASSGIN_ID_TO_FILTERED_BOX = self.JOINT_PROP_WITH_FILTERED_HUMAN
            self.USE_ALL_PROPED_BOX_TO_ASSIGN_ID = True
            # padding image s.t. w/h to be multiple of 32
            self.FLOW_MULTIPLE = 2 ** 6
            self.FLOW_PADDING_END = 0
            self.FLOW_MODE = self.FLOW_PADDING_END
            self.SIMILARITY_TYPE = 0
    
    class _PATH:
        def __init__(self):
            self.SEQ_DIR = 'data/pt17/images/bonn/000001_bonn/'
            self.JOINTS_DIR = 'images_joint/'
            self.IMAGES_OUT_DIR = 'images_out/'
            self.JSON_SAVE_DIR = 'pred_json/'
            self.LOG_SAVE_DIR = 'logs/'
            self.POSETRACK_ROOT = 'data/pt17/'
            self.GT_TRAIN_ANNOTATION_DIR = os.path.join(self.POSETRACK_ROOT, 'train_anno_json/')
            self.GT_VAL_ANNOTATION_DIR = os.path.join(self.POSETRACK_ROOT, 'valid_anno_json/')
    
    class _TEST:
        def __init__(self):
            self.USE_GT_VALID_BOX = False
            self.USE_GT_JOINTS_TO_PROP = True
    
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
update_config(nise_cfg, nise_args.nise_config)

suffix = set_path_from_nise_cfg(nise_cfg)
nise_logger, _ = create_nise_logger(nise_cfg, suffix, 'valid')

mkrs = Munkres()

