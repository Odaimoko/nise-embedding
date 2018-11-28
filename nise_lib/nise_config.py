import argparse
import os
import os.path
import sys

import numpy as np

def get_nise_arg_parser():
    parser = argparse.ArgumentParser(description = 'PyTorch CPN Training')
    parser.add_argument('-j', '--workers', default = 4, type = int, metavar = 'N',
                        help = 'number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default = 1, type = int, metavar = 'N',
                        help = 'number of GPU to use (default: 1)')
    parser.add_argument('--epochs', default = 32, type = int, metavar = 'N',
                        help = 'number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default = 0, type = int, metavar = 'N',
                        help = 'manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default = 'checkpoint', type = str, metavar = 'PATH',
                        help = 'path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default = '', type = str, metavar = 'PATH',
                        help = 'path to latest checkpoint')
    return parser


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


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
            
            # tag loss weighted against heatmap loss, since heatmap's entries are less than 1
            self.emb_tag_weight = 4
    
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
            self.human_bbox_size = (256, 192)
    
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
            
            self.VIS_HUMAN_THRES = 0.6
    
    class _ALG:
        def __init__(self):
            self._DEQUE_CAPACITY = 3
            self._OKS_MULTIPLIER = 1e4
            # only bbox score over this are recognized as human
            self._HUMAN_THRES = .9
            # only bbox area over this are recognized as human
            self._AREA_THRES = 32 * 32
            # only bbox ratio not over this are recognized as human
            self._ASPECT_RATIO_THRES = 0.75
            # if want more joint prop boxes, set this to false
            self.FILTER_HUMAN_WHEN_DETECT = False
            # if not filtered when detected, filter when prop??
            self.JOINT_PROP_WITH_FILTERED_HUMAN = True and not self.FILTER_HUMAN_WHEN_DETECT
            self.FILTER_BBOX_WITH_SMALL_AREA = True
            self.ASSGIN_ID_TO_FILTERED_BOX = True
    
    class _PATH:
        def __init__(self):
            self.SEQ_DIR = 'data/pt17/images/bonn/000001_bonn/'
            self.JOINTS_DIR = 'images_joint/'
            self.IMAGES_OUT_DIR = 'images_out/'
            self.JSON_SAVE_DIR = 'pred_json/'
            self.POSETRACK_ROOT = 'data/pt17/'
            self.GT_TRAIN_ANNOTATION_DIR = os.path.join(self.POSETRACK_ROOT, 'train_anno_json/')
            self.GT_VAL_ANNOTATION_DIR = os.path.join(self.POSETRACK_ROOT, 'valid_anno_json/')
    
    class _TEST:
        def __init__(self):
            self.USE_GT_VALID_BOX = True
    
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
        
        self.TEST=NiseConfig._TEST()


cfg = NiseConfig()
