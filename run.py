#
# ──────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: U S E   P R E T R A I N E D   M O D E L S   T O   T E S T
#       R E - I M P L E M E N T E D  S I M P L E   B A S E L I N E :
# ────────────────────────────────────────────────────────────────────────────

import visdom
from pathlib import PurePosixPath
import pprint
import torch.multiprocessing as mp

# local packages
import nise_lib._init_paths
from flownet_utils import tools
from nise_lib.nise_config import nise_cfg, nise_logger
from nise_lib.nise_functions import *
from nise_lib.nise_debugging_func import *
from nise_lib.core import *
from tron_lib.core.config import cfg as tron_cfg
from simple_lib.core.config import config as simple_cfg

if __name__ == '__main__':
    
    mp.set_start_method('spawn', force = True)
    
    flow_model = None
    maskRCNN = None
    simple_joint_est_model = None
    human_det_dataset = None
    # viz = visdom.Visdom(env = 'run-with-flownet')
    # ─── FROM FLOWNET 2.0 ───────────────────────────────────────────────────────────
    if nise_cfg.DEBUG.load_flow_model:
        parser = argparse.ArgumentParser()
        flow_init_parser_and_tools(parser, tools)
        flow_args, rest = parser.parse_known_args()
        flow_model = load_flow_model(flow_args, parser, tools)
        # flow_model = nn.DataParallel(flow_model)
    
    # ─── FROM SIMPLE BASELINE ───────────────────────────────────────────────────────
    if nise_cfg.DEBUG.load_joint_est_model:
        simple_args, simple_joint_est_model = load_simple_model()
        # simple_joint_est_model = nn.DataParallel(simple_joint_est_model)
        debug_print('Simple pose detector loaded.')
    
    # ─── HUMAN DETECT ───────────────────────────────────────────────────────────────
    if nise_cfg.DEBUG.load_human_det_model:
        human_detect_args = human_detect_parse_args()
        maskRCNN, human_det_dataset = load_human_detect_model(human_detect_args, tron_cfg)
        # maskRCNN = nn.DataParallel(maskRCNN)
    
    make_nise_dirs()
    
    if nise_cfg.TEST.MODE == 'valid':
        dataset_path = nise_cfg.PATH.GT_VAL_ANNOTATION_DIR
    elif nise_cfg.TEST.MODE == 'train':
        dataset_path = nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR
    
    if nise_cfg.TEST.TASK == 1:
        
        nise_pred_task_1_debug(dataset_path,
                               human_det_dataset,
                               maskRCNN,
                               simple_joint_est_model, flow_model)
    elif nise_cfg.TEST.TASK == 2:
        
        nise_pred_task_2_debug(dataset_path,
        
                               human_det_dataset,
                               maskRCNN,
                               simple_joint_est_model, flow_model)
    elif nise_cfg.TEST.TASK == -1 or nise_cfg.TEST.TASK ==  -2:
        
        nise_flow_debug(dataset_path, simple_joint_est_model, flow_model)
