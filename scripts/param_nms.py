import visdom
from pathlib import PurePosixPath
import torch.multiprocessing as mp
import pprint

# local packages
import _init_paths
from flownet_utils import tools
from nise_lib.nise_config import nise_cfg, nise_logger, update_nise_config, set_path_from_nise_cfg, update_nise_logger,nise_args
from nise_lib.nise_functions import *
from nise_lib.nise_debugging_func import *
from nise_lib.core import *
from tron_lib.core.config import cfg as tron_cfg
from simple_lib.core.config import config as simple_cfg

if __name__ == '__main__':
    # https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
    mp.set_start_method('spawn', force = True)
    pp = pprint.PrettyPrinter(indent = 2)
    # nise_update_config(nise_cfg, 'exp_config/12_19-18_01-batch/batch_00_01-nmsthres-0.25,0.50.yaml')
    flow_model = None
    maskRCNN = None
    simple_joint_est_model = None
    human_det_dataset = None
    # viz = visdom.Visdom(env = 'run-with-flownet')
    
    # ─── FROM SIMPLE BASELINE ───────────────────────────────────────────────────────
    if nise_cfg.DEBUG.load_simple_model:
        simple_args, simple_joint_est_model = load_simple_model()
        simple_joint_est_model = nn.DataParallel(simple_joint_est_model).cuda()
        debug_print('Simple pose detector loaded.')
    
    # ─── HUMAN DETECT ───────────────────────────────────────────────────────────────
    if nise_cfg.DEBUG.load_human_det_model:
        human_detect_args = human_detect_parse_args()
        maskRCNN, human_det_dataset = load_human_detect_model(human_detect_args, tron_cfg)
        # maskRCNN = nn.DataParallel(maskRCNN)
    
    if nise_cfg.TEST.MODE == 'valid':
        dataset_path = nise_cfg.PATH.GT_VAL_ANNOTATION_DIR
    elif nise_cfg.TEST.MODE == 'train':
        dataset_path = nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR
    
    for t1 in np.arange(.05, .5, .1):
        for t2 in np.arange(0.3, .71, .2):
            t1 = float(t1)
            t2 = float(t2)
            
            y = create_yaml_nms([0, 50], (t1, t2), original_yaml = nise_args.nise_config)
            # debug_print('New Yaml:', y)
            update_nise_config(nise_cfg, y)
            suffix, suffix_with_range = set_path_from_nise_cfg(nise_cfg)
            make_nise_dirs()
            update_nise_logger(nise_logger, nise_cfg, suffix)
            debug_print('Running posetrack 17: NMS thresholds are %.2f, %.2f.' % (t1, t2))
            # debug_print(pp.pformat(nise_cfg))
            
            nise_flow_debug(dataset_path, simple_joint_est_model, flow_model)
