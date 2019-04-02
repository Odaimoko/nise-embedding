import visdom
from pathlib import PurePosixPath
import torch.multiprocessing as mp
import pprint

# local packages
import _init_paths
from flownet_utils import tools
from nise_lib.nise_config import nise_cfg, nise_logger, update_nise_config, set_path_from_nise_cfg, update_nise_logger, \
    nise_args
from nise_lib.nise_functions import *
from nise_lib.nise_debugging_func import *
from nise_lib.core import *
from tron_lib.core.config import cfg as tron_cfg
from simple_lib.core.config import config as simple_cfg

if __name__ == '__main__':
    # https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
    mp.set_start_method('spawn', force = True)
    pp = pprint.PrettyPrinter(indent = 2)
    
    if nise_cfg.TEST.MODE == 'valid':
        dataset_path = nise_cfg.PATH.GT_VAL_ANNOTATION_DIR
    elif nise_cfg.TEST.MODE == 'train':
        dataset_path = nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR
    threads=[]
    for box_thres in np.arange(.3, .91, .1):
        for joint_thres in np.arange(.1, .71, .1):
            box_thres = float(box_thres)
            joint_thres = float(joint_thres)
            
            y = create_yaml_track_filter([0, 50], (box_thres, joint_thres), original_yaml = nise_args.nise_config)
            # debug_print('New Yaml:', y)
            update_nise_config(nise_cfg, y)
            suffix, suffix_with_range = set_path_from_nise_cfg(nise_cfg)
            # make_nise_dirs()
            update_nise_logger(nise_logger, nise_cfg, suffix)
            debug_print('Running posetrack 17: NMS thresholds are %.2f, %.2f.' % (box_thres, joint_thres))
            # t=threading.Thread(target = nise_flow_debug, args = (dataset_path,None,None))
            nise_flow_debug(dataset_path, None, None,None)
            # threads.append(t)
            # t.start()
    for t in threads:
        t.join()