import visdom
from pathlib import PurePosixPath
import torch.multiprocessing as mp
import pprint

# local packages
import init_paths
from flownet_utils import tools
from nise_lib.nise_config import nise_cfg, nise_logger, update_nise_config, set_path_from_nise_cfg, update_nise_logger, \
    nise_args
from nise_lib.nise_functions import *
from nise_lib.nise_debugging_func import *
from nise_lib.core import *
from tron_lib.core.config import cfg as tron_cfg
from easydict import EasyDict as edict
from simple_lib.core.config import config as simple_cfg
from copy import deepcopy

if __name__ == '__main__':
    # https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
    mp.set_start_method('spawn', force = True)
    pp = pprint.PrettyPrinter(indent = 2)
    threads = []
    if nise_cfg.TEST.MODE == 'valid':
        dataset_path = nise_cfg.PATH.GT_VAL_ANNOTATION_DIR
    elif nise_cfg.TEST.MODE == 'train':
        dataset_path = nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR
    for box_thres in np.arange(0, .91, .1):
        box_thres = float(box_thres)
        
        y = create_yaml_matchedDet__pckh_filter([0, 50], box_thres, original_yaml = nise_args.nise_config)
        debug_print('New Yaml:', y)
        new_args = deepcopy(nise_args)
        setattr(new_args, 'nise_config', y)
        update_nise_config(nise_cfg, new_args)
        suffix, suffix_with_range = set_path_from_nise_cfg(nise_cfg)
        print(suffix)
        # make_nise_dirs()
        
        # update_nise_logger(nise_logger, nise_cfg, suffix)
        # debug_print('Running posetrack 17: NMS thresholds are {:.2f}.' .format (box_thres))
        # nise_flow_debug(dataset_path, None, None, None)
        gen_matched_box_debug(dataset_path)
        # t=threading.Thread(target = gen_matched_box_debug, args = (dataset_path,))
        # threads.append(t)
        # t.start()

    # for t in threads:
    #     t.join()