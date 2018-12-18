#
# ──────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: U S E   P R E T R A I N E D   M O D E L S   T O   T E S T
#       R E - I M P L E M E N T E D  S I M P L E   B A S E L I N E :
# ────────────────────────────────────────────────────────────────────────────

import visdom
from pathlib import PurePosixPath
import pprint

# local packages
import nise_lib._init_paths
from flownet_utils import tools
from nise_lib.nise_config import nise_cfg, nise_logger
from nise_lib.nise_functions import *
from nise_lib.nise_debugging_func import *
from nise_lib.core import *
from tron_lib.core.config import cfg as tron_cfg
from simple_lib.core.config import config as simple_cfg

pp = pprint.PrettyPrinter(indent = 2)
debug_print(pp.pformat(nise_cfg))

make_nise_dirs()

if nise_cfg.TEST.MODE == 'valid':
    dataset_path = nise_cfg.PATH.GT_VAL_ANNOTATION_DIR
elif nise_cfg.TEST.MODE == 'train':
    dataset_path = nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR

baseline = 'unifed_boxes/valid_task_1_DETbox_allBox_propAll_propDET_nmsThres_0.05_0.5'
propGT = 'unifed_boxes/valid_task_-1_DETbox_allBox_propAll_propGT_nmsThres_0.05_0.5'
rec, prec, ap = voc_eval_for_pt(dataset_path, propGT)
debug_print('AP:', ap)
