#
# ──────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: U S E   P R E T R A I N E D   M O D E L S   T O   T E S T
#       R E - I M P L E M E N T E D  S I M P L E   B A S E L I N E :
# ────────────────────────────────────────────────────────────────────────────

from pathlib import PurePosixPath
import argparse
import pprint
import torch.multiprocessing as mp
import warnings

# local packages
import init_paths
from nise_lib.nise_config import nise_cfg, nise_logger
from nise_lib.nise_functions import *
from nise_lib.core import *
from tron_lib.core.config import cfg as tron_cfg
from nise_lib.dataloader.mNetDataset import mNetDataset

if __name__ == '__main__':
    
    mp.set_start_method('spawn', force = True)
    warnings.filterwarnings('ignore')
    maskRCNN = None
    # ─── HUMAN DETECT ───────────────────────────────────────────────────────────────
    human_detect_args = human_detect_parse_args()
    maskRCNN, human_det_dataset = load_human_detect_model(human_detect_args, tron_cfg)
    # maskRCNN = nn.DataParallel(maskRCNN)
    
    # make_nise_dirs()
    
    if nise_cfg.TEST.MODE == 'valid':
        dataset_path = nise_cfg.PATH.GT_VAL_ANNOTATION_DIR
    elif nise_cfg.TEST.MODE == 'train':
        dataset_path = nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR
    
    val_dataset = mNetDataset(nise_cfg, nise_cfg.PATH.GT_VAL_ANNOTATION_DIR,
                              nise_cfg.PATH.PRED_JSON_VAL_FOR_TRAINING_MNET,
                              nise_cfg.PATH.UNI_BOX_VAL_FOR_TRAINING_MNET, False, maskRCNN)
    
    train_dataset = mNetDataset(nise_cfg, nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR,
                                nise_cfg.PATH.PRED_JSON_TRAIN_FOR_TRAINING_MNET,
                                nise_cfg.PATH.UNI_BOX_TRAIN_FOR_TRAINING_MNET, True, maskRCNN)
    ala=train_dataset[0]
    print(ala)