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
    # make_nise_dirs()
    
    if nise_cfg.TEST.MODE == 'valid':
        dataset_path = nise_cfg.PATH.GT_VAL_ANNOTATION_DIR
    elif nise_cfg.TEST.MODE == 'train':
        dataset_path = nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR
    
    val_dataset = mNetDataset(nise_cfg, nise_cfg.PATH.GT_VAL_ANNOTATION_DIR,
                              nise_cfg.PATH.PRED_JSON_VAL_FOR_TRAINING_MNET,
                              nise_cfg.PATH.UNI_BOX_VAL_FOR_TRAINING_MNET, False)
    
    train_dataset = mNetDataset(nise_cfg, nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR,
                                nise_cfg.PATH.PRED_JSON_TRAIN_FOR_TRAINING_MNET,
                                nise_cfg.PATH.UNI_BOX_TRAIN_FOR_TRAINING_MNET, True)
    # ala = val_dataset[0]
    # dla = val_dataset[1]
    # bla = train_dataset[0]
    # cla=val_dataset[2]
    
    for t in range(10):
        t = val_dataset[t]
        print(t[1])
    
    # print(ala)
