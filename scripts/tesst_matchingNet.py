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
from torch import optim
# local packages
import init_paths
from nise_lib.nise_config import nise_cfg, nise_logger
from nise_lib.nise_functions import *
from nise_lib.core import *
from nise_lib.dataloader.mNetDataset import mNetDataset
from nise_lib.nise_models import MatchingNet

if __name__ == '__main__':
    np.set_printoptions(suppress = True)
    val_dataset = mNetDataset(nise_cfg, nise_cfg.PATH.GT_VAL_ANNOTATION_DIR,
                              nise_cfg.PATH.PRED_JSON_VAL_FOR_TRAINING_MNET,
                              nise_cfg.PATH.UNI_BOX_VAL_FOR_TRAINING_MNET, False)
    
    debug_print(pprint.pformat(val_dataset.eval(np.random.rand(len(val_dataset)))))
    for i in val_dataset:
        print(i[0].shape, i[1].shape)
