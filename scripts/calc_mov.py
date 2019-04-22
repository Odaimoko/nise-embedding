#
# ──────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: U S E   P R E T R A I N E D   M O D E L S   T O   T E S T
#       R E - I M P L E M E N T E D  S I M P L E   B A S E L I N E :
# ────────────────────────────────────────────────────────────────────────────

from pathlib import PurePosixPath
import argparse
import pprint

# local packages
import init_paths
from nise_lib.core import *
from nise_lib.nise_config import nise_cfg, nise_logger
from nise_lib.nise_functions import *

if __name__ == '__main__':

    calc_movement_videos(nise_cfg.PATH.GT_VAL_ANNOTATION_DIR)

    # calc_movement_videos(nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR)
