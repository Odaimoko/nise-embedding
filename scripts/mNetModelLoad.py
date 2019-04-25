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
from collections import OrderedDict
from torch import optim
# local packages
import init_paths
from nise_lib.nise_config import nise_cfg, nise_logger
from nise_lib.nise_functions import *
from nise_lib.nise_debugging_func import *
from nise_lib.nise_models import MatchingNet

from nise_lib.core import *
from tron_lib.core.config import cfg as tron_cfg
from simple_lib.core.config import config as simple_cfg

if __name__ == '__main__':
    model = MatchingNet(nise_cfg.MODEL.INPUTS_CHANNELS).cuda()
    gpus = [int(i) for i in os.environ.get('CUDA_VISIBLE_DEVICES', default = '0,1,2,3').split(',')]
    optimizer = optim.Adam(
        model.parameters(),lr = nise_cfg.TRAIN.LR
    )
    meta_info = torch.load('mnet_output/ep-6-0.pkl')
    state_dict = OrderedDict({k.replace('module.', ''): v for k, v in meta_info['state_dict'].items()})
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(meta_info['optimizer'])
    model = torch.nn.DataParallel(model, device_ids = gpus)
    print()