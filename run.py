#
# ──────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: U S E   P R E T R A I N E D   M O D E L S   T O   T E S T
#       R E - I M P L E M E N T E D  S I M P L E   B A S E L I N E :
# ────────────────────────────────────────────────────────────────────────────

import visdom
from pathlib import PurePosixPath
# local packages
import nise_lib._init_paths
from flownet_utils import tools
from nise_lib.nise_functions import *
from nise_lib.nise_debugging_func import *
from nise_lib.core import *
from tron_lib.core.config import cfg as tron_cfg
from simple_lib.core.config import config as simple_cfg

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

nise_args = get_nise_arg_parser()
setattr(nise_args, 'simple_model_file', simple_args.simple_model_file)
if nise_args.nise_mode == 'valid':
    dataset_path = nise_cfg.PATH.GT_VAL_ANNOTATION_DIR
elif nise_args.nise_mode == 'train':
    dataset_path = nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR

nise_cfg.PATH.IMAGES_OUT_DIR += nise_args.nise_mode
nise_cfg.PATH.JOINTS_DIR += nise_args.nise_mode
make_nise_dirs()

if nise_args.nise_task == '1':
    
    nise_pred_task_1_debug(dataset_path,
                           os.path.join(nise_cfg.PATH.JSON_SAVE_DIR,
                                        PurePosixPath(dataset_path).name + '_pred_task_' + nise_args.nise_task +
                                        ('_gt' if nise_cfg.TEST.USE_GT_VALID_BOX else '')),
                           human_det_dataset,
                           maskRCNN,
                           simple_joint_est_model, flow_model)
elif nise_args.nise_task == '2':
    
    nise_pred_task_3_debug(dataset_path,
                           os.path.join(nise_cfg.PATH.JSON_SAVE_DIR,
                                        PurePosixPath(dataset_path).name + '_pred_task_' + nise_args.nise_task)
                           + '_propthres_' + str(nise_cfg.ALG._PROP_HUMAN_THRES),
                           human_det_dataset,
                           maskRCNN,
                           simple_joint_est_model, flow_model)
