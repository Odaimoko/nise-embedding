#
# ──────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: U S E   P R E T R A I N E D   M O D E L S   T O   T E S T
#       R E - I M P L E M E N T E D  S I M P L E   B A S E L I N E :
# ────────────────────────────────────────────────────────────────────────────

import visdom
# local packages
import nise_lib._init_paths
from flownet_utils import tools
from nise_lib.nise_functions import *
from nise_lib.nise_debugging_func import *
from nise_lib.core import *
from tron_lib.core.config import cfg as tron_cfg
from simple_lib.core.config import config as simple_cfg

viz = visdom.Visdom(env = 'run-with-flownet')
make_nise_dirs()

batch_size = 8
# ─── FROM FLOWNET 2.0 ───────────────────────────────────────────────────────────
if nise_cfg.DEBUG.load_flow_model:
    parser = argparse.ArgumentParser()
    flow_init_parser_and_tools(parser, tools)
    flow_args, rest = parser.parse_known_args()
    flow_model = load_flow_model(flow_args, parser, tools)

# ─── FROM SIMPLE BASELINE ───────────────────────────────────────────────────────
if nise_cfg.DEBUG.load_joint_est_model:
    simple_args, simple_joint_est_model = load_simple_model()
    debug_print('Simple pose detector loaded.')

# ─── HUMAN DETECT ───────────────────────────────────────────────────────────────
if nise_cfg.DEBUG.load_human_det_model:
    human_detect_args = human_detect_parse_args()
    maskRCNN, human_det_dataset = load_human_detect_model(human_detect_args, tron_cfg)
 
nise_args = get_nise_arg_parser()
setattr(nise_args, 'simple_model_file', simple_args.simple_model_file)
nise_pred_task_3_debug(nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR,
                       nise_cfg.PATH.JSON_SAVE_DIR, human_det_dataset, maskRCNN,
                       simple_joint_est_model, flow_model)
# train_est_on_posetrack(simple_args,simple_cfg, simple_joint_est_model, None)
