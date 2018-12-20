#
# ──────────────────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: U S E   P R E T R A I N E D   M O D E L S   T O   T E S T
#       R E - I M P L E M E N T E D  S I M P L E   B A S E L I N E :
# ────────────────────────────────────────────────────────────────────────────

import pprint
import visdom
from pathlib import PurePosixPath
import pprint
import matplotlib.pyplot as plt

plt.switch_backend('agg')
# local packages
import nise_lib._init_paths
from nise_lib.core import *
from nise_lib.nise_config import suffix

pp = pprint.PrettyPrinter(indent = 2)
debug_print(pp.pformat(nise_cfg))

make_nise_dirs()

if nise_cfg.TEST.MODE == 'valid':
    dataset_path = nise_cfg.PATH.GT_VAL_ANNOTATION_DIR
elif nise_cfg.TEST.MODE == 'train':
    dataset_path = nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR

allDet = 'unifed_boxes/det_pkl'
baseline = 'unifed_boxes/valid_task_1_DETbox_allBox_propAll_propDET_nmsThres_0.05_0.5'
propGT = 'unifed_boxes/valid_task_-1_DETbox_allBox_propAll_propGT_nmsThres_0.05_0.5'
propDET = 'unifed_boxes/valid_task_-1_DETbox_allBox_propAll_propDET_nmsThres_0.05_0.5'

baseline_tfiou = 'unifed_boxes/valid_task_1_DETbox_allBox_tfIoU_nmsThres_0.05_0.5'
propDET_tfiou = 'unifed_boxes/valid_task_-1_DETbox_allBox_propAll_propDET_tfIoU_nmsThres_0.05_0.5'
propGT_tfiou = 'unifed_boxes/valid_task_-1_DETbox_allBox_propAll_propGT_tfIoU_nmsThres_0.05_0.5'

eval_dir = nise_cfg.PATH.UNIFIED_JSON_DIR
eval_dir = nise_cfg.PATH._UNIFIED_JSON_DIR + 'valid_task_-1_DETbox_allBox_propAll_propGT_tfIoU_nmsThres_0.15_0.30'
rec, prec, ap = voc_eval_for_pt(dataset_path, eval_dir)
torch.save([rec, prec, ap], os.path.join(*[eval_dir, 'ap.pkl']))
debug_print(eval_dir)
debug_print('AP:', ap)

# eval_dirs = os.listdir(nise_cfg.PATH._UNIFIED_JSON_DIR)
# eval_dirs = sorted(eval_dirs)
# print(eval_dirs)
# for eval_dir in eval_dirs:
#     eval_dir = os.path.join(nise_cfg.PATH._UNIFIED_JSON_DIR, eval_dir)
#     pks = os.listdir(eval_dir)
#     if len(pks) < 50:
#         continue
#     rec, prec, ap = voc_eval_for_pt(dataset_path, eval_dir)
#     torch.save([rec, prec, ap], os.path.join(*[eval_dir, 'ap.pkl']))
#     debug_print(eval_dir)
#     debug_print('AP:', ap)
#     propgt_plot = plt.plot(rec.numpy(), prec.numpy())
#     plt.savefig(os.path.join(eval_dir + '_PRcurve.png'))
