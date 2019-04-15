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
import _init_paths
from nise_lib.core import *
from nise_lib.nise_config import suffix
from plogs.logutils import Levels

make_nise_dirs()

if nise_cfg.TEST.MODE == 'valid':
    dataset_path = nise_cfg.PATH.GT_VAL_ANNOTATION_DIR
elif nise_cfg.TEST.MODE == 'train':
    dataset_path = nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR

allDet = 'unifed_boxes-pre-commissioning/det_pkl'
baseline = 'unifed_boxes-pre-commissioning/valid_task_1_DETbox_allBox_propAll_propDET_nmsThres_0.05_0.5'
propGT = 'unifed_boxes-pre-commissioning/valid_task_-1_DETbox_allBox_propAll_propGT_nmsThres_0.05_0.5'
propDET = 'unifed_boxes-pre-commissioning/valid_task_-1_DETbox_allBox_propAll_propDET_nmsThres_0.05_0.5'

baseline_tfiou = 'unifed_boxes/valid_task_1_DETbox_allBox_tfIoU_nmsThres_0.05_0.5'
propDET_tfiou = 'unifed_boxes/valid_task_-1_DETbox_allBox_propAll_propDET_tfIoU_nmsThres_0.05_0.5'
propGT_tfiou = 'unifed_boxes-pre-commissioning/valid_task_-1_DETbox_allBox_propAll_propGT_tfIoU_nmsThres_0.05_0.5'

uni_for_t3='unifed_boxes-commi-onlydet/valid_task_1_DETbox_allBox_tfIoU_nmsThres_0.35_0.50'

matched_det_box ='unified_boxes-matchedDet/valid_task_-3_noid'

info_level = Levels.SKY_BLUE

eval_dir = uni_for_t3
# eval_dir = nise_cfg.PATH._UNIFIED_JSON_DIR + 'valid_task_-1_DETbox_allBox_propAll_propGT_tfIoU_nmsThres_0.05_0.90'
rec, prec, ap, total_pred_boxes_scores, npos = voc_eval_for_pt(dataset_path, eval_dir)
torch.save([rec, prec, ap], os.path.join(*[eval_dir, 'ap.pkl']))
debug_print("Pkl saved. Evaluate box AP", eval_dir, lvl = info_level)
debug_print('In total %d predictions, %d gts.' % (total_pred_boxes_scores.shape[0], npos), lvl = info_level)
debug_print('AP: %.4f' % ap.item(), lvl = info_level)



# root=nise_cfg.PATH._UNIFIED_JSON_DIR
# root='unifed_boxes-commi-onlydet'
# eval_dirs = os.listdir(root)
# eval_dirs = [f for f in eval_dirs if not 'png' in f]
# eval_dirs = sorted(eval_dirs)
# print(eval_dirs)
# APs = []
# num_preds = []
# npos = 0
# for eval_dir in eval_dirs:
#     eval_dir = os.path.join(root, eval_dir)
#     pks = os.listdir(eval_dir)
#     if len(pks) < 50:
#         debug_print("skip", eval_dir)
#         continue
#     rec, prec, ap, total_pred_boxes_scores, npos = voc_eval_for_pt(dataset_path, eval_dir)
#     torch.save([rec, prec, ap], os.path.join(*[eval_dir, 'ap.pkl']))
#     num_preds.append(total_pred_boxes_scores.shape[0])
#     debug_print("Pkl saved. Evaluate box AP", eval_dir, lvl = info_level)
#     debug_print('In total %d predictions, %d gts.' % (total_pred_boxes_scores.shape[0], npos), lvl = info_level)
#     debug_print('AP: %.4f' % ap.item(), lvl = info_level)
#     APs.append(ap)
#     propgt_plot = plt.plot(rec.numpy(), prec.numpy())
#     plt.savefig(os.path.join(eval_dir + '_PRcurve.png'))
#     debug_print("PR curve saved.")
#
# for dir, ap, num_pred in zip(eval_dirs, APs, num_preds):
#     debug_print(dir, lvl = info_level)
#     debug_print('In total %d predictions, %d gts.\tAP: %.4f' % (num_pred, npos, ap.item()),
#                 lvl = info_level)
