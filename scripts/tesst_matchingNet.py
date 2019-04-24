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
from mem_util.gpu_mem_track import MemTracker
import inspect


class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def validate(config, val_dataset):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    # switch to evaluate mode
    num_samples = len(val_dataset)
    all_scores = []
    labels = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i in range(num_samples):
            debug_print(i)
            # if i < 210:
            #     continue
            inputs, target = val_dataset[i]
            num_images = inputs.size(0)
            
            target = target.cuda(non_blocking = True)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            all_scores.append(np.random.rand(num_images))
            labels.append(target.view(-1).cpu().numpy())
            
            if i % config.TRAIN.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    .format(i, len(val_dataset), batch_time = batch_time)
                debug_print(msg)
        all_scores = np.concatenate(all_scores)
        
        labels = np.concatenate(labels)
        
        perf_indicator = val_dataset.eval(labels, all_scores)
    
    return perf_indicator


if __name__ == '__main__':
    np.set_printoptions(suppress = True)
    val_dataset = mNetDataset(nise_cfg, nise_cfg.PATH.GT_VAL_ANNOTATION_DIR,
                              nise_cfg.PATH.PRED_JSON_VAL_FOR_TRAINING_MNET,
                              nise_cfg.PATH.UNI_BOX_VAL_FOR_TRAINING_MNET, False)
    gpt = MemTracker(inspect.currentframe()).track
    # train_dataset = mNetDataset(nise_cfg, nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR,
    #                             nise_cfg.PATH.PRED_JSON_TRAIN_FOR_TRAINING_MNET,
    #                             nise_cfg.PATH.UNI_BOX_TRAIN_FOR_TRAINING_MNET, True)
    validate(nise_cfg, val_dataset)
    # debug_print(pprint.pformat(val_dataset.eval(fake_gt, np.random.rand(len(val_dataset)))))
