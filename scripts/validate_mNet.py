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
from collections import OrderedDict
# local packages
import init_paths
from nise_lib.nise_config import nise_cfg, nise_logger
from nise_lib.nise_functions import *
from nise_lib.core import *
from nise_lib.dataset.mNetDataset_debug import mNetDataset
from nise_lib.dataset.mNetDataset_by_single_pair import mNetDataset as pair_dataset
from nise_lib.nise_models import *
from mem_util.gpu_mem_track import MemTracker
import inspect
from nise_lib.data import dataloader as my_dataloader

from nise_lib import nise_functions


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


def val_using_loader(config, val_loader, criterion, val_dataset, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = AverageMeter()
    losses = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    num_samples = len(val_dataset)
    all_scores = np.zeros((num_samples, 1), dtype = np.float32)
    idx = 0
    sig = torch.nn.Sigmoid()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, scales, joints_heatmap, target, sample_info) in enumerate(val_loader):
            data_time.update(time.time() - end)
            
            output = model(inputs, scales, sample_info, joints_heatmap,
                           sample_info.sum(2) > 0)  # (bsx2) x CHW
            target = target.cuda(non_blocking = True).view([-1, 1])
            loss = criterion(output, target)
            
            num_images = inputs.size(0)
            
            losses.update(loss.item(), inputs.size(0))
            
            score = sig(output)
            
            avg_acc = accuracy(target.detach().cpu().numpy(), score.detach().cpu().numpy(),
                               nise_cfg.TEST.POSITIVE_PAIR_THRES)
            
            acc.update(avg_acc, len(target))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            all_scores[idx:idx + num_images, :] = score.cpu().numpy()
            idx += num_images
            
            if i % config.TRAIN.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time = batch_time, data_time = data_time,
                    loss = losses, acc = acc)
                debug_print(msg)
        
        perf_indicator = val_dataset.eval(all_scores)
    
    return perf_indicator


if __name__ == '__main__':
    np.set_printoptions(suppress = True)
    # mp.set_start_method('spawn', force = True)
    
    gpus = os.environ.get('CUDA_VISIBLE_DEVICES', default = '0').split(',')
    #
    maskRCNN = None
    if nise_cfg.DEBUG.load_human_det_model:
        human_detect_args = human_detect_parse_args()
        maskRCNN, human_det_dataset = load_human_detect_model(
            human_detect_args, tron_cfg)
    
    model, meta_info = load_mNet_model(nise_cfg.PATH.mNet_MODEL_FILE, maskRCNN)
    loss_calc = torch.nn.BCEWithLogitsLoss()
    
    val_pair = pair_dataset(nise_cfg, nise_cfg.PATH.GT_VAL_ANNOTATION_DIR,
                            nise_cfg.PATH.PRED_JSON_VAL_FOR_TRAINING_MNET,
                            nise_cfg.PATH.UNI_BOX_VAL_FOR_TRAINING_MNET, False)
    
    val_loader = torch.utils.data.DataLoader(
        val_pair,
        batch_size = 4 * len(gpus),
        shuffle = False,
        num_workers = 4,
        pin_memory = False,
    )
    # for i, (inputs, scales, joints_heatmap, target, sample_info) in enumerate(val_loader):
    #     print(inputs.shape)
    p2 = val_using_loader(nise_cfg, val_loader, loss_calc, val_pair, model)
    model_p = PurePosixPath(nise_cfg.PATH.mNet_MODEL_FILE)
    result_file = os.path.join(nise_cfg.PATH.MODEL_SAVE_DIR_FOR_TRAINING_MNET, model_p.parts[-1]+'.eval')
    torch.save(p2, result_file)
    