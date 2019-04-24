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
from nise_lib.dataloader.mNetDataset_by_single_pair import mNetDataset as pair_dataset
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
    with torch.no_grad():
        end = time.time()
        for i in range(num_samples):
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
        debug_print('IMAGE WISE', len(labels))
        perf_indicator = val_dataset.eval(labels, all_scores)
    
    return perf_indicator


def validate_model(config, val_dataset, model, output_dir):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    num_samples = len(val_dataset)
    all_scores = []
    labels = []
    sig = torch.nn.Sigmoid()
    with torch.no_grad():
        end = time.time()
        for i in range(num_samples):
            if i < 165:
                continue
            inputs, target = val_dataset[i]
            target = target.view([-1])
            if target.numel() == 0:
                #     some image dont have detections
                continue
            output = model(inputs)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            score = sig(output)
            
            all_scores.append(score.view(-1).cpu().numpy())
            labels.append(target.view(-1).cpu().numpy())
            
            if i % config.TRAIN.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i + 1, len(val_dataset), batch_time = batch_time,
                    loss = losses)
                debug_print(msg)
        all_scores = np.concatenate(all_scores)
        labels = np.concatenate(labels)
        perf_indicator = val_dataset.eval(labels, all_scores)
    
    return perf_indicator


def val_using_loader(config, val_loader, val_dataset, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    num_samples = len(val_dataset)
    all_scores = np.zeros((num_samples, 1), dtype = np.float32)
    idx = 0
    sig = torch.nn.Sigmoid()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            
            torch.cuda.empty_cache()
            
            output = model(inputs)  # torch.Size([bs, 16/17, 96, 96])
            
            num_images = inputs.size(0)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            score = sig(output)
            
            all_scores[idx:idx + num_images, :] = score.cpu().numpy()
            idx += num_images
            
            if i % config.TRAIN.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(val_loader), batch_time = batch_time,
                    loss = losses)
                debug_print(msg)
        
        perf_indicator = val_dataset.eval(all_scores)
    
    return perf_indicator


if __name__ == '__main__':
    import threading
    
    np.set_printoptions(suppress = True)
    # mp.set_start_method('spawn', force = True)
    model = MatchingNet(nise_cfg.MODEL.INPUTS_CHANNELS)
    gpus = [0, 1]
    model = torch.nn.DataParallel(model, device_ids = [0, 1, 2, 3]).cuda()
    val_dataset = mNetDataset(nise_cfg, nise_cfg.PATH.GT_VAL_ANNOTATION_DIR,
                              nise_cfg.PATH.PRED_JSON_VAL_FOR_TRAINING_MNET,
                              nise_cfg.PATH.UNI_BOX_VAL_FOR_TRAINING_MNET, False)
    
    # p1 = validate(nise_cfg, val_dataset)
    val_pair = pair_dataset(nise_cfg, nise_cfg.PATH.GT_VAL_ANNOTATION_DIR,
                            nise_cfg.PATH.PRED_JSON_VAL_FOR_TRAINING_MNET,
                            nise_cfg.PATH.UNI_BOX_VAL_FOR_TRAINING_MNET, False)
    valid_loader = torch.utils.data.DataLoader(
        val_pair,
        batch_size = nise_cfg.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle = False,
        num_workers = nise_cfg.TRAIN.WORKERS,
        pin_memory = False,
    )
    
    threads = []
    threads.append(threading.Thread(target = val_using_loader, args = (nise_cfg, valid_loader, val_pair, model)))
    threads.append(threading.Thread(target = validate, args = (nise_cfg, val_dataset)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # p2 = val_using_loader(nise_cfg, valid_loader, val_pair, model)
    # perf_indicator = validate_model(nise_cfg, val_dataset, model, None)
