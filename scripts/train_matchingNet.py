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

gpuTracker = MemTracker(inspect.currentframe())
t = gpuTracker.track


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


@log_time('Training For One Epoch...')
def train_1_ep(config, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        t()
        torch.cuda.empty_cache()
        
        # input torch.Size([bs, 542, 96, 96])
        data_time.update(time.time() - end)
        
        output = model(inputs)
        t()
        target = target.cuda(non_blocking = True)
        
        # target_weight?
        loss = criterion(output, target)
        
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        t()
        optimizer.step()
        t()
        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % config.TRAIN.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                epoch, i, len(train_loader), batch_time = batch_time,
                speed = inputs.size(0) / batch_time.val,
                data_time = data_time, loss = losses)
            debug_print(msg)


def validate(config, val_loader, val_dataset, model, criterion, output_dir):
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
            t()
            torch.cuda.empty_cache()
            t()
            output = model(inputs)  # torch.Size([bs, 16/17, 96, 96])
            t()
            target = target.cuda(non_blocking = True)
            loss = criterion(output, target)
            
            
            num_images = inputs.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            
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
    
    mp.set_start_method('spawn', force = True)
    # warnings.filterwarnings('ignore')
    # make_nise_dirs()
    
    if nise_cfg.TEST.MODE == 'valid':
        dataset_path = nise_cfg.PATH.GT_VAL_ANNOTATION_DIR
    elif nise_cfg.TEST.MODE == 'train':
        dataset_path = nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR
    
    val_dataset = mNetDataset(nise_cfg, nise_cfg.PATH.GT_VAL_ANNOTATION_DIR,
                              nise_cfg.PATH.PRED_JSON_VAL_FOR_TRAINING_MNET,
                              nise_cfg.PATH.UNI_BOX_VAL_FOR_TRAINING_MNET, False)
    
    train_dataset = mNetDataset(nise_cfg, nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR,
                                nise_cfg.PATH.PRED_JSON_TRAIN_FOR_TRAINING_MNET,
                                nise_cfg.PATH.UNI_BOX_TRAIN_FOR_TRAINING_MNET, True)
    debug_print("Init Network...")
    model = MatchingNet(nise_cfg.MODEL.INPUTS_CHANNELS)
    debug_print("Done")
    gpus = [int(i) for i in os.environ.get('CUDA_VISIBLE_DEVICES', default = '').split(',')]
    debug_print("Distribute Network to GPUs...", gpus)
    model = torch.nn.DataParallel(model, device_ids = gpus).cuda()
    debug_print("Done")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = nise_cfg.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle = nise_cfg.TRAIN.SHUFFLE,
        num_workers = nise_cfg.TRAIN.WORKERS,
        pin_memory = False
    )
    
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = nise_cfg.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle = False,
        num_workers = nise_cfg.TRAIN.WORKERS,
        pin_memory = False,
    )
    debug_print("Done")
    model.train()
    loss_calc = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr = nise_cfg.TRAIN.LR
    )
    final_output_dir = nise_cfg.PATH.MODEL_SAVE_DIR_FOR_TRAINING_MNET
    debug_print("BEGIN TO TRAIN.", lvl = Levels.SUCCESS)
    t()
    # for i in train_dataset:
    #     print(i[0].shape,i[1].shape)
    
    # for i, (inputs, target) in enumerate(train_loader):
    #     print(inputs.shape,target.shape)
    
    for epoch in range(nise_cfg.TRAIN.START_EPOCH, nise_cfg.TRAIN.END_EPOCH):
        train_1_ep(nise_cfg, train_loader, model, loss_calc, optimizer, epoch)
        t()
        perf_indicator = validate(nise_cfg, valid_loader, val_dataset, model,
                                  loss_calc, final_output_dir)
        ap = perf_indicator['ap']
        debug_print('=> saving checkpoint to {}'.format(final_output_dir))
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir, 'ep-{}-{}.pkl'.format(epoch + 1, ap)))
