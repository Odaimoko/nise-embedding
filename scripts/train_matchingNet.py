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
from nise_lib.dataset.mNetDataset import mNetDataset
from nise_lib.dataset.mNetDataset_by_single_pair import mNetDataset as pair_dataset

from nise_lib.nise_models import *
from mem_util.gpu_mem_track import MemTracker
import inspect
from pdb import set_trace

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
    acc = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    sig = torch.nn.Sigmoid()
    
    for i, (inputs, scales, joints_heatmap, target, meta_info) in enumerate(train_loader):
        # if i < 63: continue
        all_samples = meta_info['all_samples']
        
        idx = all_samples.sum(2) > 0  # bs x 8
        data_time.update(time.time() - end)
        output = model(inputs, scales, all_samples, joints_heatmap, idx)  # (bsx2) x CHW
        # debug_print('Before target', target.numel())
        target = target[idx]
        # debug_print("AFter target", target.numel())
        target = target.cuda(non_blocking = True).view([-1, 1])
        loss = criterion(output, target)
        
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))
        
        score = sig(output)
        
        avg_acc = accuracy(target.detach().cpu().numpy(), score.detach().cpu().numpy(),
                           nise_cfg.TEST.POSITIVE_PAIR_THRES)
        
        acc.update(avg_acc, len(target))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % config.TRAIN.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time = batch_time,
                speed = inputs.size(0) / batch_time.val,
                data_time = data_time, loss = losses, acc = acc)
            debug_print(msg)

@log_time("Validating...")
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
    # warnings.filterwarnings('ignore')
    # make_nise_dirs()
    
    debug_print(pprint.pformat(nise_cfg), lvl = Levels.SKY_BLUE)
    gpus = os.environ.get('CUDA_VISIBLE_DEVICES', default = '0').split(',')
    gpus = list(range(len(gpus)))
    
    debug_print("Init Network...")
    maskRCNN = None
    if nise_cfg.DEBUG.load_human_det_model:
        human_detect_args = human_detect_parse_args()
        maskRCNN, human_det_dataset = load_human_detect_model(human_detect_args, tron_cfg)
    
    meta_info = None
    if nise_cfg.PATH.mNet_MODEL_FILE:
        debug_print("Load model from ", nise_cfg.PATH.mNet_MODEL_FILE)
        model, meta_info = load_mNet_model(nise_cfg.PATH.mNet_MODEL_FILE, maskRCNN)
    else:
        model = MatchingNet(nise_cfg.MODEL.INPUTS_CHANNELS, maskRCNN)
        debug_print("Done")
        debug_print("Distribute Network to GPUs...", gpus)
        model = torch.nn.DataParallel(model, device_ids = gpus).cuda()
        debug_print("Done")
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = nise_cfg.TRAIN.LR
    )
    
    if meta_info:
        optimizer.load_state_dict(meta_info['optimizer'])
        nise_cfg.TRAIN.START_EPOCH = meta_info['epoch']
    
    loss_calc = torch.nn.BCEWithLogitsLoss()
    
    # val_dataset = mNetDataset(nise_cfg, nise_cfg.PATH.GT_VAL_ANNOTATION_DIR,
    #                           nise_cfg.PATH.PRED_JSON_VAL_FOR_TRAINING_MNET,
    #                           nise_cfg.PATH.UNI_BOX_VAL_FOR_TRAINING_MNET, False, maskRCNN)
    
    val_pair = pair_dataset(nise_cfg, nise_cfg.PATH.GT_VAL_ANNOTATION_DIR,
                            nise_cfg.PATH.PRED_JSON_VAL_FOR_TRAINING_MNET,
                            nise_cfg.PATH.UNI_BOX_VAL_FOR_TRAINING_MNET, False)
    train_dataset = mNetDataset(nise_cfg, nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR,
                                nise_cfg.PATH.PRED_JSON_TRAIN_FOR_TRAINING_MNET,
                                nise_cfg.PATH.UNI_BOX_TRAIN_FOR_TRAINING_MNET, True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = nise_cfg.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle = nise_cfg.TRAIN.SHUFFLE,
        num_workers = nise_cfg.TRAIN.WORKERS,
        pin_memory = True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        val_pair,
        batch_size = nise_cfg.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle = False,
        num_workers = nise_cfg.TRAIN.WORKERS,
        pin_memory = False,
    )
    model.train()
    final_output_dir = nise_cfg.PATH.MODEL_SAVE_DIR_FOR_TRAINING_MNET
    debug_print("BEGIN TO TRAIN.", lvl = Levels.SUCCESS)
    
    # for i in train_dataset:
    #     print(i[0].shape,i[1].shape)
    
    for epoch in range(nise_cfg.TRAIN.START_EPOCH, nise_cfg.TRAIN.END_EPOCH):
        train_1_ep(nise_cfg, train_loader, model, loss_calc, optimizer, epoch)
        
        # perf_indicator = validate(nise_cfg, val_pair, model, final_output_dir)
        # perf_indicator = val_using_loader(nise_cfg, valid_loader, loss_calc, val_pair, model)
        # ap = perf_indicator['ap']
        perf_indicator = None
        ap = 0
        pklname = os.path.join(final_output_dir, 'ep-{}-{}.pkl'.format(epoch + 1, ap))
        debug_print('=> saving checkpoint to {}'.format(pklname))
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, pklname)
