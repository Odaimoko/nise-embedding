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
from nise_lib.dataset.mNetDataset_debug import mNetDataset as mNetDataset_debug
from nise_lib.dataset.mNetDataset import mNetDataset
from nise_lib.dataset.mNetDataset_by_single_pair import mNetDataset as pair_dataset
from nise_lib.nise_models import MatchingNet
from mem_util.gpu_mem_track import MemTracker
import inspect
from nise_lib.data import dataloader as my_dataloader

from nise_lib import nise_functions

print('MAIN', nise_functions)


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
        torch.cuda.empty_cache()
        target = target.view([-1])
        debug_print('Before target', target.numel())
        bs, img_bs, C, H, W = inputs.shape
        inputs = inputs.view([-1, C, H, W])
        inputs = inputs[target != -1]
        target = target[target != -1].view([-1, 1])  # to match output
        debug_print("AFter target", target.numel())
        data_time.update(time.time() - end)
        output = model(inputs)
        target = target.cuda(non_blocking = True)
        
        # target_weight?
        loss = criterion(output, target)
        
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
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
                epoch, i + 1, len(train_loader), batch_time = batch_time,
                speed = inputs.size(0) / batch_time.val,
                data_time = data_time, loss = losses)
            debug_print(msg)


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
            
            output = model(inputs.cuda(1))  # torch.Size([bs, 16/17, 96, 96])
            
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
    np.set_printoptions(suppress = True)
    # mp.set_start_method('spawn', force = True)
    
    gpus = list(range(len(os.environ.get('CUDA_VISIBLE_DEVICES', default = '0').split(','))))
    #
    # maskRCNN = None
    # if nise_cfg.DEBUG.load_human_det_model:
    #     human_detect_args = human_detect_parse_args()
    #     maskRCNN, human_det_dataset = load_human_detect_model(human_detect_args, tron_cfg)
    # model = MatchingNet(nise_cfg.MODEL.INPUTS_CHANNELS, maskRCNN)
    # model = torch.nn.DataParallel(model).cuda()
    #
    train_dataset = mNetDataset(nise_cfg, nise_cfg.PATH.GT_TRAIN_ANNOTATION_DIR,
                                nise_cfg.PATH.PRED_JSON_TRAIN_FOR_TRAINING_MNET,
                                nise_cfg.PATH.UNI_BOX_TRAIN_FOR_TRAINING_MNET, True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 16 * len(gpus),
        shuffle = False,
        num_workers = 0,
        pin_memory = True
    )
    for inputs, scales, joints_heatmap, labels, meta_info in train_loader:
        all_samples = meta_info['all_samples']
        # num_total_samples = meta_info['num_total_samples']
        # db_entry = meta_info['db_entry']
        # print(inputs.shape)  # bs x 2,3,562,1000
        #
        # bs, _, C, H, W = inputs.shape
        # inputs = inputs.view([-1, C, H, W]).cuda()
        # with torch.no_grad():
        #     fmaps = model.module.conv_body(inputs)
        # torch.cuda.empty_cache()
        # _, C, H, W = fmaps.shape
        # fmaps = fmaps.view([bs, -1, C, H, W])
        # idx = all_samples.sum(2) > 0  # bs x 8
        # valid_samples = all_samples[idx]
        # valid_joints_heatmap = joints_heatmap[idx]
        # valid_labels = labels[idx]
        # out = model(fmaps, scales, all_samples, joints_heatmap, idx)  # (bsx2) x CHW
        # # print(scales.shape)  # bs,2
        # # pprint.pprint(all_samples)  # bs, 8 ,6
        # # pprint.pprint(num_total_samples)  # tensor([ 8,  8,  8,  8])
        # # pprint.pprint(db_entry)
        # print(labels)
        # print(idx)
        # print(out.shape, idx.sum())
        # print()
    
    # for i, (inputs, target) in enumerate(train_loader):
    #     print(inputs.shape, target.shape)
    
    loss_calc = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr = nise_cfg.TRAIN.LR
    )
    model_path = 'mnet_output/ep-22-0.pkl'
    meta_info = torch.load(model_path)
    model.load_state_dict(meta_info['state_dict'])
    optimizer.load_state_dict(meta_info['optimizer'])
    #
    # val_pair = pair_dataset(nise_cfg, nise_cfg.PATH.GT_VAL_ANNOTATION_DIR,
    #                         nise_cfg.PATH.PRED_JSON_VAL_FOR_TRAINING_MNET,
    #                         nise_cfg.PATH.UNI_BOX_VAL_FOR_TRAINING_MNET, False)
    # valid_loader = my_dataloader.DataLoader(
    #     val_pair,
    #     batch_size = 24 * len(gpus),
    #     shuffle = False,
    #     num_workers = 8,
    #     pin_memory = False,
    # )
    # val_pair.vis_one_pair(1659)
    # with Pool() as po:
    #     debug_print('Pool created.')
    #     po.map(val_pair.vis_one_pair, [i for i in range(len(val_pair))])
    # p2 = val_using_loader(nise_cfg, valid_loader, val_pair, model)
    # print(p2)
    # torch.save(p2, 'mNet_eval_result.pkl')
    
    # val_dataset = mNetDataset(nise_cfg, nise_cfg.PATH.GT_VAL_ANNOTATION_DIR,
    #                           nise_cfg.PATH.PRED_JSON_VAL_FOR_TRAINING_MNET,
    #                           nise_cfg.PATH.UNI_BOX_VAL_FOR_TRAINING_MNET, False, None)
    #
    # # p1 = validate(nise_cfg, val_dataset)
    # threads = []
    # threads.append(threading.Thread(target = val_using_loader, args = (nise_cfg, valid_loader, val_pair, model)))
    # threads.append(threading.Thread(target = validate, args = (nise_cfg, val_dataset)))
    # for t in threads:
    #     t.start()
    # for t in threads:
    #     t.join()
    # perf_indicator = validate_model(nise_cfg, val_dataset, model, None)
