import argparse
import json
import os
import pathlib
import torch
import sys
import time
import numpy as np
from functools import wraps
from pathlib import PurePosixPath
import yaml

import colorama

# local packages

from simple_lib.core.config import config as simple_cfg
from simple_lib.core.config import update_config
from simple_models.pose_resnet import get_pose_net as get_simple_pose_net
from tron_lib.core.test_for_pt import _get_blobs
from nise_lib.nise_config import mkrs
from nise_lib.nise_debugging_func import *
from nise_utils.imutils import *
from plogs.logutils import Levels
from nise_lib.nise_models import MatchingNet
from mem_util.gpu_mem_track import MemTracker
import inspect

# DEBUGGING
gpuMemTrack = MemTracker(inspect.currentframe()).track


def debug_print(*args, indent = 0, lvl = Levels.INFO):
    args = [str(a) for a in args]
    msg = ''.join(['\t'] * indent) + ' '.join(args)
    if nise_cfg.DEBUG.PRINT:
        global nise_logger
        nise_logger._log(msg, lvl)


# DECORATORS

def log_time(*text, record = None):
    def real_deco(func):
        @wraps(func)
        def impl(*args, **kw):
            r = debug_print if not record else record  # 如果没有record，默认print
            t = (func.__name__,) if not text else text
            start = time.time()
            result = func(*args, **kw)
            end = time.time()
            r(*t, '%.3f s.' % (end - start,), lvl = Levels.STATUS)
            return result
        
        return impl
    
    return real_deco


# ─── LOAD MODEL ─────────────────────────────────────────────────────────────────

from tron_lib.core.config import cfg_from_file, cfg_from_list, assert_and_infer_cfg
from tron_lib.modeling.model_builder import Generalized_RCNN_for_posetrack
import tron_lib.nn as mynn
from tron_lib.tron_utils.detectron_weight_helper import load_detectron_weight
import tron_lib.tron_utils.net as net_utils
import tron_lib.datasets.dummy_datasets as datasets


def human_detect_parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description = 'Train a X-RCNN network')
    
    parser.add_argument(
        '--dataset', dest = 'dataset', required = True,
        help = 'Dataset to use')
    
    parser.add_argument(
        '--tron_cfg', dest = 'cfg_file', required = True,
        help = 'Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest = 'set_cfgs',
        help = 'Set config keys. Key value sequence seperate by whitespace.'
               'e.g. [key] [value] [key] [value]',
        default = [], nargs = '+')
    
    parser.add_argument(
        '--no_cuda', dest = 'cuda', help = 'Do not use CUDA device', action = 'store_false')
    
    parser.add_argument(
        '--load_ckpt', help = 'checkpoint path to load')
    parser.add_argument(
        '--load_detectron', help = 'path to the detectron weight pickle file')
    
    args, rest = parser.parse_known_args()
    return args


def load_human_detect_model(args, tron_cfg):
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")
    # print('Called with args:')
    # print(args)
    
    if args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        tron_cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
        tron_cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))
    
    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    # When testing, this is set to True. WHen inferring, this is False
    # QQ: Why????????????????????????
    # Don't need to load imagenet pretrained weights
    tron_cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False
    
    assert_and_infer_cfg()
    
    model = Generalized_RCNN_for_posetrack(tron_cfg)
    model.eval()
    if args.cuda:
        model.cuda()
    
    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(
            load_name, map_location = lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])
    
    if args.load_detectron:
        logger.info("loading detectron weights %s", args.load_detectron)
        load_detectron_weight(model, args.load_detectron)
    
    model = mynn.DataParallel(
        model, cpu_keywords = ['im_info', 'roidb'], minibatch = True)
    
    return model, dataset


def flow_init_parser_and_tools(parser, tools):
    import flow_datasets
    import flow_losses
    import flow_models
    parser.add_argument('--crop_size', type = int, nargs = '+', default = [256, 256],
                        help = "Spatial dimension to crop training samples for training")
    parser.add_argument('--gradient_clip', type = float, default = None)
    parser.add_argument('--schedule_lr_frequency', type = int, default = 0,
                        help = 'in number of iterations (0 for no schedule)')
    parser.add_argument('--schedule_lr_fraction', type = float, default = 10)
    parser.add_argument("--rgb_max", type = float, default = 255.)
    
    parser.add_argument('--number_workers', '-nw',
                        '--num_workers', type = int, default = 8)
    parser.add_argument('--number_gpus', '-ng', type = int,
                        default = -1, help = 'number of GPUs to use')
    parser.add_argument('--no_cuda', action = 'store_true')
    
    parser.add_argument('--seed', type = int, default = 1)
    parser.add_argument('--name', default = 'run', type = str,
                        help = 'a name to append to the save directory')
    parser.add_argument('--save', '-s', default = './work',
                        type = str, help = 'directory for saving')
    
    parser.add_argument('--validation_frequency', type = int,
                        default = 5, help = 'validate every n epochs')
    parser.add_argument('--validation_n_batches', type = int, default = -1)
    parser.add_argument('--render_validation', action = 'store_true',
                        help = 'run inference (save flows to file) and every validation_frequency epoch')
    
    parser.add_argument('--inference', action = 'store_true')
    parser.add_argument('--inference_size', type = int, nargs = '+', default = [-1, -1],
                        help = 'spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    parser.add_argument('--inference_batch_size', type = int, default = 1)
    parser.add_argument('--inference_n_batches', type = int, default = -1)
    parser.add_argument('--save_flow', action = 'store_true',
                        help = 'save predicted flows to file')
    
    parser.add_argument('--flownet_resume', default = '', type = str, metavar = 'PATH',
                        help = 'path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter',
                        type = int, default = 1, help = "Log every n batches")
    
    parser.add_argument('--fp16', action = 'store_true',
                        help = 'Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type = float, default = 1024.,
                        help = 'Loss scaling, positive power of 2 values can improve fp16 convergence.')
    
    # ─── TOOLS ──────────────────────────────────────────────────────────────────────
    
    tools.add_arguments_for_module(
        parser, flow_models, argument_for_class = 'model', default = 'FlowNet2')
    
    tools.add_arguments_for_module(
        parser, flow_losses, argument_for_class = 'loss', default = 'L1Loss')
    
    tools.add_arguments_for_module(parser, torch.optim, argument_for_class = 'optimizer', default = 'Adam',
                                   skip_params = ['params'])
    
    tools.add_arguments_for_module(parser, flow_datasets, argument_for_class = 'training_dataset',
                                   default = 'MpiSintelFinal',
                                   skip_params = ['is_cropped'],
                                   parameter_defaults = {'root': './MPI-Sintel/flow/training'})
    
    tools.add_arguments_for_module(parser, flow_datasets, argument_for_class = 'validation_dataset',
                                   default = 'MpiSintelClean',
                                   skip_params = ['is_cropped'],
                                   parameter_defaults = {'root': './MPI-Sintel/flow/training',
                                                         'replicates': 1})
    
    tools.add_arguments_for_module(parser, flow_datasets, argument_for_class = 'inference_dataset',
                                   default = 'MpiSintelClean',
                                   skip_params = ['is_cropped'],
                                   parameter_defaults = {'root': './MPI-Sintel/flow/training',
                                                         'replicates': 1})


def load_flow_model(args, parser, tools):
    import flow_datasets
    import flow_losses
    import flow_models
    # Parse the official arguments
    with tools.TimerBlock("Parsing Arguments") as block:
        if args.number_gpus < 0:
            args.number_gpus = torch.cuda.device_count()
        
        # Get argument defaults (hastag #thisisahack)
        parser.add_argument('--IGNORE', action = 'store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))
        
        # Print all arguments, color the non-defaults
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))
        
        args.model_class = tools.module_to_dict(flow_models)[args.model]
        args.optimizer_class = tools.module_to_dict(torch.optim)[
            args.optimizer]
        args.loss_class = tools.module_to_dict(flow_losses)[args.loss]
        
        args.training_dataset_class = tools.module_to_dict(flow_datasets)[
            args.training_dataset]
        args.validation_dataset_class = tools.module_to_dict(flow_datasets)[
            args.validation_dataset]
        args.inference_dataset_class = tools.module_to_dict(flow_datasets)[
            args.inference_dataset]
        
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.log_file = os.path.join(args.save, 'args.txt')
        
        # dict to collect activation gradients (for training debug purpose)
        args.grads = {}
        
        if args.inference:
            args.skip_validation = True
            args.skip_training = True
            args.total_epochs = 1
            args.inference_dir = "{}/inference".format(args.save)
    
    # Dynamically load model and loss class with parameters passed in
    # via "--model_[param]=[value]" or "--loss_[param]=[value]" arguments
    with tools.TimerBlock("Building {} model".format(args.model)) as block:
        class ModelAndLoss(nn.Module):
            def __init__(self, args):
                super(ModelAndLoss, self).__init__()
                kwargs = tools.kwargs_from_args(args, 'model')
                self.model = args.model_class(args, **kwargs)
                kwargs = tools.kwargs_from_args(args, 'loss')
                self.loss = args.loss_class(args, **kwargs)
            
            def forward(self, data):
                output = self.model(data)
                return output
                # loss_values = self.loss(output, target)
                #
                # if not inference:
                #     return loss_values
                # else:
                #     return loss_values, output
        
        model_and_loss = ModelAndLoss(args)
        
        block.log('Number of parameters: {}'.format(
            sum([p.data.nelement() if p.requires_grad else 0 for p in model_and_loss.parameters()])))
        
        # assing to cuda or wrap with dataparallel, model and loss
        if args.cuda and (args.number_gpus > 0) and args.fp16:
            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(
                model_and_loss, device_ids = list(range(args.number_gpus)))
            
            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda().half()
            torch.cuda.manual_seed(args.seed)
            param_copy = [param.clone().type(torch.cuda.FloatTensor).detach()
                          for param in model_and_loss.parameters()]
        
        elif args.cuda and args.number_gpus > 0:
            block.log('Initializing CUDA')
            model_and_loss = model_and_loss.cuda()
            block.log('Parallelizing')
            model_and_loss = nn.parallel.DataParallel(
                model_and_loss, device_ids = list(range(args.number_gpus)))
            torch.cuda.manual_seed(args.seed)
        
        else:
            block.log('CUDA not being used')
            torch.manual_seed(args.seed)
        
        # Load weights if needed, otherwise randomly initialize
        # 要用这个
        if args.flownet_resume and os.path.isfile(args.flownet_resume):
            block.log("Loading checkpoint '{}'".format(args.flownet_resume))
            checkpoint = torch.load(args.flownet_resume)
            if not args.inference:
                args.start_epoch = checkpoint['epoch']
            best_err = checkpoint['best_EPE']
            model_and_loss.module.model.load_state_dict(
                checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}' (at epoch {})".format(
                args.flownet_resume, checkpoint['epoch']))
        
        block.log("Initializing save directory: {}".format(args.save))
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        
        # train_logger = SummaryWriter(log_dir = os.path.join(
        #     args.save, 'train'), comment = 'training')
        # validation_logger = SummaryWriter(log_dir = os.path.join(
        #     args.save, 'validation'), comment = 'validation')
    
    return model_and_loss


def load_simple_model():
    def reset_config(config, args):
        if args.gpus:
            config.GPUS = args.gpus
        if args.workers:
            config.WORKERS = args.workers
        
        if args.simple_model_file:
            config.TEST.MODEL_FILE = args.simple_model_file
    
    def simple_parse_args():
        parser = argparse.ArgumentParser(description = 'Train keypoints network')
        # general
        parser.add_argument('--simple_cfg',
                            help = 'experiment configure file name',
                            type = str)
        
        args, rest = parser.parse_known_args()
        # update config
        update_config(args.simple_cfg)
        
        # training
        parser.add_argument('--frequent',
                            help = 'frequency of logging',
                            default = simple_cfg.PRINT_FREQ,
                            type = int)
        parser.add_argument('--gpus',
                            help = 'gpus',
                            type = str, default = '0')
        parser.add_argument('--workers',
                            help = 'num of dataloader workers',
                            type = int, default = 8)
        
        parser.add_argument('--simple-model-file',
                            help = 'model state file',
                            type = str)
        
        args, rest = parser.parse_known_args()
        
        return args
    
    simple_args = simple_parse_args()
    reset_config(simple_cfg, simple_args)
    
    simple_human_det_model = get_simple_pose_net(
        simple_cfg, is_train = True
    )
    gpus = [int(i) for i in simple_cfg.GPUS.split(',')]
    
    if simple_cfg.TEST.MODEL_FILE:
        meta_info = torch.load(simple_cfg.TEST.MODEL_FILE)
        if 'pt17-epoch' in simple_cfg.TEST.MODEL_FILE:
            state_dict = {k.replace('module.', ''): v
                          for k, v in meta_info['state_dict'].items()}
        else:
            state_dict = meta_info
        simple_human_det_model.load_state_dict(state_dict)
    simple_human_det_model = torch.nn.DataParallel(
        simple_human_det_model, device_ids = gpus).cuda()
    return simple_args, simple_human_det_model


from hr_lib.config import cfg as hr_cfg
from hr_lib.models.pose_hrnet import get_pose_net as get_hr_pose_net


def load_hr_model():
    def hr_update_cfg(cfg, args):
        cfg.defrost()
        cfg.merge_from_file(args.hr_cfg)
        
        cfg.DATASET.ROOT = os.path.join(
            cfg.DATA_DIR, cfg.DATASET.ROOT
        )
        
        cfg.MODEL.PRETRAINED = os.path.join(
            cfg.DATA_DIR, cfg.MODEL.PRETRAINED
        )
        
        cfg.freeze()
    
    def hr_parse_args():
        parser = argparse.ArgumentParser(description = 'Train keypoints network')
        # general
        parser.add_argument('--hr-cfg',
                            help = 'experiment configure file name',
                            type = str)
        
        parser.add_argument('--hr-model',
                            help = 'model state file',
                            type = str)
        
        args, rest = parser.parse_known_args()
        
        return args
    
    hr_args = hr_parse_args()
    hr_update_cfg(hr_cfg, hr_args)
    
    simple_human_det_model = get_hr_pose_net(
        hr_cfg, is_train = True
    )
    
    assert (hr_cfg.TEST.MODEL_FILE is not None)
    meta_info = torch.load(hr_args.hr_model)
    state_dict = {k.replace('module.', ''): v
                  for k, v in meta_info['state_dict'].items()}
    simple_human_det_model.load_state_dict(state_dict, strict = False)
    simple_human_det_model = torch.nn.DataParallel(
        simple_human_det_model, device_ids = hr_cfg.GPUS).cuda()
    return hr_args, simple_human_det_model


# ─── USE MODEL ──────────────────────────────────────────────────────────────────


# Reusable function for inference
def pred_flow(two_images, model):
    '''
        Already cudaed
    :param two_images: channels, 2, h, w
    :param model:
    :return:
    '''
    model.eval()
    
    c, _, h, w = two_images.shape
    with torch.no_grad():
        # data[0] torch.Size([8, 3, 2, 384, 1024]) ，bs x channels x num_images, h, w
        # target torch.Size([8, 2, 384, 1024]) maybe offsets
        # losses: list (2)
        # output: torch.Size([bs, 2, 384, 1024])
        two_images = torch.unsqueeze(two_images, 0)  # batchize
        # losses, output = model(
        #     data=two_images, target=gen_rand_flow(1, h, w), inference=True)
        output = model(two_images)
        output.squeeze_()  # out is a batch, so remove the zeroth dimension
    return output


# ─── CHECKPOINT UTIL ────────────────────────────────────────────────────────────


from tron_lib.tron_utils.logging import setup_logging

logger = setup_logging(__name__)


# ─── IMAGE UTILS ────────────────────────────────────────────────────────────────

def imcrop(img, bbox):
    def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
        img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
                           (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0, 0)), mode = "constant")
        y1 += np.abs(np.minimum(0, y1))
        y2 += np.abs(np.minimum(0, y1))
        x1 += np.abs(np.minimum(0, x1))
        x2 += np.abs(np.minimum(0, x1))
        return img, x1, x2, y1, y2
    
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]


# ─── BOX UTILS ──────────────────────────────────────────────────────────────────


def joints_to_boxes(new_joints, joint_vis = None, clamp_size = ()):
    '''

    :param new_joints:  num_people x num_joints x 2
    :param joint_vis: if some joint for a person is invisible, the coord will be 0, dont include it in min. max is not affected
    :param clamp_size: if none, dont clamp; if 2-element list,(w,h)
    :return:
    '''
    # copy
    new_joints = torch.tensor(new_joints)
    num_people, num_joints, _ = new_joints.shape
    if joint_vis is None:
        joint_vis = torch.ones(num_people, num_joints)
    joint_invis = joint_vis == 0
    for_min = torch.zeros(num_people, num_joints)
    for_min[joint_invis] = 9999
    # for_max = torch.zeros(num_people, num_joints)
    min_xs, _ = torch.min(new_joints[:, :, 0] + for_min, 1)
    min_ys, _ = torch.min(new_joints[:, :, 1] + for_min, 1)
    max_xs, _ = torch.max(new_joints[:, :, 0], 1)
    max_ys, _ = torch.max(new_joints[:, :, 1], 1)
    # extend by a centain factor
    ws = max_xs - min_xs
    hs = max_ys - min_ys
    ws = ws * nise_cfg.DATA.bbox_extend_factor[0]
    hs = hs * nise_cfg.DATA.bbox_extend_factor[1]
    min_xs -= ws
    max_xs += ws
    min_ys -= hs
    max_ys += hs
    if clamp_size:
        min_xs.clamp_(0, clamp_size[0])
        max_xs.clamp_(0, clamp_size[0])
        min_ys.clamp_(0, clamp_size[1])
        max_ys.clamp_(0, clamp_size[1])
    
    joint_prop_bboxes = torch.stack([
        min_xs, min_ys, max_xs, max_ys
    ], 1)
    return joint_prop_bboxes


# From simple_lib.dataset.coco
def box2cs(box, ratio):
    '''

    :param box: with x1y1x2y2
    :param ratio:
    :return:
    '''
    
    # our bbox is x1 y1, x2 y2, _xywh2cs takes x y w h
    bb = np.copy(box)
    bb[2], bb[3] = bb[2] - bb[0], bb[3] - bb[1]
    x, y, w, h = bb[:4]
    return xywh2cs(x, y, w, h, ratio)


def xywh2cs(x, y, w, h, training_bbox_aspect_ratio):
    pixel_std = 200
    center = np.zeros((2), dtype = np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    
    if w > training_bbox_aspect_ratio * h:
        h = w * 1.0 / training_bbox_aspect_ratio
    elif w < training_bbox_aspect_ratio * h:
        w = h * training_bbox_aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype = np.float32)
    if center[0] != -1:
        scale = scale * 1.25
    
    return center, scale


# from https://github.com/tensorflow/models/blob/master/research/object_detection/utils/np_box_ops.py
def area(boxes):
    """Computes area of boxes.
    Args:
      boxes: Numpy array with shape [N, 4] holding N boxes
    Returns:
      a numpy array with shape [N*1] representing box areas
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.
    Args:
      boxes1: a numpy array with shape [N, 4] holding N boxes
      boxes2: a numpy array with shape [M, 4] holding M boxes
    Returns:
      a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis = 1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis = 1)
    
    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape),
        all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape),
        all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def tf_iou(boxes1: np.ndarray, boxes2: np.ndarray):
    """Computes pairwise intersection-over-union between box collections.
    Args:
      boxes1: a numpy array with shape [N, 4] holding N boxes.
      boxes2: a numpy array with shape [M, 4] holding N boxes.
    Returns:
      a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2)
    area1 = area(boxes1)
    area2 = area(boxes2)
    union = np.expand_dims(area1, axis = 1) + np.expand_dims(
        area2, axis = 0) - intersect
    return intersect / union


def unioned_box(box1, box2):
    '''
    input size should be (4/5,) 
    :param box1: 
    :param box2: 
    :return: The minimum box contains both
    '''
    assert (box1 >= 0).sum() == (box2 >= 0).sum() and (box2 >= 0).sum() == 4
    u = [
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3]),
    ]
    if isinstance(box1, np.ndarray):
        u = np.array(u)
    else:
        assert isinstance(box1, torch.Tensor)
        u = torch.tensor(u)
    return u


def filter_bbox_with_scores(boxes, thres = nise_cfg.ALG._HUMAN_THRES):
    if boxes.numel() == 0:
        return boxes, torch.tensor([])
    scores = boxes[:, -1]
    valid_scores_idx = torch.nonzero(scores >= thres).squeeze_().long()  # in case it's 6 x **1** x 5
    filtered_box = boxes[valid_scores_idx, :]
    filtered_box = expand_vector_to_tensor(filtered_box)
    valid_scores_idx = expand_vector_to_tensor(valid_scores_idx,1)
    return filtered_box, valid_scores_idx


def filter_bbox_with_area(boxes, thres = nise_cfg.ALG._AREA_THRES):
    if boxes.numel() == 0:
        return boxes, torch.tensor([])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    valid_area_idx = torch.nonzero(
        area >= thres).squeeze_().long()  # in case it's 6 x **1** x 5
    filtered_box = expand_vector_to_tensor(boxes[valid_area_idx, :])
    return filtered_box, valid_area_idx


# @log_time('Getting box_fmap...')
def get_box_fmap(fmap_info: dict, boxes, method = 'pool'):
    '''
    :param boxes: torch tensor, bs x 5
    :return:
    '''
    from tron_lib.modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
    from tron_lib.model.roi_pooling.functions.roi_pool import RoIPoolFunction
    
    fmap, scale = fmap_info['fmap'], fmap_info['scale'][0]
    boxes_np = boxes.numpy()
    rois_np = np.zeros([boxes_np.shape[0], 5])
    rois_np[:, 1:] = boxes_np[:, 0:4] * scale
    map_res = nise_cfg.MODEL.FEATURE_MAP_RESOLUTION
    
    rois = torch.from_numpy(rois_np)
    rois = rois.int().float()
    
    # debug_print(boxes, boxes.shape, lvl = Levels.ERROR)
    # debug_print(rois, lvl = Levels.CRITICAL)
    if method == 'pool':
        boxes_fmap = RoIPoolFunction(map_res, map_res, .25)(fmap, rois)
    else:
        boxes_fmap = RoIAlignFunction(map_res, map_res, .25, 2)(fmap.cuda(), rois.cuda())
    
    return boxes_fmap.cpu()


def gen_img_fmap(tron_cfg, original_img, maskRCNN):
    inputs, im_scale = _get_blobs(original_img, None, tron_cfg.TEST.SCALE, tron_cfg.TEST.MAX_SIZE)
    if tron_cfg.DEDUP_BOXES > 0 and not tron_cfg.MODEL.FASTER_RCNN:
        # No use but serves to check whether the yaml file is loaded to cfg
        v = inputs['rois']
    
    if isinstance(maskRCNN, mynn.DataParallel):
        mask = list(maskRCNN.children())[0]
    else:
        mask = maskRCNN
    
    with torch.no_grad():
        hai = mask.Conv_Body(torch.from_numpy(inputs['data']).cuda())
    return {
        'fmap': hai[3].cpu(),
        'scale': im_scale,
    }


# ─── MATCHING ───────────────────────────────────────────────────────────────────

def load_mNet_model(model_file):
    model = MatchingNet(nise_cfg.MODEL.INPUTS_CHANNELS)
    gpus = [int(i) for i in os.environ.get('CUDA_VISIBLE_DEVICES', default = '0').split(',')]
    debug_print(gpus)
    model = torch.nn.DataParallel(model, device_ids = gpus).cuda()
    meta_info = torch.load(model_file)
    model.load_state_dict(meta_info['state_dict'])
    return model


def get_joints_oks_mtx(j1, j2):
    '''

    :param j1: num_people 1 x 16 x 2
    :param j2: num_people 2 x 16 x 2
    :return: n1 x n2
    '''
    num_person_prev = j1.shape[0]
    num_person_cur = j2.shape[0]
    j1 = to_numpy(j1)
    j2 = to_numpy(j2)
    
    # sigma = np.array([
    #     .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
    #     .87, .89, .89]) / 10.0
    sigma = np.ones(nise_cfg.DATA.num_joints)
    var = sigma ** 2
    
    dist_mat = np.zeros(
        [num_person_prev, num_person_cur, nise_cfg.DATA.num_joints])
    for i in range(num_person_cur):
        diff_sq = (j1 - j2[i]) ** 2  # num_per_cur * 16 x 2
        eucl = diff_sq.sum(2)  # keypoint wise distance # num_per_cur * 16
        
        dist_mat[:, i, :] = eucl
    
    e = dist_mat / var / 2
    e = np.sum(np.exp(-e), axis = 2) / e.shape[2]
    return to_torch(e)


def bipartite_matching_greedy(C: np.ndarray):
    """
    Computes the bipartite matching between the rows and columns, given the
    cost matrix, C.
    """
    C = C.copy()  # to avoid affecting the original matrix
    prev_ids = []
    cur_ids = []
    row_ids = np.arange(C.shape[0])
    col_ids = np.arange(C.shape[1])
    while C.size > 0:
        # Find the lowest cost element
        i, j = np.unravel_index(C.argmin(), C.shape)
        # Add to results and remove from the cost matrix
        row_id = row_ids[i]
        col_id = col_ids[j]
        prev_ids.append(row_id)
        cur_ids.append(col_id)
        C = np.delete(C, i, 0)
        C = np.delete(C, j, 1)
        row_ids = np.delete(row_ids, i, 0)
        col_ids = np.delete(col_ids, j, 0)
    return prev_ids, cur_ids


def get_matching_indices(dist_mat: np.ndarray):
    '''
    
    :param dist_mat: n1 x n2
    :return:  result. each pair in result (a,b) means the a th of n1 <-> the b th of n2
    '''
    # to use munkres package, we need int. munkres minimize cost, so use negative version
    # but if converted to numpy, will have precision problem
    scaled_distance_matrix = -nise_cfg.ALG._OKS_MULTIPLIER * dist_mat
    # scaled_distance_matrix = scaled_distance_matrix.numpy()
    mask = (scaled_distance_matrix <= -1e-9).astype(np.float32)
    scaled_distance_matrix *= mask
    indices = mkrs.compute(scaled_distance_matrix.tolist())
    return indices


# ─── MISC ───────────────────────────────────────────────────────────────────────


def mkdir(path):
    path = path.strip().rstrip("\\")
    
    if not os.path.exists(path):
        os.makedirs(path)
        debug_print('Make dir', path)
        return True
    else:
        return False


def make_nise_dirs():
    mkdir(nise_cfg.PATH.IMAGES_OUT_DIR)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    mkdir(nise_cfg.PATH.JOINTS_DIR)
    mkdir(nise_cfg.PATH.DETECT_JSON_DIR)
    mkdir(nise_cfg.PATH.FLOW_JSON_DIR)
    mkdir(nise_cfg.PATH.DET_EST_JSON_DIR)
    mkdir(nise_cfg.PATH.UNIFIED_JSON_DIR)


def expand_vector_to_tensor(tensor, target_dim = 2):
    if tensor.numel() == 0:
        # size is 0
        return tensor
    while len(tensor.shape) < target_dim:  # vector
        tensor = tensor.unsqueeze(0)
    return tensor


def get_type_from_dir(dirpath, type_list):
    files = []
    for f in os.listdir(dirpath):
        if pathlib.PurePosixPath(f).suffix.lower() in type_list:
            files.append(os.path.join(dirpath, f))
    return files


def get_joints_from_annorects(annorects):
    if annorects is not None and len(annorects) == 0:
        return torch.tensor([]), torch.tensor([])
    all_joints = []
    is_from_gt = False
    for i in range(len(annorects)):
        rect = annorects[i]
        
        joints_3d = np.zeros((nise_cfg.DATA.num_joints, 3), dtype = np.float)
        points = rect['annopoints']
        # there's a person, but no annotations
        if points is None or len(points) <= 0:  # 因为有些图并没有annotation
            continue
        else:
            points = points[0]['point']
        for pt_info in points:
            # analogous to coco.py  # matlab based on 1.
            i_pt = pt_info['id'][0]
            if 'is_visible' in pt_info.keys():  # from gt
                is_from_gt = True
                joints_3d[i_pt, 0] = pt_info['x'][0] - 1 if pt_info['x'][0] > 0 else 0
                joints_3d[i_pt, 1] = pt_info['y'][0] - 1 if pt_info['y'][0] > 0 else 0
            else:  # from pred
                joints_3d[i_pt, 0] = pt_info['x'][0] if pt_info['x'][0] >= 0 else 0
                joints_3d[i_pt, 1] = pt_info['y'][0] if pt_info['y'][0] >= 0 else 0
            
            joints_3d[i_pt, 2] = 1  # t_vis and pt_info['x'][0] >= 0 and pt_info['y'][0] >= 0
        # head_bbox = [rect['x1'][0], rect['y1'][0], rect['x2'][0], rect['y2'][0]]
        # head_bbox = np.array(head_bbox)
        all_joints.append(joints_3d)
    
    joints = torch.tensor(all_joints).float()
    joints = expand_vector_to_tensor(joints)
    if is_from_gt:  # ones for those labeled
        scores = joints[:, :, 2]
    else:
        scores = get_pred_joint_scores(annorects)
    return joints, scores


def get_pred_joint_scores(annorects):
    all_scores = []
    for i in range(len(annorects)):
        rect = annorects[i]
        single_score = np.zeros((nise_cfg.DATA.num_joints), dtype = np.float)
        points = rect['annopoints']
        if points is None or len(points) <= 0:  # 因为有些图并没有annotation
            continue
        else:
            points = points[0]['point']
        for pt_info in points:
            i_pt = pt_info['id'][0]
            t_vis = pt_info['score'][0]
            single_score[i_pt] = t_vis
        all_scores.append(single_score)
    
    scores = torch.tensor(all_scores).float()
    
    return scores


def removeRectsWithoutPoints(rects):
    def rectHasPoints(rect):
        return ("annopoints" in rect.keys() and
                rect["annopoints"] is not None and
                (len(rect["annopoints"]) > 0 and len(rect["annopoints"][0]) > 0) and
                ("point" in rect["annopoints"][0].keys()))
    
    if rects is None:
        return []
    else:
        
        rects = [rect for rect in rects if rectHasPoints(rect)]
        return rects


def get_pck_from_anno(gt_annorects, pred_annorects):
    distThresh = 0.5
    MIN_SCORE = -9999
    
    def getPointGTbyID(points, pidx):
        point = []
        for i in range(len(points)):
            if (points[i]["id"] != None and points[i]["id"][0] == pidx):  # if joint id matches
                point = points[i]
                break
        
        return point
    
    def getHeadSize(x1, y1, x2, y2):
        headSize = 0.6 * np.linalg.norm(np.subtract([x2, y2], [x1, y1]))
        return headSize
    
    num_pred = len(pred_annorects)
    num_gt = len(gt_annorects)
    # print(num_pred, num_gt)
    
    if not num_gt or not num_pred:
        return np.array([]), np.array([])
    
    nJoints = 15
    # distance between predicted and GT joints
    dist = np.full((num_pred, num_gt, nJoints), np.inf)
    # score of the predicted joint
    score = np.full((num_pred, nJoints), np.nan)
    # body joint prediction exist
    hasPr = np.zeros((num_pred, nJoints), dtype = bool)
    # body joint is annotated
    hasGT = np.zeros((num_gt, nJoints), dtype = bool)
    
    idxsPr = []
    for ridxPr in range(num_pred):
        if (("annopoints" in pred_annorects[ridxPr].keys()) and
                ("point" in pred_annorects[ridxPr]["annopoints"][0].keys())):
            # 如果在prFrames里有anno，那就加入对应的 idx
            idxsPr += [ridxPr]
    # 这一句就是过滤一下
    pred_annorects = [pred_annorects[ridx] for ridx in idxsPr]
    
    # iterate over GT poses
    for ridxGT in range(num_gt):
        # GT pose
        rectGT = gt_annorects[ridxGT]  # 一个人
        pointsGT = []
        if len(rectGT["annopoints"]) > 0:
            pointsGT = rectGT["annopoints"][0]["point"]
        # iterate over all possible body joints
        for i in range(nJoints):
            # GT joint in LSP format
            ppGT = getPointGTbyID(pointsGT, i)
            if len(ppGT) > 0:
                hasGT[ridxGT, i] = True
    
    for ridxPr in range(num_pred):
        # predicted pose
        rectPr = pred_annorects[ridxPr]  # a person including track_id
        pointsPr = rectPr["annopoints"][0]["point"]
        for i in range(nJoints):
            # predicted joint in LSP format
            ppPr = getPointGTbyID(pointsPr, i)
            if len(ppPr) > 0:
                if not ("score" in ppPr.keys()):
                    score[ridxPr, i] = MIN_SCORE
                else:
                    score[ridxPr, i] = ppPr["score"][0]
                hasPr[ridxPr, i] = True
    
    # predictions and GT are present
    # iterate over GT poses
    # 计算两两之间的距离
    for ridxGT in range(num_gt):
        # GT pose
        rectGT = gt_annorects[ridxGT]
        # compute reference distance as head size
        headSize = getHeadSize(rectGT["x1"][0], rectGT["y1"][0],
                               rectGT["x2"][0], rectGT["y2"][0])
        pointsGT = []
        if len(rectGT["annopoints"]) > 0:
            pointsGT = rectGT["annopoints"][0]["point"]
        # iterate over predicted poses
        for ridxPr in range(num_pred):
            # predicted pose
            rectPr = pred_annorects[ridxPr]
            pointsPr = rectPr["annopoints"][0]["point"]
            
            # iterate over all possible body joints
            for i in range(nJoints):
                # GT joint
                ppGT = getPointGTbyID(pointsGT, i)
                # predicted joint
                ppPr = getPointGTbyID(pointsPr, i)
                # compute distance between predicted and GT joint locations
                if hasPr[ridxPr, i] and hasGT[ridxGT, i]:
                    pointGT = [ppGT["x"][0], ppGT["y"][0]]
                    pointPr = [ppPr["x"][0], ppPr["y"][0]]
                    dist[ridxPr, ridxGT, i] = np.linalg.norm(np.subtract(pointGT, pointPr)) / headSize
    
    dist = np.array(dist)
    hasGT = np.array(hasGT)
    
    # number of annotated joints，每个gt有多少个点标注了的
    nGTp = np.sum(hasGT, axis = 1)
    # 距离太小，根据headsize占比来match一个。
    # 由于最终输出的是人的trackid来决定joint的id，这里先匹配人，pck这个变量就是pred和gt里人的相近程度。
    match = dist <= distThresh
    pck = 1.0 * np.sum(match, axis = 2)
    for i in range(hasPr.shape[0]):
        for j in range(hasGT.shape[0]):
            if nGTp[j] > 0:
                pck[i, j] = pck[i, j] / nGTp[j]
    return pck, match


def find_det_for_gt_and_assignID(gt_annorects, pred_annorects, _nise_cfg):
    pck, match = get_pck_from_anno(gt_annorects, pred_annorects)
    
    # preserve best GT match only，因为有可能出现一个gt对应多个pred或者反之，这里只取分数最高的，把更小的entry都设置成0。
    idx = np.argmax(pck, axis = 1)  # 对pred来说，每个pred对应第几个gt
    val = np.max(pck, axis = 1)
    for ridxPr in range(pck.shape[0]):
        for ridxGT in range(pck.shape[1]):
            if (ridxGT != idx[ridxPr]):
                pck[ridxPr, ridxGT] = 0
    prToGT = np.argmax(pck, axis = 0)
    val = np.max(pck, axis = 0)
    prToGT[val == 0] = -1
    # print( prToGT) # 一个list，长度为 num_gt，entry 是第 i 个 gt对应的pred 的 index(数据类型是 ndarray)
    
    gt_id = [t['track_id'][0] for t in gt_annorects]
    # print(gt_id)
    det_id = []
    prToGT_return = []
    for i, ridxPr in enumerate(prToGT):
        if ridxPr != -1:
            if (_nise_cfg.DEBUG.USE_HIGH_PCKH_DET_BOX and
                pck[ridxPr, i] >= _nise_cfg.DEBUG.HIGH_PCKH_THRES) \
                    or not _nise_cfg.DEBUG.USE_HIGH_PCKH_DET_BOX:
                det_id.append(gt_id[i])
                prToGT_return.append(ridxPr)
                # debug_print(ridxPr, i, "{:.2f}".format(pck[ridxPr, i]))
    
    det_id = np.array(det_id)
    prToGT_return = np.array(prToGT_return)
    sort_prToGT_idx = np.argsort(prToGT_return)
    
    return prToGT_return[sort_prToGT_idx], det_id[sort_prToGT_idx]


def get_anno_matched_joints(gt_annorects, pred_annorects, _nise_cfg):
    '''
    find gt box for every detection box, eliminate those joints not matched with gt.
    This serves to reduce False positives.
    This function does not reduce prediction boxes
    :param gt_annorects:
    :param pred_annorects:
    :param _nise_cfg:
    :return:
    '''
    num_pred = len(pred_annorects)
    num_gt = len(gt_annorects)
    
    pred_joints, pred_joints_scores = get_joints_from_annorects(pred_annorects)
    
    if not num_gt or not num_pred:
        return pred_joints, pred_joints_scores
    
    pck, match = get_pck_from_anno(gt_annorects, pred_annorects)
    idx = np.argmax(pck, axis = 1)
    val = np.max(pck, axis = 1)
    
    for idPr in range(num_pred):
        idGT = idx[idPr]
        valGT = val[idPr]
        # print(idGT, valGT)
        if valGT != 0:
            corres = torch.from_numpy(np.array(
                match[idPr, idGT], dtype = np.uint8)) \
                .float()
            pred_joints[idPr, :, :] *= corres.unsqueeze(1)
            pred_joints_scores[idPr, :] *= corres
        else:
            pred_joints[idPr, :, :] = 0
            pred_joints_scores[idPr, :] = 0
    
    return pred_joints, pred_joints_scores


def find_gt_for_det_and_assignID(gt_annorects, pred_annorects, _nise_cfg):
    '''
    find gt box for every detection box, eliminate those joints not matched with gt.
    This serves to reduce False positives.
    This function does not reduce prediction boxes
    :param gt_annorects:
    :param pred_annorects:
    :param _nise_cfg:
    :return:
    '''
    num_pred = len(pred_annorects)
    num_gt = len(gt_annorects)
    
    if not num_gt or not num_pred:
        return torch.tensor([])
    
    pck, match = get_pck_from_anno(gt_annorects, pred_annorects)
    idx = np.argmax(pck, axis = 1)
    val = np.max(pck, axis = 1)
    pred_id = np.zeros(num_pred)
    gt_id = [t['track_id'][0] for t in gt_annorects]
    for idPr in range(num_pred):
        idGT = idx[idPr]
        valGT = val[idPr]
        # print(idGT, valGT)
        if valGT != 0 \
                and ((_nise_cfg.DEBUG.USE_HIGH_PCKH_DET_BOX and
                      pck[idPr, idGT] >= _nise_cfg.DEBUG.HIGH_PCKH_THRES) \
                     or not _nise_cfg.DEBUG.USE_HIGH_PCKH_DET_BOX):
            pred_id[idPr] = gt_id[idGT]
        else:
            pred_id[idPr] = -1
    
    return pred_id


# ─── PARAMETER TUNE ─────────────────────────────────────────────────────────────

def create_yaml_nms(series, thres, original_yaml = 'exp_config/t-flow-debug.yaml'):
    with open(original_yaml, 'r')as f:
        c = yaml.load(f)
    training_start_time = time.strftime("%m_%d-%H_%M", time.localtime())
    
    nc = copy.deepcopy(c)
    nc['TEST']['FROM'] = series[0]
    nc['TEST']['TO'] = series[1]
    nc['TEST']['ONLY_TEST'] = []
    nc['ALG']['UNIFY_NMS_THRES_1'] = thres[0]
    nc['ALG']['UNIFY_NMS_THRES_2'] = thres[1]
    long_file_name = 'exp_config/%s-batch-%02d_%02d-nmsthres-%.2f,%.2f.yaml' % (
        training_start_time, series[0], series[-1], thres[0], thres[1])
    with open(long_file_name, 'w')as f:
        yaml.dump(nc, f)
    return long_file_name


def create_yaml_track_filter(series, thres, original_yaml = 'exp_config/t-flow-debug.yaml'):
    with open(original_yaml, 'r')as f:
        c = yaml.load(f)
    training_start_time = time.strftime("%m_%d-%H_%M", time.localtime())
    
    nc = copy.deepcopy(c)
    nc['TEST']['FROM'] = series[0]
    nc['TEST']['TO'] = series[1]
    nc['TEST']['ONLY_TEST'] = []
    nc['ALG']['ASSIGN_BOX_THRES'] = thres[0]
    nc['ALG']['OUTPUT_JOINT_THRES'] = thres[1]
    long_file_name = 'exp_config/commi/%s-batch-%02d_%02d-box_joint_thres-%.2f,%.2f.yaml' % (
        training_start_time, series[0], series[-1], thres[0], thres[1])
    with open(long_file_name, 'w')as f:
        yaml.dump(nc, f)
    return long_file_name


def create_yaml_matchedDet__pckh_filter(series, thres, original_yaml = 'exp_config/t-flow-debug.yaml'):
    with open(original_yaml, 'r')as f:
        c = yaml.load(f)
    training_start_time = time.strftime("%m_%d-%H_%M", time.localtime())
    
    nc = copy.deepcopy(c)
    nc['TEST']['FROM'] = series[0]
    nc['TEST']['TO'] = series[1]
    nc['TEST']['ONLY_TEST'] = []
    nc['DEBUG']['HIGH_PCKH_THRES'] = thres
    long_file_name = 'exp_config/commi/%s-batch-%02d_%02d-hi_pckh_thres-%.2f.yaml' % (
        training_start_time, series[0], series[-1], thres)
    with open(long_file_name, 'w')as f:
        yaml.dump(nc, f)
    return long_file_name


def is_skip_video(nise_cfg, i, file_name):
    s = True
    for fn in nise_cfg.TEST.ONLY_TEST:
        if fn in file_name:  # priority
            s = False
            break
    if nise_cfg.TEST.ONLY_TEST:
        return s
    if s == True:
        if i >= nise_cfg.TEST.FROM and i < nise_cfg.TEST.TO:
            s = False
    return s


# ─── EVALUATION ──────────────────────────────────────────────────────────
def voc_eval_single_img(gt_boxes, pred_boxes, iou_thres = nise_cfg.TEST.MAP_TP_IOU_THRES):
    '''
    
    :param gt_boxes: Tensor, size of num_people x 5
    :param pred_boxes: Tensor, size of num_people x 5
    :return: binary vector, indicating if pred boxes are tp or not
    '''
    bin_vec = torch.zeros(pred_boxes.shape[0])
    if bin_vec.numel() == 0:  # no predictions in this img
        return bin_vec
    pred_box_np = pred_boxes.numpy()[:, :4]
    gt_box_np = gt_boxes.numpy()[:, :4]
    pred_to_gt_iou = tf_iou(pred_box_np, gt_box_np, )
    inds = get_matching_indices((pred_to_gt_iou))
    for prev, gt in inds:
        overlap = pred_to_gt_iou[prev, gt]
        if overlap >= iou_thres:
            bin_vec[prev] = 1
    return bin_vec


# @log_time('Loading gt and predictions ... ')
def eval_load_gt_and_pred_boxes(anno_file_names, pred_anno_dir = None):
    npos = 0
    gt_boxes_list = []
    pred_boxes_list = []
    bin_vec_list = []
    if pred_anno_dir is None:
        pred_anno_dir = nise_cfg.PATH.UNIFIED_JSON_DIR
    debug_print('Evaluating', pred_anno_dir)
    for i, file_name in enumerate(anno_file_names[:]):
        debug_print(i, file_name)
        
        with open(file_name, 'r') as f:
            gt = json.load(f)['annolist']
        p = PurePosixPath(file_name)
        uni_path = os.path.join(pred_anno_dir, p.stem + '.pkl')
        pred_anno = torch.load(uni_path)
        start = 0
        end = 50
        for j, frame in enumerate(gt[start:]):
            j += start
            img_file_path = frame['image'][0]['name']
            img_file_path = os.path.join(nise_cfg.PATH.POSETRACK_ROOT, img_file_path)
            # debug_print(j, img_file_path, indent = 1)
            annorects = frame['annorect']
            if (annorects is not None and len(annorects) != 0):
                gt_joints, gt_scores = get_joints_from_annorects(annorects)
            else:
                # dont eval
                continue
            gt_boxes = joints_to_boxes(gt_joints[:, :, :2], gt_joints[:, :, 2])
            pred_boxes = pred_anno[j][img_file_path]
            gt_boxes, _ = filter_bbox_with_area(gt_boxes)
            pred_boxes, _ = filter_bbox_with_area(pred_boxes)
            bin_vec = voc_eval_single_img(gt_boxes, pred_boxes)
            gt_boxes_list.append(gt_boxes)
            pred_boxes_list.append(pred_boxes)
            bin_vec_list.append(bin_vec)
            npos += gt_boxes.shape[0]
    total_pred_boxes = torch.cat(pred_boxes_list, 0)
    total_pred_boxes_scores = total_pred_boxes[:, 4]
    total_bin_vec = torch.cat(bin_vec_list)
    return total_pred_boxes_scores, total_bin_vec, npos


def voc_ap_for_pt(rec, prec):
    '''
    
    :param rec, torch vector
    :param prec, torch vector
    :return: ap, torch scalar?
    '''
    
    mrec = np.concatenate(([0.], rec.numpy(), [1.]))
    mpre = np.concatenate(([0.], prec.numpy(), [0.]))
    
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return torch.tensor(ap)


def voc_eval_for_pt(gt_anno_dir, pred_anno_dir = None):
    '''
    
    :param gt_anno_dir:
    :return:
    '''
    anno_file_names = get_type_from_dir(gt_anno_dir, ['.json'])
    anno_file_names = sorted(anno_file_names)
    total_pred_boxes_scores, total_bin_vec, npos = eval_load_gt_and_pred_boxes(anno_file_names, pred_anno_dir)
    sorted_scores, sorted_idx = torch.sort(total_pred_boxes_scores, descending = True)
    tp = total_bin_vec[sorted_idx]
    fp = 1 - tp
    tp = tp.cumsum(dim = 0)
    fp = fp.cumsum(dim = 0)
    rec = tp / float(npos)
    prec = tp / torch.max(tp + fp, torch.tensor(np.finfo(float).eps))
    ap = voc_ap_for_pt(rec, prec)
    return rec, prec, ap, sorted_scores, npos


if __name__ == '__main__':
    # # test oks distance
    # num_person = 1
    # # h, w = 576, 1024
    # # person = gen_rand_joints(num_person, h, w)
    # # threesome = torch.cat(
    # #     [person + torch.rand(num_person, 16, 2), gen_rand_joints(1, h, w)])
    # # dist = get_joints_oks_mtx(person, threesome)
    # # print(dist)
    # top_boxes = np.ones([num_person, 4])
    # top_boxes[0, 2] -= .5
    # all_boxes = np.ones([num_person + 1, 4])
    # all_boxes[0] += 0.3
    # all_boxes[1] += 0.5
    # print(top_boxes)
    # print(all_boxes)
    # # iou's input is [x,y,w,h]
    # top_to_all_overlaps = tf_iou(top_boxes, all_boxes, np.zeros(1))
    # print(top_to_all_overlaps)
    pass
