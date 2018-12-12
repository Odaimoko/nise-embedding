import os
import sys
import colorama
import argparse
import pathlib
from functools import wraps
from easydict import EasyDict as edict
from nise_lib.nise_config import nise_cfg
import time

# local packages
import nise_lib._init_paths
from nise_lib.nise_debugging_func import *

import flow_models
import flow_losses
import flow_datasets
from nise_utils.imutils import *
from simple_lib.core.config import config as simple_cfg
from simple_lib.core.config import update_config
from simple_models.pose_resnet import get_pose_net
import time
from pathlib import PurePosixPath
from nise_lib.nise_config import nise_cfg

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────────

from tron_lib.core.config import cfg_from_file, cfg_from_list, assert_and_infer_cfg
from tron_lib.modeling.model_builder import Generalized_RCNN_for_posetrack
import tron_lib.nn as mynn
from tron_lib.utils.detectron_weight_helper import load_detectron_weight
import tron_lib.utils.net as net_utils
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
                            required = True,
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
    
    simple_human_det_model = get_pose_net(
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


from tron_lib.utils.logging import setup_logging

logger = setup_logging(__name__)


def save_ckpt(output_dir, args, step, train_size, model, optimizer):
    """Save checkpoint"""
    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_step{}.pth'.format(step))
    if isinstance(model, mynn.DataParallel):
        model = model.module
    model_state_dict = model.state_dict()
    torch.save({
        'step': step,
        'train_size': train_size,
        'batch_size': args.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)


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


def joints_to_bboxes(new_joints, joint_vis = None, clamp_size = ()):
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


def filter_bbox_with_scores(boxes, thres = nise_cfg.ALG._HUMAN_THRES):
    scores = boxes[:, -1]
    valid_scores_idx = torch.nonzero(scores >= thres).squeeze_().long()  # in case it's 6 x **1** x 5
    filtered_box = boxes[valid_scores_idx, :]
    return filtered_box, valid_scores_idx


def filter_bbox_with_area(boxes, thres = nise_cfg.ALG._AREA_THRES):
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    valid_area_idx = torch.nonzero(
        area >= thres).squeeze_().long()  # in case it's 6 x **1** x 5
    filtered_box = boxes[valid_area_idx, :]
    return filtered_box, valid_area_idx


def expand_vector_to_tensor(tensor, target_dim = 2):
    if tensor.numel() == 0:
        # size is 0
        return tensor
    while len(tensor.shape) < target_dim:  # vector
        tensor = tensor.unsqueeze(0)
    return tensor


def get_joints_oks_mtx(j1, j2):
    '''

    :param j1: num_people 1 x 16 x 2
    :param j2: num_people 2 x 16 x 2
    :return:
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


# ─── MISC ───────────────────────────────────────────────────────────────────────


def mkdir(path):
    path = path.strip().rstrip("\\")
    
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False


def make_nise_dirs():
    mkdir(nise_cfg.PATH.IMAGES_OUT_DIR)
    mkdir(nise_cfg.PATH.JSON_SAVE_DIR)
    mkdir(nise_cfg.PATH.JOINTS_DIR)


def get_type_from_dir(dirpath, type_list):
    files = []
    for f in os.listdir(dirpath):
        if pathlib.PurePosixPath(f).suffix.lower() in type_list:
            files.append(os.path.join(dirpath, f))
    return files


def get_joints_from_annorects(annorects):
    all_joints = []
    for i in range(len(annorects)):
        rect = annorects[i]
        
        joints_3d = np.zeros((nise_cfg.DATA.num_joints, 3), dtype = np.float)
        # joints_3d_vis = np.zeros((nise_cfg.DATA.num_joints, 1), dtype = np.float)
        points = rect['annopoints']
        # there's a person, but no annotations
        if points is None or len(points) <= 0:  # 因为有些图并没有annotation
            continue
        else:
            points = points[0]['point']
        for pt_info in points:
            # analogous to coco.py  # matlab based on 1.
            i_pt = pt_info['id'][0]
            joints_3d[i_pt, 0] = pt_info['x'][0] - 1
            joints_3d[i_pt, 1] = pt_info['y'][0] - 1
            
            t_vis = pt_info['is_visible'][0]
            if t_vis > 1: t_vis = 1
            joints_3d[i_pt, 2] = t_vis
        # head_bbox = [rect['x1'][0], rect['y1'][0], rect['x2'][0], rect['y2'][0]]
        # head_bbox = np.array(head_bbox)
        all_joints.append(joints_3d)
    
    joints = torch.tensor(all_joints).float()
    
    return joints


# DECORATORS

def log_time(*text, record = None):
    def real_deco(func):
        @wraps(func)
        def impl(*args, **kw):
            r = debug_print if not record else record  # 如果没有record，默认print
            t = (func.__name__,) if not text else text
            start = time.time()
            func(*args, **kw)
            end = time.time()
            r(*t, '%.3f s.' % (end - start,))
            # r(' Start time: %.3f s. End time: %.3f s'%(start, end))
        
        return impl
    
    return real_deco


if __name__ == '__main__':
    # test oks distance
    num_person = 2
    h, w = 576, 1024
    person = gen_rand_joints(num_person, h, w)
    threesome = torch.cat(
        [person + torch.rand(num_person, 16, 2), gen_rand_joints(1, h, w)])
    dist = get_joints_oks_mtx(person, threesome)
    print(dist)
