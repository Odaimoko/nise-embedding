from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

import torch.nn as nn
import torchvision
# import resnet
# from pose import Bar
# from pose.utils.logger import Logger, savefig
# from pose.utils.evaluation import accuracy, AverageMeter, final_preds
# from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
# from pose.utils.osutils import mkdir_p, isfile, isdir, join
# from pose.utils.imutils import batch_with_heatmap
# from pose.utils.transforms import fliplr, flip_back
# import pose.models as models
# import pose.datasets as datasets

import torch.optim.lr_scheduler as scheduler
import numpy as np
import visdom
from ae_lib.models.posenet import PoseNet
from dataloader.mscocoMulti import MscocoMulti
from ae_config import cfg, get_arg_parser

# try AE on coco2017

net_cfg = {'nstack': 2, 'inp_dim': 256, 'oup_dim': 34,
           'num_parts': 17, 'increase': 128, 'keys': ['imgs']}

net = PoseNet(**net_cfg).cuda()
args = get_arg_parser().parse_args()
train_loader = torch.utils.data.DataLoader(
  MscocoMulti(cfg),
  batch_size = cfg.batch_size * args.num_gpus, shuffle = True,
  num_workers = args.workers, pin_memory = True)


def train(epoch, data_loader, model, optimizer):
  pass


heatmap_loss_func = torch.nn.MSELoss().cuda()

for i, (inputs, targets, valid, meta) in enumerate(train_loader):
  '''
    这个loader的特殊之处在于，进来的每一个inputs都是一个人（obj），所以所有关节的id已经统一了。
    但是posetrack就不一样
  '''
  if i > 3:
    break
  to_net = inputs.permute(0, 2, 3, 1).cuda()
  # targets = targets.cuda()
  print(to_net.shape)
  result = net(to_net)  # torch.Size([bs, nStack, 68, 64, 48])
  # labels = {
  #   'keypoints': torch.rand([1, 30, 17, 2]),
  #   'heatmaps': torch.rand([1, 17, 128, 128]),
  #   'masks': torch.rand([1, 128, 128])
  # }
  # push_loss, pull_loss, joint_loss = net.calc_loss(result,**labels )
  
  heatmap_losses = []
  hourglass_final_layer_id = net_cfg['nstack'] - 1
  num_joints = net_cfg['num_parts']
  heatmap_joints = result[:, hourglass_final_layer_id, 0:num_joints]
  tags = result[:, hourglass_final_layer_id, num_joints:]
  
  for gt_heatmap in targets:
    gt_heatmap=gt_heatmap.cuda()
    heatmap_losses.append(heatmap_loss_func(heatmap_joints,gt_heatmap))
  
  print(i, inputs.shape, valid.shape)
