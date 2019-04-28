import json
import os
import pathlib
import torch
import sys
import time
import numpy as np
from pathlib import PurePosixPath

# local packages
from nise_lib.nise_config import nise_cfg
from nise_lib.nise_functions import *
from nise_utils.imutils import *
from plogs.logutils import Levels
import tron_lib.nn as mynn
from mem_util.gpu_mem_track import MemTracker
import inspect

from tron_lib.modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
from tron_lib.model.roi_pooling.functions.roi_pool import RoIPoolFunction


# DEBUGGING


def conv3x3(in_planes, out_planes, stride = 2):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)


class Conv3x3(nn.Module):
    
    def __init__(self, inplanes, planes, stride = 2):
        super(Conv3x3, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum = 0.1)
        self.relu = nn.ReLU(inplace = True)
        self.stride = stride
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        return out


class MatchingNet(nn.Module):
    
    def __init__(self, channels, maskRCNN):
        super(MatchingNet, self).__init__()
        if isinstance(maskRCNN, mynn.DataParallel):
            maskRCNN = list(maskRCNN.children())[0]
        else:
            maskRCNN = maskRCNN
        self.conv_body = maskRCNN.Conv_Body
        model_children = list(self.conv_body.children())
        for m in model_children:
            for param in m.parameters():
                param.requires_grad = False
        self.conv_body.P2only = True
        self.conv1 = Conv3x3(channels, channels)  # out 48x48
        self.pool1 = nn.MaxPool2d(3)  # out 16x16
        self.conv2 = Conv3x3(channels, channels)  # out 8x8
        self.conv3 = Conv3x3(channels, channels)  # out 4x4
        self.conv4 = Conv3x3(channels, channels)  # out 2x2
        self.linear = nn.Linear(in_features = channels * 4, out_features = 1)
        self.gpuMemTracker = MemTracker(inspect.currentframe())
    
    # @log_time("Forwarding...")
    def forward(self, fmaps, scales, all_samples, joints_heatmap, idx):
        # self.gpuMemTracker.track()
        bs, num_samples, _, H, W = fmaps.shape
        inputs = []
        for b in range(bs):
            p_fmap = fmaps[b, :1]  # make it batch
            c_fmap = fmaps[b, 1:]
            boxes = expand_vector_to_tensor(all_samples[b, idx[b], 2:])  # should get rid of -1s
            if boxes.numel() == 0:
                continue
            p_box_fmap = self.get_box_fmap({'fmap': p_fmap, 'scale': scales[b]}, boxes, 'align')
            c_box_fmap = self.get_box_fmap({'fmap': c_fmap, 'scale': scales[b]}, boxes, 'align')
            # num_samples x 256 x 96 x 96
            inputs.append(torch.cat([p_box_fmap, c_box_fmap, joints_heatmap[b, idx[b]]], 1))
        x = torch.cat(inputs)
        out = self.original_forward(x)
        return out
    
    def original_forward(self, x):
        out = self.pool1(self.conv1(x))
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        batch_size = out.shape[0]
        out = self.linear(out.view(batch_size, -1))
        return out
    
    def get_box_fmap(self, fmap_info: dict, boxes, method = 'pool'):
        '''
        :param boxes: torch tensor, bs x 5
        :return:
        '''
        
        fmap, scale = fmap_info['fmap'], fmap_info['scale'][0]  # scale is [x,y]'s, but they are the same
        assert fmap.is_cuda
        rois = torch.zeros([boxes.shape[0], 5])
        rois[:, 1:] = boxes[:, 0:4] * scale
        map_res = nise_cfg.MODEL.FEATURE_MAP_RESOLUTION
        
        boxes_fmap = RoIAlignFunction(map_res, map_res, .25, 2)(fmap.cuda(), rois.cuda())
        
        return boxes_fmap


def load_mNet_model(model_file, maskRCNN):
    model = MatchingNet(nise_cfg.MODEL.INPUTS_CHANNELS, maskRCNN)
    gpus = list(range(len(os.environ.get('CUDA_VISIBLE_DEVICES', default = '0').split(','))))
    debug_print(gpus)
    model = torch.nn.DataParallel(model, device_ids = gpus).cuda()
    meta_info = torch.load(model_file)
    model.load_state_dict(meta_info['state_dict'])
    return model


if __name__ == '__main__':
    rand_mat = torch.rand(100, 542, 96, 96)
    dnet = MatchingNet(542, None)
    o = dnet(rand_mat)
    print(o.shape)
