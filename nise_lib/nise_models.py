import json
import os
import pathlib
import torch
import sys
import time
import numpy as np
from pathlib import PurePosixPath

# local packages
from nise_utils.imutils import *
from plogs.logutils import Levels


def make_conv_layer():
    pass


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
    
    def __init__(self, channels):
        super(MatchingNet, self).__init__()
        self.conv1 = Conv3x3(channels, channels)  # out 48x48
        self.pool1 = nn.MaxPool2d(3)  # out 16x16
        self.conv2 = Conv3x3(channels, channels)  # out 8x8
        self.conv3 = Conv3x3(channels, channels)  # out 4x4
        self.conv4 = Conv3x3(channels, channels)  # out 2x2
        self.linear = nn.Linear(in_features = channels * 4, out_features = 1)
    
    # @log_time("Forwarding...")
    def forward(self, x):
        out = self.pool1(self.conv1(x))
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        batch_size = out.shape[0]
        out = self.linear(out.view(batch_size, -1))
        return out


if __name__ == '__main__':
    rand_mat = torch.rand(100, 542, 96, 96)
    dnet = MatchingNet(542)
    o = dnet(rand_mat)
    print(o.shape)
