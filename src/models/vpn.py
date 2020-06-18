"""
This implementation of the model from the paper "Cross-view Semantic 
Segmentation for Sensing Surroundings" is directly adapted from the code
provided by the original authors at
https://github.com/pbw-Berwin/View-Parsing-Network (accessed 08/06/2020)

"""


# File   : models.py
# Author : Bowen Pan
# Email  : panbowen0607@gmail.com
# Date   : 09/18/2018
#
# Distributed under terms of the MIT license.
import os
import sys
import math
import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F

import numpy as np
from itertools import combinations

from torchvision.models import resnet

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


class TransformModule(nn.Module):
    def __init__(self, dim=(37, 60), num_view=8):
        super(TransformModule, self).__init__()
        self.num_view = num_view
        self.dim = dim
        self.mat_list = nn.ModuleList()

        # MODIFIED: dims need not be square
        for i in range(self.num_view):
            fc_transform = nn.Sequential(
                        nn.Linear(dim[0] * dim[1], dim[0] * dim[1]),
                        nn.ReLU(),
                        nn.Linear(dim[0] * dim[1], dim[0] * dim[1]),
                        nn.ReLU()
                    )
            self.mat_list += [fc_transform]

    def forward(self, x):
        # shape x: B, V, C, H, W
        x = x.view(list(x.size()[:3]) + [self.dim[0] * self.dim[1],])
        view_comb = self.mat_list[0](x[:, 0])
        for index in range(x.size(1))[1:]:
            view_comb += self.mat_list[index](x[:, index])
        view_comb = view_comb.view(list(view_comb.size()[:2]) + list(self.dim))
        return view_comb


class SumModule(nn.Module):
    def __init__(self):
        super(SumModule, self).__init__()

    def forward(self, x):
        # shape x: B, V, C, H, W
        x = torch.sum(x, dim=1, keepdim=False)
        return x


class VPNModel(nn.Module):
    def __init__(self, num_views, num_class, output_size, fc_dim, map_extents, 
                 map_resolution):

        super(VPNModel, self).__init__()
        self.num_views = num_views
        self.output_size = output_size

        self.seg_size = (
            int((map_extents[3] - map_extents[1]) / map_resolution),
            int((map_extents[2] - map_extents[0]) / map_resolution),
        )

        # MODIFIED: we fix the transform module, the encoder and decoder to be 
        # the ones described in the paper 
        self.encoder = resnet18(True)
        self.transform_module = TransformModule(dim=self.output_size, 
                                                num_view=self.num_views)
        self.decoder = PPMBilinear(num_class, fc_dim, False)

    def forward(self, x, *args):
        B, N, C, H, W = x.view([-1, self.num_views, int(x.size()[1] / self.num_views)] \
                            + list(x.size()[2:])).size()

        x = x.view(B*N, C, H, W)
        x = self.encoder(x)[0]
        x = x.view([B, N] + list(x.size()[1:]))
        x = self.transform_module(x)
        x = self.decoder([x], self.seg_size)
        
        return x



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_feature_maps=False):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        conv_out = []
        x = self.layer1(x); conv_out.append(x)
        x = self.layer2(x); conv_out.append(x)
        x = self.layer3(x); conv_out.append(x)
        # x = self.layer4(x)

        if return_feature_maps:
            return conv_out
        return [x]


def resnet18(pretrained=False, **kwargs):
    model = ResNet(resnet.BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        weights = load_url('http://sceneparsing.csail.mit.edu/model/'\
                           'pretrained_resnet/resnet18-imagenet.pth')
        state_dict = model.state_dict()
        for key, weight in state_dict.items():
            weight.copy_(weights[key])
        model.load_state_dict(state_dict)
    return model



def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)



# pyramid pooling, bilinear upsample
class PPMBilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinear, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None, return_feat=False):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)
        feat = x
        if segSize is not None:
            x = nn.functional.upsample(
                x, size=segSize, mode='bilinear', align_corners=False)
        if self.use_softmax:  # is True during inference
            x = nn.functional.softmax(x, dim=1)
        
        # MODIFIED: we use BCE loss, so do not convert to log-softmax
        if return_feat:
            return x, feat
        return x