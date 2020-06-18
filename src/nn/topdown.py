import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNetLayer

class TopdownNetwork(nn.Sequential):

    def __init__(self, in_channels, channels, layers=[6, 1, 1], 
                 strides=[1, 2, 2], blocktype='basic'):
        
        modules = list()
        self.downsample = 1
        for nblocks, stride in zip(layers, strides):

            # Add a new residual layer
            module = ResNetLayer(
                in_channels, channels, nblocks, 1/stride, blocktype=blocktype)
            modules.append(module)

            # Halve the number of channels at each layer
            in_channels = module.out_channels
            channels = channels // 2
            self.downsample *= stride
        
        self.out_channels = in_channels

        
        super().__init__(*modules)