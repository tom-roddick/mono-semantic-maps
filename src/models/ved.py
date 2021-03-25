"""
This implementation of the model from the paper "Monocular Semantic Occupancy 
Grid Mapping with Convolutional Variational Encoder-Decoder Networks" is 
directly adapted from the code provided by the original authors at 
https://github.com/Chenyang-Lu/mono-semantic-occupancy (accessed 08/06/2020).

Modifications to the original code are identified in comments.

MIT License

Copyright (c) 2019 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ..nn import losses


class VariationalEncoderDecoder(nn.Module):

    def __init__(self, num_class, bottleneck_dim, map_extents, map_resolution):
        
        super().__init__()
        self.model = VaeMapping(num_class, bottleneck_dim)
        self.output_size = (
            int((map_extents[3] - map_extents[1]) / map_resolution),
            int((map_extents[2] - map_extents[0]) / map_resolution),
        )

    
    def forward(self, image, *args):

        # Downsample input image so that it more closely matches
        # the input dimensions used in the original paper
        image = image[:, :, ::2, ::2]

        # Run model forwards
        logits, mu, logvar = self.model(image, self.output_size, self.training)

        return logits, mu, logvar
    






def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class upsample(nn.Module):

    def __init__(self, if_deconv, channels=None):
        super(upsample, self).__init__()
        if if_deconv:
            self.upsample = nn.ConvTranspose2d(
                channels, channels, 4, stride=2, padding=1, bias=False)
        else:
            self.upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)

        return x


class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class encoder_after_vgg(nn.Module):

    def __init__(self, bottleneck_dim=32):
        super(encoder_after_vgg, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # MODIFIED: The original VED paper assumed fixed input dimensions of 
        # 256x512, which leads to a bottleneck dimension of 8x4. Since our
        # input size varies depending on dataset we have to specify the
        # bottleneck dimension manually. 
        self.mu_dec = nn.Linear(bottleneck_dim * 128, 512)
        self.logvar_dec = nn.Linear(bottleneck_dim * 128, 512)
        

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1, 3)
        mu = self.mu_dec(x)
        logvar = self.logvar_dec(x)

        return mu, logvar


class decoder_conv(nn.Module):
    def __init__(self, num_class, if_deconv=True):
        super(decoder_conv, self).__init__()

        self.up1 = upsample(if_deconv=if_deconv, channels=128)
        self.conv1 = double_conv(128, 256)
        self.up2 = upsample(if_deconv=if_deconv, channels=256)
        self.conv2 = double_conv(256, 256)
        self.up3 = upsample(if_deconv=if_deconv, channels=256)
        self.conv3 = double_conv(256, 256)
        self.up4 = upsample(if_deconv=if_deconv, channels=256)
        self.conv4 = double_conv(256, 256)
        self.up5 = upsample(if_deconv=if_deconv, channels=256)
        self.conv5 = double_conv(256, 256)

        # MODIFIED: Add an additional upsampling layer
        self.up6 = upsample(if_deconv=if_deconv, channels=256)
        self.conv6 = double_conv(256, 256)

        self.conv_out = nn.Conv2d(256, num_class, 3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x, output_size):
        x = x.view(-1, 128, 2, 2)
        x = self.up1(x)
        x = self.conv1(x)

        x = self.up2(x)
        x = self.conv2(x)

        x = self.up3(x)
        x = self.conv3(x)

        x = self.up4(x)
        x = self.conv4(x)

        x = self.up5(x)
        x = self.conv5(x)

        # MODIFIED: Add additional upsampling layer
        x = self.up6(x)
        x = self.conv6(x)

        # MODIFIED: Resample to match label dimensions
        x = F.upsample(x, size=output_size, mode='bilinear')

        x = self.conv_out(x)

        return x


class VaeMapping(nn.Module):

    def __init__(self, num_class, bottleneck_dim=32):
        super(VaeMapping, self).__init__()

        self.vgg16 = models.vgg16_bn(pretrained=True)
        self.vgg16_feature = nn.Sequential(*list(self.vgg16.features.children())[:])
        self.encoder_afterv_vgg = encoder_after_vgg(bottleneck_dim)
        self.decoder = decoder_conv(num_class, if_deconv=True)

    def reparameterize(self, is_training, mu, logvar):
        if is_training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, output_size, is_training=False, defined_mu=None):

        x = self.vgg16_feature(x)
        mu, logvar = self.encoder_afterv_vgg(x)
        z = self.reparameterize(is_training, mu, logvar)
        if defined_mu is not None:
            z = defined_mu
        pred_map = self.decoder(z, output_size)

        return pred_map, mu, logvar


def loss_function_map(pred_map, map, mu, logvar):

    # MODIFIED: move weights to same GPU as inputs
    CE = F.cross_entropy(pred_map, map.view(-1, 64, 64), weight=
        torch.Tensor([0.6225708,  2.53963754, 15.46416047, 0.52885405]).to(map), ignore_index=4)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return 0.9*CE + 0.1*KLD, CE, KLD

