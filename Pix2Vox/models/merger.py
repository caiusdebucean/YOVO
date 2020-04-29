# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch

from echoAI.Activation.Torch.mish import Mish

class Merger(torch.nn.Module):
    def __init__(self, cfg):
        super(Merger, self).__init__()
        self.cfg = cfg

        # Activation choice - default = 5 x leaky relu + softmax
        if cfg.NETWORK.ALTERNATIVE_ACTIVATION_A == 'relu':
            activation_A = torch.nn.ReLU()
        elif cfg.NETWORK.ALTERNATIVE_ACTIVATION_A == 'elu':
            activation_A = torch.nn.ELU()
        elif cfg.NETWORK.ALTERNATIVE_ACTIVATION_A == 'leaky relu':
            activation_A = torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        elif cfg.NETWORK.ALTERNATIVE_ACTIVATION_A == 'mish':
            activation_A = Mish()

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(16),
            activation_A
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 8, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(8),
            activation_A
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 4, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(4),
            activation_A
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(4, 2, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(2),
            activation_A
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(2, 1, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(1),
            activation_A
        )

    def forward(self, raw_features, coarse_volumes):
        n_views_rendering = coarse_volumes.size(1)
        raw_features = torch.split(raw_features, 1, dim=1)
        volume_weights = []

        for i in range(n_views_rendering):
            raw_feature = torch.squeeze(raw_features[i], dim=1)
            # print(raw_feature.size())       # torch.Size([batch_size, 9, 32, 32, 32])

            volume_weight = self.layer1(raw_feature)
            # print(volume_weight.size())     # torch.Size([batch_size, 16, 32, 32, 32])
            volume_weight = self.layer2(volume_weight)
            # print(volume_weight.size())     # torch.Size([batch_size, 8, 32, 32, 32])
            volume_weight = self.layer3(volume_weight)
            # print(volume_weight.size())     # torch.Size([batch_size, 4, 32, 32, 32])
            volume_weight = self.layer4(volume_weight)
            # print(volume_weight.size())     # torch.Size([batch_size, 2, 32, 32, 32])
            volume_weight = self.layer5(volume_weight)
            # print(volume_weight.size())     # torch.Size([batch_size, 1, 32, 32, 32])

            volume_weight = torch.squeeze(volume_weight, dim=1)
            # print(volume_weight.size())     # torch.Size([batch_size, 32, 32, 32])
            volume_weights.append(volume_weight)

        volume_weights = torch.stack(volume_weights).permute(1, 0, 2, 3, 4).contiguous()
        volume_weights = torch.softmax(volume_weights, dim=1)
        # print(volume_weights.size())        # torch.Size([batch_size, n_views, 32, 32, 32])
        # print(coarse_volumes.size())        # torch.Size([batch_size, n_views, 32, 32, 32])
        coarse_volumes = coarse_volumes * volume_weights
        coarse_volumes = torch.sum(coarse_volumes, dim=1)

        return torch.clamp(coarse_volumes, min=0, max=1)
