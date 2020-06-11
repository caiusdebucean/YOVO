# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch
from echoAI.Activation.Torch.mish import Mish
from dropblock import DropBlock3D

class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.dropblock = cfg.NETWORK.USE_DROPBLOCK
        # Activation choice - default = 4 x relu, sigmoid
        if cfg.NETWORK.ALTERNATIVE_ACTIVATION_A == 'relu':
            activation_A = torch.nn.ReLU()
        elif cfg.NETWORK.ALTERNATIVE_ACTIVATION_A == 'elu':
            activation_A = torch.nn.ELU()
        elif cfg.NETWORK.ALTERNATIVE_ACTIVATION_A == 'leaky relu':
            activation_A = torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        elif cfg.NETWORK.ALTERNATIVE_ACTIVATION_A == 'mish':
            activation_A = Mish()

        if cfg.NETWORK.ALTERNATIVE_ACTIVATION_B == 'relu':
            activation_B = torch.nn.ReLU()
        elif cfg.NETWORK.ALTERNATIVE_ACTIVATION_B == 'elu':
            activation_B = torch.nn.ELU()
        elif cfg.NETWORK.ALTERNATIVE_ACTIVATION_B == 'leaky relu':
            activation_B = torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        elif cfg.NETWORK.ALTERNATIVE_ACTIVATION_B == 'mish':
            activation_B = Mish()
         
        #If use dropblock regularization, get drop value from cfg, else 0
        if self.dropblock == True:
            drop_prob = cfg.NETWORK.DROPBLOCK_VALUE_3D
        else:
            drop_prob = 0
        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(2048, 512, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(512),
            activation_A
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            activation_A
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            activation_A
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            DropBlock3D(block_size=1, drop_prob=drop_prob),
            torch.nn.BatchNorm3d(8),
            activation_A
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )

    def forward(self, image_features):
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        image_features = torch.split(image_features, 1, dim=0)
        gen_volumes = []
        raw_features = []

        for features in image_features:
            gen_volume = features.view(-1, 2048, 2, 2, 2)
            # print(gen_volume.size())   # torch.Size([batch_size, 2048, 2, 2, 2])
            gen_volume = self.layer1(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 512, 4, 4, 4])
            gen_volume = self.layer2(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 128, 8, 8, 8])
            gen_volume = self.layer3(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 32, 16, 16, 16])
            gen_volume = self.layer4(gen_volume)
            raw_feature = gen_volume
            # print(gen_volume.size())   # torch.Size([batch_size, 8, 32, 32, 32])
            gen_volume = self.layer5(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])
            raw_feature = torch.cat((raw_feature, gen_volume), dim=1)
            # print(raw_feature.size())  # torch.Size([batch_size, 9, 32, 32, 32])

            gen_volumes.append(torch.squeeze(gen_volume, dim=1))
            raw_features.append(raw_feature)

        gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        # print(gen_volumes.size())      # torch.Size([batch_size, n_views, 32, 32, 32])
        # print(raw_features.size())     # torch.Size([batch_size, n_views, 9, 32, 32, 32])
        return raw_features, gen_volumes
