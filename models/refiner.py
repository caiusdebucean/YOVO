# -*- coding: utf-8 -*-

import torch

from echoAI.Activation.Torch.mish import Mish
from dropblock import DropBlock3D

class Refiner(torch.nn.Module):
    def __init__(self, cfg):
        super(Refiner, self).__init__()
        self.cfg = cfg
        self.dropblock = cfg.NETWORK.USE_DROPBLOCK
        #If use dropblock regularization, get drop value from cfg, else 0
        if self.dropblock == True:
            drop_prob = cfg.NETWORK.DROPBLOCK_VALUE_3D
        else:
            drop_prob = 0

        # Activation choice - default = 3 x leaky relu, 4 x relu + sigmoid
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

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            #DropBlock3D(block_size=3, drop_prob=drop_prob),
            torch.nn.BatchNorm3d(32),
            activation_A,
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            #DropBlock3D(block_size=3, drop_prob=drop_prob),
            torch.nn.BatchNorm3d(64),
            activation_A,
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            activation_A,
            torch.nn.MaxPool3d(kernel_size=2)
        )

        if cfg.NETWORK.EXTEND_REFINER:
            if cfg.NETWORK.REFINER_VERSION == 1:
                self.layer4 = torch.nn.Sequential(
                    torch.nn.Linear(8192, 2048),
                    torch.nn.Dropout(p=0.1, inplace=False),
                    activation_A
                    )
                self.layer4_5 = torch.nn.Sequential(
                    torch.nn.Linear(2048, 2048),
                    activation_A
                )
                self.layer5 = torch.nn.Sequential(
                    torch.nn.Linear(2048, 8192),
                    torch.nn.Dropout(p=0.1, inplace=False),
                    activation_A
                    )
            elif cfg.NETWORK.REFINER_VERSION == 2:
                self.layer4 = torch.nn.Sequential(
                    torch.nn.Linear(8192, 4096),
                    torch.nn.Dropout(p=0.1, inplace=False),
                    activation_A
                    )
                self.layer4_51 = torch.nn.Sequential(
                    torch.nn.Linear(4096, 2048),
                    torch.nn.Dropout(p=0.2, inplace=False),
                    activation_A
                )
                self.layer4_52 = torch.nn.Sequential(
                    torch.nn.Linear(2048, 4096),
                    torch.nn.Dropout(p=0.2, inplace=False),
                    activation_A
                )
                self.layer5 = torch.nn.Sequential(
                    torch.nn.Linear(4096, 8192),
                    torch.nn.Dropout(p=0.1, inplace=False),
                    activation_A
                    )
            else:
                print("Refiner Version not implemented. Please choose between Versions: [1 , 2]")
                exit()
        else:
            self.layer4 = torch.nn.Sequential(
                torch.nn.Linear(8192, 2048),
                torch.nn.Dropout(p=0.1, inplace=False),
                activation_A
                )        
            self.layer5 = torch.nn.Sequential(
                torch.nn.Linear(2048, 8192),
                torch.nn.Dropout(p=0.1, inplace=False),
                activation_A
                )

        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            #DropBlock3D(block_size=3, drop_prob=drop_prob),
            torch.nn.BatchNorm3d(64),
            activation_A
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            #DropBlock3D(block_size=2, drop_prob=drop_prob),
            torch.nn.BatchNorm3d(32),
            activation_A
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, coarse_volumes):
        cfg = self.cfg
        volumes_32_l = coarse_volumes.view((-1, 1, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX))
        # print(volumes_32_l.size())       # torch.Size([batch_size, 1, 32, 32, 32])
        volumes_16_l = self.layer1(volumes_32_l)
        # print(volumes_16_l.size())       # torch.Size([batch_size, 32, 16, 16, 16])
        volumes_8_l = self.layer2(volumes_16_l)
        # print(volumes_8_l.size())        # torch.Size([batch_size, 64, 8, 8, 8])
        volumes_4_l = self.layer3(volumes_8_l)
        # print(volumes_4_l.size())        # torch.Size([batch_size, 128, 4, 4, 4])
        flatten_features = self.layer4(volumes_4_l.view(-1, 8192))
        # print(flatten_features.size())   # torch.Size([batch_size, 2048])

        if cfg.NETWORK.EXTEND_REFINER:
            if cfg.NETWORK.REFINER_VERSION == 1:
                flatten_features = self.layer4_5(flatten_features)
            elif cfg.NETWORK.REFINER_VERSION == 2:
                flatten_features = self.layer4_51(flatten_features)
                flatten_features = self.layer4_52(flatten_features)

        flatten_features = self.layer5(flatten_features)
        # print(flatten_features.size())   # torch.Size([batch_size, 8192])
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 128, 4, 4, 4)
        # print(volumes_4_r.size())        # torch.Size([batch_size, 128, 4, 4, 4])
        volumes_8_r = volumes_8_l + self.layer6(volumes_4_r)
        # print(volumes_8_r.size())        # torch.Size([batch_size, 64, 8, 8, 8])
        volumes_16_r = volumes_16_l + self.layer7(volumes_8_r)
        # print(volumes_16_r.size())       # torch.Size([batch_size, 32, 16, 16, 16])
        volumes_32_r = (volumes_32_l + self.layer8(volumes_16_r)) * 0.5
        # print(volumes_32_r.size())       # torch.Size([batch_size, 1, 32, 32, 32])

        return volumes_32_r.view((-1, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX, self.cfg.CONST.N_VOX))

