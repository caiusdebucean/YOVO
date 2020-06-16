# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models
import torch.utils.model_zoo as model_zoo
from echoAI.Activation.Torch.mish import Mish
from collections import OrderedDict

from dropblock import DropBlock2D

model_paths = {
    'darknet19': 'https://s3.ap-northeast-2.amazonaws.com/deepbaksuvision/darknet19-deepBakSu-e1b3ec1e.pth'
}

class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Activation choice - default = 3 x elu
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

        self.architecture = cfg.NETWORK.ENCODER
        self.multi_level = cfg.NETWORK.MULTI_LEVEL_TRAIN
        self.dropblock = cfg.NETWORK.USE_DROPBLOCK 
        # Layer Definition

        #If use dropblock regularization, get drop value from cfg, else 0
        if self.dropblock == True:
            drop_prob = cfg.NETWORK.DROPBLOCK_VALUE_2D
        else:
            drop_prob = 0

        if self.multi_level == True:
            self.multi_conv6_1 = torch.nn.Sequential(
                torch.nn.Conv2d(96, 192, kernel_size=3),
                DropBlock2D(block_size=1, drop_prob=drop_prob),
                torch.nn.BatchNorm2d(192),
                activation_B
            )
            self.multi_conv6_2 = torch.nn.Sequential(
                torch.nn.Conv2d(192, 384, kernel_size=3),
                DropBlock2D(block_size=2, drop_prob=drop_prob),
                torch.nn.BatchNorm2d(384),
                activation_B
            )
            self.multi_conv6_3 = torch.nn.Sequential(
                torch.nn.Conv2d(384, 256, kernel_size=3),
                torch.nn.BatchNorm2d(256),
                activation_A
            )

            self.multi_conv4_1 = torch.nn.Sequential(
                torch.nn.Conv2d(128, 320, kernel_size=3),
                DropBlock2D(block_size=1, drop_prob=drop_prob),
                torch.nn.BatchNorm2d(320),
                activation_B
            )
            self.multi_conv4_2 = torch.nn.Sequential(
                torch.nn.Conv2d(320, 256, kernel_size=3),
                torch.nn.BatchNorm2d(256),
                activation_A
            )
            # self.multi_conv3 = torch.nn.Sequential(
            #     torch.nn.Conv2d(128, 256, kernel_size=3),
            #     torch.nn.BatchNorm2d(128),
            #     activation_A
            # )


        if self.architecture == 'original':
            vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
            self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(512, 512, kernel_size=3),
                torch.nn.BatchNorm2d(512),
                activation_A
            )
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(512, 512, kernel_size=3),
                torch.nn.BatchNorm2d(512),
                activation_A,
                torch.nn.MaxPool2d(kernel_size=3)
            )
            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(512, 256, kernel_size=1),
                torch.nn.BatchNorm2d(256),
                activation_A
            )

            # Don't update params in VGG16
            for param in vgg16_bn.parameters():
                param.requires_grad = False
        elif self.architecture == 'darknet19': #TODO: Implement a pretrained  darknet architecture
            #darknet19 = model_zoo.load_url(model_paths['darknet19'],  progress=True)
            exit()
            print('Darknet model is loaded')
        elif self.architecture == 'mobilenet':
            mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=True)
            self.mobilenet = torch.nn.Sequential(*list(mobilenet_v2.features.children()))[:14]
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(96, 128, kernel_size=3),
                DropBlock2D(block_size=1, drop_prob=drop_prob),
                torch.nn.BatchNorm2d(128),
                activation_B
            )
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(128, 128, kernel_size=3),
                DropBlock2D(block_size=2, drop_prob=drop_prob),
                torch.nn.BatchNorm2d(128),
                activation_B
            )
            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, kernel_size=3),
                torch.nn.BatchNorm2d(256),
                activation_A
            )
            print(self.mobilenet)
            # Don't update params in MobileNetV2
            # for param in mobilenet_v2.parameters():
            #     param.requires_grad = False
            print('MobileNet_V2 model is loaded')
    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []



        if self.architecture == 'original':
            for img in rendering_images:
                features = self.vgg(img.squeeze(dim=0))
                # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
                features = self.layer1(features)
                # print(features.size())    # torch.Size([batch_size, 512, 26, 26])
                features = self.layer2(features)
                # print(features.size())    # torch.Size([batch_size, 512, 24, 24])
                features = self.layer3(features)
                # print(features.size())    # torch.Size([batch_size, 256, 8, 8])
                image_features.append(features)
        if self.architecture == 'darknet':
            exit()
        if self.architecture == 'mobilenet':
            for img in rendering_images:
                features = self.mobilenet(img.squeeze(dim=0))
                if self.multi_level == True:
                    new_features_6 = self.multi_conv6_1(features)
                    new_features_6 = self.multi_conv6_2(new_features_6)
                    new_features_6 = self.multi_conv6_3(new_features_6)
                    image_features.append(new_features_6)

                features = self.layer1(features)
                if self.multi_level == True:
                    new_features_4 = self.multi_conv4_1(features)
                    new_features_4 = self.multi_conv4_2(new_features_4)
                    image_features.append(new_features_4)

                features = self.layer2(features)
                features = self.layer3(features)
                image_features.append(features) #[32, 256, 8 ,8]

        image_features = torch.stack(image_features)
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 8, 8])
        # exit()
        return image_features
