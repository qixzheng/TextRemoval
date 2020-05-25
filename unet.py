#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):

        super(UNet, self).__init__()

        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))

        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))

        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))

        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        self._init_weights()


    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        diffY = torch.tensor([pool4.size()[2] - upsample5.size()[2]])
        diffX = torch.tensor([pool4.size()[3] - upsample5.size()[3]])

        upsample5 = nn.functional.pad(upsample5, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        diffY = torch.tensor([pool3.size()[2] - upsample4.size()[2]])
        diffX = torch.tensor([pool3.size()[3] - upsample4.size()[3]])

        upsample4 = nn.functional.pad(upsample4, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        diffY = torch.tensor([pool2.size()[2] - upsample3.size()[2]])
        diffX = torch.tensor([pool2.size()[3] - upsample3.size()[3]])

        upsample3 = nn.functional.pad(upsample3, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        diffY = torch.tensor([pool1.size()[2] - upsample2.size()[2]])
        diffX = torch.tensor([pool1.size()[3] - upsample2.size()[3]])

        upsample2 = nn.functional.pad(upsample2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        diffY = torch.tensor([x.size()[2] - upsample1.size()[2]])
        diffX = torch.tensor([x.size()[3] - upsample1.size()[3]])

        upsample1 = nn.functional.pad(upsample1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        concat1 = torch.cat((upsample1, x), dim=1)

        return self._block6(concat1)



