# Copyright (c) OpenMMLab. All rights reserved.
"""Adapted from LIGA-Stereo."""
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn

from mmdet3d.models.utils import upconv_module
from mmdet.models import NECKS


@NECKS.register_module()
class SPPUNetNeck(BaseModule):

    def __init__(self,
                 in_channels,
                 start_level,
                 sem_channels=[128, 32],
                 stereo_channels=[32, 32],
                 spp_channel=32,
                 with_upconv=True,
                 cat_img_feature=True,
                 norm_cfg=None):
        super(SPPUNetNeck, self).__init__()

        self.in_channels = in_channels
        self.start_level = start_level
        self.sem_channels = sem_channels
        self.stereo_channels = stereo_channels
        self.spp_channel = spp_channel
        self.with_upconv = with_upconv
        self.cat_img_feature = cat_img_feature

        self.spp_branches = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(s, stride=s),
                ConvModule(
                    self.in_channels[-1],
                    self.spp_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=norm_cfg))
            for s in [(64, 64), (32, 32), (16, 16), (8, 8)]
        ])

        concat_channel = self.spp_channel * len(self.spp_branches) + sum(
            self.in_channels[self.start_level:])

        if self.with_upconv:
            assert self.start_level == 2
            self.upconv_module = upconv_module(
                [concat_channel, self.in_channels[1], self.in_channels[0]],
                [64, 32])
            stereo_channel = 32
        else:
            stereo_channel = concat_channel
            assert self.start_level >= 1

        self.lastconv = nn.Sequential(
            ConvModule(
                stereo_channel,
                self.stereo_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg),
            nn.Conv2d(
                self.stereo_channels[0],
                self.stereo_channels[1],
                kernel_size=1,
                padding=0,
                stride=1,
                bias=False))
        if self.cat_img_feature:
            self.rpnconv = nn.Sequential(
                ConvModule(
                    concat_channel,
                    self.sem_channels[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg),
                ConvModule(
                    self.sem_channels[0],
                    self.sem_channels[1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg))

    def forward(self, feats):
        feat_shape = tuple(feats[self.start_level].shape[2:])
        assert len(feats) == len(self.in_channels)

        spp_branches = []
        for branch_module in self.spp_branches:
            x = branch_module(feats[-1])
            x = F.interpolate(
                x, feat_shape, mode='bilinear', align_corners=True)
            spp_branches.append(x)

        concat_feature = torch.cat((*feats[self.start_level:], *spp_branches),
                                   1)
        stereo_feature = concat_feature

        if self.with_upconv:
            stereo_feature = self.upconv_module(
                [stereo_feature, feats[1], feats[0]])

        stereo_feature = self.lastconv(stereo_feature)

        if self.cat_img_feature:
            sem_feature = self.rpnconv(concat_feature)
        else:
            sem_feature = None

        return stereo_feature, sem_feature
