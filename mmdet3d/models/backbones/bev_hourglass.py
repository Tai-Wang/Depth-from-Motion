# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.utils import convbn
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class BEVHourglass(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=None,
                 output_prehg_feat=True):
        super(BEVHourglass, self).__init__()
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.output_prehg_feat = output_prehg_feat
        """
        from mmcv.cnn import ConvModule
        self.compress_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            norm_cfg=norm_cfg)
        """
        self.compress_conv = nn.Sequential(
            convbn(
                in_channels,
                out_channels,
                3,
                1,
                1,
                1,
                gn=(norm_cfg['type'] == 'GN')), nn.ReLU(inplace=True))

        self.bev_hourglass = hourglass2d(
            self.out_channels, gn=(norm_cfg['type'] == 'GN'))
        self.num_bev_features = self.out_channels

    def forward(self, spatial_features):
        x = self.compress_conv(spatial_features)
        spatial_features_2d_prehg = x
        x = self.bev_hourglass(x, None, None)[0]
        spatial_features_2d = x

        outputs = spatial_features_2d
        if self.output_prehg_feat:
            outputs = (spatial_features_2d_prehg, spatial_features_2d)

        return outputs


class hourglass2d(nn.Module):

    def __init__(self, inplanes, gn=False):
        super(hourglass2d, self).__init__()

        self.conv1 = nn.Sequential(
            convbn(
                inplanes,
                inplanes * 2,
                kernel_size=3,
                stride=2,
                pad=1,
                dilation=1,
                gn=gn), nn.ReLU(inplace=True))

        self.conv2 = convbn(
            inplanes * 2,
            inplanes * 2,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            gn=gn)

        self.conv3 = nn.Sequential(
            convbn(
                inplanes * 2,
                inplanes * 2,
                kernel_size=3,
                stride=2,
                pad=1,
                dilation=1,
                gn=gn), nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            convbn(
                inplanes * 2,
                inplanes * 2,
                kernel_size=3,
                stride=1,
                pad=1,
                dilation=1,
                gn=gn), nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(
                inplanes * 2,
                inplanes * 2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False),
            nn.SyncBatchNorm(inplanes *
                             2) if not gn else nn.GroupNorm(32, inplanes *
                                                            2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(
                inplanes * 2,
                inplanes,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False),
            nn.SyncBatchNorm(inplanes) if not gn else nn.GroupNorm(
                32, inplanes))  # +x

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(
                self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post
