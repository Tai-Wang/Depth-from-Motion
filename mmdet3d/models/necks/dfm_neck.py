# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from torch import nn

from ..builder import NECKS
from .imvoxel_neck import ResModule


@NECKS.register_module()
class DfMNeck(nn.Module):
    """Dual-path neck for monocular and stereo bev fusion in DfM.

    Args:
        in_channels (int): Input channels of multi-scale feature map.
        out_channels (int): Output channels of multi-scale feature map.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN3d'),
                 num_frames=2):
        super().__init__()
        if not isinstance(in_channels, list):
            in_channels = [in_channels, in_channels * 2, in_channels * 4]
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.mono_layers = nn.Sequential(
            ResModule(in_channels[0], norm_cfg=norm_cfg),
            ConvModule(
                in_channels=in_channels[0],
                out_channels=in_channels[1],
                kernel_size=3,
                stride=(1, 1, 2),
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)),
            ResModule(in_channels[1], norm_cfg=norm_cfg),
            ConvModule(
                in_channels=in_channels[1],
                out_channels=in_channels[2],
                kernel_size=3,
                stride=(1, 1, 2),
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)),
            ResModule(in_channels[2], norm_cfg=norm_cfg),
            ConvModule(
                in_channels=in_channels[2],
                out_channels=out_channels,
                kernel_size=3,
                padding=(1, 1, 0),
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        self.stereo_layers = nn.Sequential(
            ResModule(in_channels[0] * num_frames, norm_cfg=norm_cfg),
            ConvModule(
                in_channels=in_channels[0] * num_frames,
                out_channels=in_channels[1],
                kernel_size=3,
                stride=(1, 1, 2),
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)),
            ResModule(in_channels[1], norm_cfg=norm_cfg),
            ConvModule(
                in_channels=in_channels[1],
                out_channels=in_channels[2],
                kernel_size=3,
                stride=(1, 1, 2),
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)),
            ResModule(in_channels[2], norm_cfg=norm_cfg),
            ConvModule(
                in_channels=in_channels[2],
                out_channels=out_channels,
                kernel_size=3,
                padding=(1, 1, 0),
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        self.aggregate_layer = nn.Conv2d(
            2 * out_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C_in, N_x, N_y, N_z).

        Returns:
            list[torch.Tensor]: of shape (N, C_out, N_y, N_x).
        """
        # input x should be concat of features of all the frames
        assert x.shape[1] == self.in_channels[0] * self.num_frames
        mono_bev_feat = self.mono_layers(x[:, :self.in_channels[0]])
        assert mono_bev_feat.shape[-1] == 1
        mono_bev_feat = mono_bev_feat[..., 0].transpose(-1, -2)
        stereo_bev_feat = self.stereo_layers(x)
        assert stereo_bev_feat.shape[-1] == 1
        stereo_bev_feat = stereo_bev_feat[..., 0].transpose(-1, -2)
        # transform to Anchor3DHead axis order (y, x).
        bev_feats = torch.cat([mono_bev_feat, stereo_bev_feat], dim=1)
        weight = self.aggregate_layer(bev_feats).sigmoid()
        bev_feats = weight * mono_bev_feat + (1 - weight) * stereo_bev_feat
        return [bev_feats]

    def init_weights(self):
        """Initialize weights of neck."""
        pass
