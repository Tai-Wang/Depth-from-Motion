# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule
from torch import nn

from ..builder import NECKS


@NECKS.register_module()
class OutdoorImVoxelNeck(nn.Module):
    """Neck for ImVoxelNet outdoor scenario.

    Args:
        in_channels (int): Input channels of multi-scale feature map.
        out_channels (int): Output channels of multi-scale feature map.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN3d'),
                 output_bev=True):
        super().__init__()
        self.output_bev = output_bev
        if not isinstance(in_channels, list):
            in_channels = [in_channels, in_channels * 2, in_channels * 4]
        self.model = nn.Sequential(
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

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C_in, N_x, N_y, N_z).

        Returns:
            list[torch.Tensor]: of shape (N, C_out, N_y, N_x).
        """
        x = self.model.forward(x)
        assert x.shape[-1] == 1 and self.output_bev
        # Anchor3DHead axis order is (y, x).
        return [x[..., 0].transpose(-1, -2)]

    def init_weights(self):
        """Initialize weights of neck."""
        pass


class ResModule(nn.Module):
    """3d residual block for ImVoxelNeck.

    Args:
        n_channels (int): Input channels of a feature map.
    """

    def __init__(self, n_channels, norm_cfg=dict(type='BN3d')):
        super().__init__()
        self.conv0 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True))
        self.conv1 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C, N_x, N_y, N_z).

        Returns:
            torch.Tensor: 5d feature map.
        """
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = identity + x
        x = self.activation(x)
        return x
