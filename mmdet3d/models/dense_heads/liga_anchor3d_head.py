# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn as nn

from mmdet3d.models.utils import convbn
from mmdet.models import HEADS
from .anchor3d_head import Anchor3DHead


@HEADS.register_module()
class LIGAAnchor3DHead(Anchor3DHead):
    """Anchor head for LIGA-DfM or LIGA-Stereo.

    Differences with Anchor3DHead:
    1. use kernel_size=3 for conv_cls and conv_reg
    2. add num_convs option for previous layers of two branches

    Args:
        num_convs (int): The number of previous conv layers of classification
            and regression branch.
    """

    def __init__(self, num_convs=2, norm_cfg=None, **kwargs):
        self.num_convs = num_convs
        self.norm_cfg = norm_cfg
        super().__init__(**kwargs)

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        if self.num_convs > 0:
            self.cls_convs = []
            self.reg_convs = []
            for _ in range(self.num_convs):

                self.cls_convs.append(
                    nn.Sequential(
                        convbn(
                            self.in_channels,
                            self.feat_channels,
                            3,
                            1,
                            1,
                            1,
                            gn=True), nn.ReLU(inplace=True)))
                self.reg_convs.append(
                    nn.Sequential(
                        convbn(
                            self.in_channels,
                            self.feat_channels,
                            3,
                            1,
                            1,
                            1,
                            gn=True), nn.ReLU(inplace=True)))
                """
                from mmcv.cnn import ConvModule
                self.cls_convs.append(
                    ConvModule(
                        self.in_channels,
                        self.in_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        dilation=1,
                        norm_cfg=self.norm_cfg))
                self.reg_convs.append(
                    ConvModule(
                        self.in_channels,
                        self.in_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        dilation=1,
                        norm_cfg=self.norm_cfg))
                """

            self.cls_convs = nn.Sequential(*self.cls_convs)
            self.reg_convs = nn.Sequential(*self.reg_convs)
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(
            self.feat_channels,
            self.cls_out_channels,
            kernel_size=3,
            padding=1,
            stride=1)
        self.conv_reg = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.box_code_size,
            kernel_size=3,
            padding=1,
            stride=1)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(self.feat_channels,
                                          self.num_anchors * 2, 1)

    def forward_single(self, x):
        """Forward function on a single-scale feature map.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox
                regression and direction classification predictions.
        """
        cls_feats = x
        reg_feats = x
        if self.num_convs > 0:
            cls_feats = self.cls_convs(cls_feats)
            reg_feats = self.reg_convs(reg_feats)
        cls_score = self.conv_cls(cls_feats)
        bbox_pred = self.conv_reg(reg_feats)
        dir_cls_preds = None
        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(cls_feats)
        return cls_score, bbox_pred, dir_cls_preds
