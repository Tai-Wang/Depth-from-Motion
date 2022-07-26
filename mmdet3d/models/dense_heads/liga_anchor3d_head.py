# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn as nn

from mmdet3d.models.utils import convbn
from mmdet.core import multi_apply
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

    def __init__(self,
                 num_convs=2,
                 norm_cfg=None,
                 normalizer_clamp_value=10,
                 **kwargs):
        self.num_convs = num_convs
        self.norm_cfg = norm_cfg
        self.normalizer_clamp_value = normalizer_clamp_value
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
                            gn=(self.norm_cfg['type'] == 'GN')),
                        nn.ReLU(inplace=True)))
                self.reg_convs.append(
                    nn.Sequential(
                        convbn(
                            self.in_channels,
                            self.feat_channels,
                            3,
                            1,
                            1,
                            1,
                            gn=(self.norm_cfg['type'] == 'GN')),
                        nn.ReLU(inplace=True)))
                """
                from mmcv.cnn import ConvModule
                self.cls_convs.append(
                    ConvModule(
                        self.in_channels,
                        self.feat_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        dilation=1,
                        norm_cfg=self.norm_cfg))
                self.reg_convs.append(
                    ConvModule(
                        self.in_channels,
                        self.feat_channels,
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

    def get_anchors(self, featmap_sizes, input_metas, device='cuda'):
        """Save anchor_list to self.anchors based on the get_anchors in
        Anchor3dHead."""
        num_imgs = len(input_metas)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        self.anchors = anchor_list
        return anchor_list

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple[list[torch.Tensor]]: Multi-level class score, bbox
                and direction predictions.
        """
        # TODO: remove this hack for LIGA Anchor3DHead
        if not isinstance(feats, list):
            feats = [feats]
        return multi_apply(self.forward_single, feats)

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

    def loss_single(self, cls_score, bbox_pred, dir_cls_preds, labels,
                    label_weights, bbox_targets, bbox_weights, dir_targets,
                    dir_weights, anchors, num_total_samples):
        """Different in dealing with the clamp value for loss normalizer."""
        # classification loss
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        assert labels.max().item() <= self.num_classes

        loss_cls = self.loss_cls(
            cls_score,
            labels,
            label_weights,
            avg_factor=(num_total_samples + self.normalizer_clamp_value))

        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, self.box_code_size)
        bbox_targets = bbox_targets.reshape(-1, self.box_code_size)
        bbox_weights = bbox_weights.reshape(-1, self.box_code_size)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero(
                        as_tuple=False).reshape(-1)
        num_pos = len(pos_inds)

        pos_bbox_pred = bbox_pred[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_weights = bbox_weights[pos_inds]

        # dir loss
        if self.use_direction_classifier:
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).reshape(-1, 2)
            dir_targets = dir_targets.reshape(-1)
            dir_weights = dir_weights.reshape(-1)
            pos_dir_cls_preds = dir_cls_preds[pos_inds]
            pos_dir_targets = dir_targets[pos_inds]
            pos_dir_weights = dir_weights[pos_inds]

        if num_pos > 0:
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                pos_bbox_weights = pos_bbox_weights * bbox_weights.new_tensor(
                    code_weight)
            if self.diff_rad_by_sin:
                pos_bbox_pred, pos_bbox_targets = self.add_sin_difference(
                    pos_bbox_pred, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_targets,
                pos_bbox_weights,
                avg_factor=max(num_total_samples, self.normalizer_clamp_value))

            # direction classification loss
            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = self.loss_dir(
                    pos_dir_cls_preds,
                    pos_dir_targets,
                    pos_dir_weights,
                    avg_factor=max(num_total_samples,
                                   self.normalizer_clamp_value))
        else:
            loss_bbox = pos_bbox_pred.sum()
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_preds.sum()

        losses = (loss_cls, loss_bbox, loss_dir)

        # iou loss
        if self.loss_iou is not None:
            # TODO: check the batch_size > 1 case
            anchors = anchors.view(-1, self.box_code_size)
            decode_bbox_preds = self.bbox_coder.decode(anchors[pos_inds],
                                                       bbox_pred[pos_inds])
            decode_bbox_targets = self.bbox_coder.decode(
                anchors[pos_inds], bbox_targets[pos_inds])
            loss_iou = self.loss_iou(
                decode_bbox_preds,
                decode_bbox_targets,
                weight=None,
                avg_factor=max(num_total_samples, self.normalizer_clamp_value))
            losses += (loss_iou, )

        return losses
