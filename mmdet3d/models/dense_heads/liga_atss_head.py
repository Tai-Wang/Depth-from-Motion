# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale

from mmdet.core import anchor_inside_flags, multiclass_nms, unmap
from mmdet.models import HEADS
from mmdet.models.dense_heads import ATSSHead

EPS = 1e-12


@HEADS.register_module()
class LIGAATSSHead(ATSSHead):
    """Expanded ATSS head with some extra functions used in LIGA-Stereo."""

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 reg_class_agnostic=True,
                 num_extra_reg_channel=0,
                 seperate_extra_reg_branch=False,
                 reg_avg_factor='default',
                 **kwargs):
        self.reg_class_agnostic = reg_class_agnostic
        self.num_reg_channel = num_extra_reg_channel + 4
        self.reg_avg_factor = reg_avg_factor
        self.seperate_extra_reg_branch = \
            seperate_extra_reg_branch and self.num_reg_channel > 4
        assert self.reg_avg_factor in ['default', 'sum_centerness']
        super(LIGAATSSHead, self).__init__(
            num_classes,
            in_channels,
            stacked_convs=stacked_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head.

        add reg_class_agnostic
        """
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        if self.seperate_extra_reg_branch:
            self.extra_reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            if self.seperate_extra_reg_branch:
                self.extra_reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        if not self.seperate_extra_reg_branch:
            if self.reg_class_agnostic:
                self.atss_reg = nn.Conv2d(
                    self.feat_channels,
                    self.num_anchors * self.num_reg_channel,
                    3,
                    padding=1)
            else:
                self.atss_reg = nn.Conv2d(
                    self.feat_channels,
                    self.num_anchors * self.num_reg_channel * self.num_classes,
                    3,
                    padding=1)
        else:
            if self.reg_class_agnostic:
                self.atss_reg = nn.Conv2d(
                    self.feat_channels, self.num_anchors * 4, 3, padding=1)
                self.atss_extra_reg = nn.Conv2d(
                    self.feat_channels,
                    self.num_anchors * (self.num_reg_channel - 4),
                    3,
                    padding=1)
            else:
                self.atss_reg = nn.Conv2d(
                    self.feat_channels,
                    self.num_anchors * 4 * self.num_classes,
                    3,
                    padding=1)
                self.atss_extra_reg = nn.Conv2d(
                    self.feat_channels,
                    self.num_anchors * (self.num_reg_channel - 4) *
                    self.num_classes,
                    3,
                    padding=1)
        self.atss_centerness = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        if self.seperate_extra_reg_branch:
            extra_reg_feat = x
            for extra_reg_conv in self.extra_reg_convs:
                extra_reg_feat = extra_reg_conv(extra_reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        if not self.seperate_extra_reg_branch:
            bbox_pred = scale(self.atss_reg(reg_feat)).float()
        else:
            bbox_pred = scale(self.atss_reg(reg_feat)).float()
            extra_bbox_pred = self.atss_extra_reg(extra_reg_feat).float()
            if self.reg_class_agnostic:
                # [B, self.num_anchors * 4, H, W] &
                # [B, self.num_anchors * (self.num_reg_channel - 4), H, W]
                B, _, H, W = bbox_pred.shape
                bbox_pred = bbox_pred.view(B, self.num_anchors, 4, H, W)
                extra_bbox_pred = extra_bbox_pred(B, self.num_anchors,
                                                  self.num_reg_channel - 4, H,
                                                  W)
                bbox_pred = torch.cat([bbox_pred, extra_bbox_pred], dim=2)
                bbox_pred = bbox_pred.view(
                    B, self.num_anchors * self.num_reg_channel, H, W)
            else:
                # [B, self.num_anchors * 4 * self.num_classes, H, W] &
                # [B, self.num_anchors * (self.num_reg_channel - 4), H, W]
                B, _, H, W = bbox_pred.shape
                bbox_pred = bbox_pred.view(B, self.num_anchors, 4,
                                           self.num_classes, H, W)
                extra_bbox_pred = extra_bbox_pred.view(
                    B, self.num_anchors, self.num_reg_channel - 4,
                    self.num_classes, H, W)
                bbox_pred = torch.cat([bbox_pred, extra_bbox_pred], dim=2)
                bbox_pred = bbox_pred.view(
                    B,
                    self.num_anchors * self.num_reg_channel * self.num_classes,
                    H, W)
        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness

    def loss_single(self, anchors, cls_score, bbox_pred, centerness, labels,
                    label_weights, bbox_targets, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        if self.reg_class_agnostic:
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(-1, self.num_reg_channel)
        else:
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(-1, self.num_reg_channel,
                                                     self.num_classes)
            # add a blank bg class, --> [N, 4, N_cls+1]
            bbox_pred = torch.cat(
                [bbox_pred, torch.zeros_like(bbox_pred[:, :, :1])], dim=-1)
            bbox_pred = torch.gather(
                bbox_pred,
                dim=2,
                index=labels.reshape(-1, 1, 1).repeat(1, self.num_reg_channel,
                                                      1)).squeeze(-1)

        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, self.num_reg_channel)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)

            # regression loss
            if self.reg_avg_factor == 'default':
                loss_bbox_avg_factor = 1.0
            elif self.reg_avg_factor == 'sum_centerness':
                loss_bbox_avg_factor = centerness_targets.sum()
            else:
                raise ValueError('wrong reg_avg_factor')
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=centerness_targets,
                avg_factor=loss_bbox_avg_factor)

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets,
                avg_factor=num_total_samples)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_anchors * 1, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, anchors in zip(
                cls_scores, bbox_preds, centernesses, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            if self.reg_class_agnostic:
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            else:
                max_cls_indices = scores.argmax(1)
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(
                    -1, self.num_reg_channel, self.num_classes)
                # add a blank bg class, --> [N, 4, N_cls+1]
                bbox_pred = torch.cat(
                    [bbox_pred,
                     torch.zeros_like(bbox_pred[:, :, :1])], dim=-1)
                bbox_pred = torch.gather(
                    bbox_pred,
                    dim=2,
                    index=max_cls_indices.reshape(-1, 1, 1).repeat(
                        1, self.num_reg_channel, 1)).squeeze(-1)

            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_centerness,
                num_bbox_channels=self.num_reg_channel)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_centerness

    def centerness_target(self, anchors, bbox_targets):
        # NOTE: here needs decode first in this head design
        # only calculate pos centerness targets, otherwise there may be nan
        gts = self.bbox_coder.decode(anchors, bbox_targets)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True,
                           return_sampling_results=False):
        """Fix some incompatible minor places compared to the version in
        ATSSHead."""
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        if getattr(self, 'num_reg_channel', None) is None:
            self.num_reg_channel = anchors.shape[1]
        bbox_targets = anchors.new_zeros(
            [anchors.shape[0], self.num_reg_channel])
        bbox_weights = anchors.new_zeros(
            [anchors.shape[0], self.num_reg_channel])
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if hasattr(self, 'bbox_coder'):
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes,
                    sampling_result.pos_gt_bboxes[:, :4])
            else:
                # used in VFNetHead
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        if not return_sampling_results:
            return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                    pos_inds, neg_inds)
        else:
            return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                    pos_inds, neg_inds, assign_result)
