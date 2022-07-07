# Copyright (c) OpenMMLab. All rights reserved.
# predict depth according to the input stereo volume & depth loss
# Depth Loss Head for stereo matching supervision.

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import HEADS


@HEADS.register_module()
class DepthHead(nn.Module):
    """Depth prediction head with loss computation.

    To be refactored after challenge.
    """

    def __init__(self,
                 depth_cfg,
                 in_channels=32,
                 with_convs=True,
                 depth_loss=dict(type='ce', loss_weight=1.0),
                 downsample_factor=4,
                 num_views=5,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super().__init__()
        self.in_channels = in_channels
        self.depth_cfg = depth_cfg
        # whether involve conv layers for depth feature extraction
        self.with_convs = with_convs
        # prepare depth
        self.depth_loss = depth_loss
        self.downsample_factor = downsample_factor
        self.num_views = num_views
        self.norm_cfg = norm_cfg
        self.depth_loss_type = depth_loss['type']
        if self.depth_loss_type == 'balanced_ce':
            self.fg_weight = depth_loss['fg_weight']
            self.bg_weight = depth_loss['bg_weight']
        if self.depth_loss_type == 'focal':
            self.alpha = depth_loss['alpha']
            self.gamma = depth_loss['gamma']
        if self.depth_loss_type == 'balanced_focal':
            self.fg_weight = depth_loss['fg_weight']
            self.bg_weight = depth_loss['bg_weight']
            self.alpha = depth_loss['alpha']
            self.gamma = depth_loss['gamma']
        self.loss_weight = depth_loss['loss_weight']
        self.min_depth = depth_cfg['min_depth']
        self.max_depth = depth_cfg['max_depth']
        self._init_layers()

    def _init_layers(self):
        """
        from mmcv.cnn import ConvModule
        self.depth_convs = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True))
        """
        if self.with_convs:
            self.conv_depth = nn.Conv3d(
                self.in_channels, 1, 3, 1, 1, bias=False)
        self.upsample_cost = nn.Upsample(
            scale_factor=self.downsample_factor,
            mode='trilinear',
            align_corners=True)

    def loss(self,
             depth_preds,
             depth_volumes,
             depth_img,
             depth_fgmask_imgs=None):
        """
        Args:
            depth_preds: [B*N, H, W]
            depth_volumes: [B*N, D, H, W]
            depth_img: [B*N, H, W]
        """
        depth_samples = self.depth_samples.to(depth_preds.device)
        depth_loss = 0.
        assert len(depth_preds) == len(depth_volumes)
        mask = (depth_img > self.min_depth) & (depth_img < self.max_depth)
        gt = depth_img[mask]
        # depth_fgmask_imgs is a fg_mask with box_id
        if depth_fgmask_imgs is not None:
            fg_mask = depth_fgmask_imgs[mask].bool()
        depth_interval = depth_samples[1] - depth_samples[0]

        # only support one-level depth loss and one type of loss
        depth_pred = depth_preds[mask]
        depth_cost = depth_volumes.permute(0, 2, 3, 1)[mask]

        loss_type = self.depth_loss_type
        loss_type_weight = self.loss_weight
        if depth_pred.shape[0] == 0:
            print('no gt warning')
            loss = depth_preds.mean() * 0.0
        else:
            if loss_type == 'l1':
                loss = F.smooth_l1_loss(depth_pred, gt, reduction='none')
                loss = loss.mean()
            elif loss_type == 'purel1':
                loss = F.l1_loss(depth_pred, gt, reduction='none')
                loss = loss.mean()
            elif loss_type == 'ce':
                depth_log_prob = F.log_softmax(depth_cost, dim=1)
                distance = torch.abs(depth_samples -
                                     gt.unsqueeze(-1)) / depth_interval
                probability = 1 - distance.clamp(max=1.0)
                loss = -(probability * depth_log_prob).sum(-1)
                loss = loss.mean()
            elif loss_type == 'balanced_ce':
                depth_log_prob = F.log_softmax(depth_cost, dim=1)
                distance = torch.abs(depth_samples -
                                     gt.unsqueeze(-1)) / depth_interval
                probability = 1 - distance.clamp(max=1.0)
                loss = -(probability * depth_log_prob).sum(-1)
                loss = (self.fg_weight * loss[fg_mask]).sum() + (
                    self.bg_weight * loss[~fg_mask]).sum()
                loss /= len(gt)
            elif loss_type == 'focal':
                depth_log_prob = F.log_softmax(depth_cost, dim=1)
                depth_prob = depth_log_prob.exp()
                distance = torch.abs(depth_samples -
                                     gt.unsqueeze(-1)) / depth_interval
                probability = 1 - distance.clamp(max=1.0)
                loss = -(probability *
                         (self.alpha * (1 - depth_prob).pow(self.gamma) *
                          depth_log_prob)).sum(-1)
                loss = loss.mean()
            elif loss_type == 'balanced_focal':
                depth_log_prob = F.log_softmax(depth_cost, dim=1)
                depth_prob = depth_log_prob.exp()
                distance = torch.abs(depth_samples -
                                     gt.unsqueeze(-1)) / depth_interval
                probability = 1 - distance.clamp(max=1.0)
                loss = -(probability *
                         (self.alpha * (1 - depth_prob).pow(self.gamma) *
                          depth_log_prob)).sum(-1)
                loss = (self.fg_weight * loss[fg_mask]).sum() + (
                    self.bg_weight * loss[~fg_mask]).sum()
                loss /= len(gt)
            elif loss_type.startswith('gaussian'):
                depth_log_prob = F.log_softmax(depth_cost, dim=1)
                distance = torch.abs(depth_samples - gt.unsqueeze(-1))
                sigma = float(loss_type.split('_')[1])
                if dist.get_rank() == 0:
                    print('depth loss using gaussian normalized', sigma)
                probability = torch.exp(-0.5 * (distance**2) / (sigma**2))
                probability /= torch.clamp(
                    probability.sum(1, keepdim=True), min=1.0)
                loss = -(probability * depth_log_prob).sum(-1)
                loss = loss.mean()
            elif loss_type.startswith('laplacian'):
                depth_log_prob = F.log_softmax(depth_cost, dim=1)
                distance = torch.abs(depth_samples - gt.unsqueeze(-1))
                sigma = float(loss_type.split('_')[1])
                if dist.get_rank() == 0:
                    print('depth loss using laplacian normalized', sigma)
                probability = torch.exp(-distance / sigma)
                probability /= torch.clamp(
                    probability.sum(1, keepdim=True), min=1.0)
                loss = -(probability * depth_log_prob).sum(-1)
                loss = loss.mean()
            elif loss_type == 'hard_ce':
                depth_log_prob = F.log_softmax(depth_cost, dim=1)
                distance = torch.abs(depth_samples -
                                     gt.unsqueeze(-1)) / depth_interval
                probability = 1 - distance.clamp(max=1.0)
                probability[probability >= 0.5] = 1.0
                probability[probability < 0.5] = .0

                loss = -(probability * depth_log_prob).sum(-1)

                loss = loss.mean()
            else:
                raise NotImplementedError

            depth_loss += self.loss_weight * loss_type_weight * loss

        return depth_loss

    def forward(self, stereo_features):
        """
        Args:
            stereo_features: [B*num_views, C, D, H, W]

        Returns:
            depth_preds: [B, num_views, H, W]
        """
        _, _, D, H, W = stereo_features.shape
        # depth_volumes = self.depth_convs(stereo_features)
        if self.with_convs:
            depth_volumes = self.conv_depth(stereo_features).view(
                -1, self.num_views, D, H, W)
        else:
            depth_volumes = stereo_features
        # (B, N, D, H, W) -> (B, N, 4D, 4H, 4W)
        depth_volumes = self.upsample_cost(depth_volumes)
        depth_volumes_softmax = F.softmax(depth_volumes, dim=2)
        depth_preds = torch.sum(
            depth_volumes_softmax * self.depth_samples.to(
                stereo_features.device)[None, None, :, None, None], 2)
        # TODO: include depth results into the evaluation
        return depth_volumes, depth_volumes_softmax, depth_preds
