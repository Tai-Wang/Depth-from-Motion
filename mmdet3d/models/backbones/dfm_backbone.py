# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from mmdet3d.core.bbox import points_cam2img, points_img2cam
from mmdet3d.models.utils import convbn_3d, hourglass
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class DfMBackbone(nn.Module):

    def __init__(
        self,
        in_channels,
        num_hg=1,
        downsample_factor=4,
        cv_channels=32,  # cost volume channels
        depth_cfg=dict(
            mode='UD',
            num_bins=288,
            depth_min=2,
            depth_max=59.6,
            downsample_factor=4),
        norm_cfg=dict(type='GN', num_groups=32,
                      requires_grad=True)  # use GN by default
    ):
        super(DfMBackbone, self).__init__()

        # general config
        self.norm_cfg = norm_cfg
        self.GN = True  # TODO: replace it with norm_cfg

        # stereo config
        self.downsample_factor = downsample_factor
        self.num_hg = num_hg

        # volume config
        self.cv_channels = cv_channels

        # stereo network
        self.in_channels = in_channels

        self.dres0 = nn.Sequential(
            convbn_3d(2 * in_channels, self.cv_channels, 3, 1, 1, gn=self.GN),
            nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(
            convbn_3d(self.cv_channels, self.cv_channels, 3, 1, 1, gn=self.GN))
        """
        self.dres0 = ConvModule(
            2 * in_channels,
            self.cv_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=norm_cfg)
        self.dres1 = ConvModule(
            self.cv_channels,
            self.cv_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=norm_cfg,
            act_cfg=None)
        """

        self.hg_stereo = nn.ModuleList()
        for _ in range(self.num_hg):
            self.hg_stereo.append(hourglass(self.cv_channels, gn=self.GN))

        # stereo predictions
        self.pred_stereo = nn.ModuleList()
        for _ in range(self.num_hg):
            self.pred_stereo.append(self.build_depth_pred_module())

        # mono network
        self.dres0_mono = nn.Sequential(
            convbn_3d(self.in_channels, self.cv_channels, 3, 1, 1, gn=self.GN),
            nn.ReLU(inplace=True))
        self.dres1_mono = nn.Sequential(
            convbn_3d(self.cv_channels, self.cv_channels, 3, 1, 1, gn=self.GN))
        """
        self.dres0_mono = ConvModule(
            self.in_channels,
            self.cv_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=norm_cfg)
        self.dres1_mono = ConvModule(
            self.cv_channels,
            self.cv_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=norm_cfg,
            act_cfg=None)
        """

        self.hg_mono = nn.ModuleList()
        for _ in range(self.num_hg):
            self.hg_mono.append(hourglass(self.cv_channels, gn=self.GN))

        # mono predictions
        self.pred_mono = nn.ModuleList()
        for _ in range(self.num_hg):
            self.pred_mono.append(self.build_depth_pred_module())

        # switch of stereo or mono predictions
        aggregate_out_dim = round(depth_cfg['num_bins'] //
                                  depth_cfg['downsample_factor'])
        self.aggregate_cost = nn.Conv2d(
            2 * aggregate_out_dim,
            aggregate_out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        self.init_weights()

    def build_depth_pred_module(self):
        return nn.Sequential(
            convbn_3d(self.cv_channels, self.cv_channels, 3, 1, 1, gn=self.GN),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.cv_channels, 1, 3, 1, 1, bias=False))
        """
        return nn.Sequential(
            ConvModule(
                self.cv_channels,
                self.cv_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=self.norm_cfg),
            nn.Conv3d(self.cv_channels, 1, 3, 1, 1, bias=False))
        """

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[
                    2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def mono_stereo_aggregate(self, stereo_depth_conv_module,
                              mono_depth_conv_module, cost1, mono_cost1,
                              img_shape):
        cost1 = stereo_depth_conv_module(cost1)
        mono_cost1 = mono_depth_conv_module(mono_cost1)
        # (1, 2*D, H, W)
        cost = torch.cat((cost1, mono_cost1), dim=1).flatten(
            start_dim=1, end_dim=2)
        # (1, 1, D, H, W)
        weight = self.aggregate_cost(cost).unsqueeze(dim=1).sigmoid()
        cost = weight * cost1 + (1 - weight) * mono_cost1
        return cost

    def forward(self,
                cur_stereo_feats,
                prev_stereo_feats,
                img_metas,
                cur_sem_feats=None):
        # TODO: refactor calibration matrix
        # ori_cam2imgs, cam2imgs: (B, 4, 4)
        ori_cam2imgs = torch.as_tensor(
            [img_meta['ori_cam2img'] for img_meta in img_metas],
            dtype=torch.float32,
            device=cur_stereo_feats.device)
        # cur2prevs: (B, N-1, 4, 4)
        cur2prevs = torch.stack(
            [img_meta['cur2prevs'] for img_meta in img_metas])

        # stereo matching: build stereo volume
        downsampled_depth = self.downsampled_depth.to(cur_stereo_feats.device)
        # only support batch size 1 for now
        cost_raw = build_dfm_cost(
            cur_stereo_feats,
            prev_stereo_feats,
            downsampled_depth,
            self.downsample_factor,
            ori_cam2imgs,
            cur2prevs[0],
            img_metas[0]['ori_shape'][:2],
            img_metas[0].get('flip', False),
            img_metas[0]['crop_offset'],
            img_scale_factor=img_metas[0].get('scale_factor', [1.0])[0])

        # stereo matching network
        cost0 = self.dres0(cost_raw)
        cost0 = self.dres1(cost0) + cost0
        if len(self.hg_stereo) > 0:
            all_costs = []
            cur_cost = cost0
            for hg_stereo_module in self.hg_stereo:
                cost_residual, _, _ = hg_stereo_module(cur_cost, None, None)
                cur_cost = cur_cost + cost_residual
                all_costs.append(cur_cost)
        else:
            all_costs = [cost0]
        assert len(all_costs) > 0, 'at least one hourglass'

        # mono depth estimation network
        cost0_mono = self.dres0_mono(cost_raw[:, :self.in_channels])
        cost0_mono = self.dres1_mono(cost0_mono) + cost0_mono
        if len(self.hg_mono) > 0:
            all_costs_mono = []
            cur_cost_mono = cost0_mono
            for hg_mono_module in self.hg_mono:
                cost_mono_residual, _, _ = hg_mono_module(
                    cur_cost_mono, None, None)
                cur_cost_mono = cur_cost_mono + cost_mono_residual
                all_costs_mono.append(cur_cost_mono)
        else:
            all_costs_mono = [cost0_mono]
        assert len(all_costs_mono) > 0, 'at least one hourglass'

        # stereo matching: outputs
        mono_stereo_costs = []
        # upsample the 3d volume and predict depth by pred_depth (trilinear)
        for idx in range(len(all_costs)):
            mono_stereo_cost_i = self.mono_stereo_aggregate(
                self.pred_stereo[idx], self.pred_mono[idx], all_costs[idx],
                all_costs_mono[idx], img_metas[0]['pad_shape'])
            mono_stereo_costs.append(mono_stereo_cost_i)

        assert len(mono_stereo_costs) == 1, 'Only support num_hg=1 for now.'

        return mono_stereo_costs[0], all_costs[0], all_costs_mono[0]


def build_dfm_cost(cur_feats,
                   prev_feats,
                   depths,
                   sample_factor,
                   cam2imgs,
                   cur2prevs,
                   img_shape,
                   flip=False,
                   img_crop_offset=(0, 0),
                   img_scale_factor=1.0):
    """
    Args:
        cur_feats/prev_feats: [B, C, H, W]
        depths: [1, D]

    Returns:
        cost_volume: [B, 2C, D, downsample_H (H_out), downsample_W (W_out)]
    """
    # TODO: check whether there is any bug when feats are downsampled
    device = depths.device
    img_crop_offset = torch.tensor(img_crop_offset, device=device)
    batch_size = cur_feats.shape[0]
    h, w = cur_feats.shape[-2:]
    num_depths = depths.shape[-1]
    h_out = round(h / sample_factor)
    w_out = round(w / sample_factor)
    ws = (torch.linspace(0, w_out - 1, w_out) * sample_factor).to(device)
    hs = (torch.linspace(0, h_out - 1, h_out) * sample_factor).to(device)
    ds_3d, ys_3d, xs_3d = torch.meshgrid(depths, hs, ws)
    # grid: (D, H_out, W_out, 3)
    grid = torch.stack([xs_3d, ys_3d, ds_3d], dim=-1)
    # grid: (B, D, H_out, W_out, 3)
    grid = grid[None].repeat(batch_size, 1, 1, 1, 1)
    # apply 3D transformation to get original cur and prev 3D grid
    for idx in range(batch_size):
        # grid3d: (D*H_out*W_out, 3)
        grid3d = points_img2cam(grid[idx].view(-1, 3), cam2imgs[idx])
        # only support flip transformation for now
        if flip:  # get the original 3D grid by transforming it back
            grid3d[:, 0] = -grid3d[:, 0]
        pad_ones = grid3d.new_ones(grid3d.shape[0], 1)
        homo_grid3d = torch.cat([grid3d, pad_ones], dim=1)
        cur_grid = points_cam2img(grid3d, cam2imgs[idx])[:, :2]
        prev_grid3d = (homo_grid3d @ cur2prevs[idx].transpose(0, 1))[:, :3]
        prev_grid = points_cam2img(prev_grid3d, cam2imgs[idx])[:, :2]
    # cur_grid: (B, 1, D*H_out*W_out, 2)
    cur_grid = cur_grid.view(batch_size, 1, -1, 2)
    # prev_grid: (B, 1, D*H_out*W_out, 2)
    prev_grid = prev_grid.view(batch_size, 1, -1, 2)
    # apply 2D transformation on 2.5D grid to get correct 2D feats from images
    # img transformation: flip -> scale -> crop
    if flip:
        org_h, org_w = img_shape  # should be the original image shape
        cur_grid[..., 0] = org_w - cur_grid[..., 0]
        prev_grid[..., 0] = org_w - prev_grid[..., 0]
    cur_grid *= img_scale_factor
    prev_grid *= img_scale_factor
    cur_grid -= img_crop_offset
    prev_grid -= img_crop_offset
    # normalize grid
    # w-1 because the index is from [0, shape-1]
    cur_grid[..., 0] = cur_grid[..., 0] / (w - 1) * 2 - 1
    cur_grid[..., 1] = cur_grid[..., 1] / (h - 1) * 2 - 1
    prev_grid[..., 0] = prev_grid[..., 0] / (w - 1) * 2 - 1
    prev_grid[..., 1] = prev_grid[..., 1] / (h - 1) * 2 - 1
    # TOCHECK: sample or the original better?
    cur_cost_feats = F.grid_sample(
        cur_feats,
        cur_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)  # (B, C, 1, D*H_out*W_out)
    cur_cost_feats = cur_cost_feats.view(batch_size, -1, num_depths, h_out,
                                         w_out)  # (B, C, D, H_out, W_out)
    prev_cost_feats = F.grid_sample(
        prev_feats,
        prev_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)  # (B, C, 1, D*H_out*W_out)
    prev_cost_feats = prev_cost_feats.view(batch_size, -1, num_depths, h_out,
                                           w_out)
    # cost_volume: (B, 2C, D, H_out, W_out)
    cost_volume = torch.cat([cur_cost_feats, prev_cost_feats], dim=1)
    return cost_volume
