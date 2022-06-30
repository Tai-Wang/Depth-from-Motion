# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
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
        num_3dconvs=1,
        num_hg=1,
        depth_cfg=dict(mode='UD', num_bins=288, depth_min=2, depth_max=59.6),
        voxel_cfg=dict(
            point_cloud_range=[2, -30.4, -3, 59.6, 30.4, 1],
            voxel_size=[0.2, 0.2, 0.2]),
        downsample_factor=4,
        sem_atten_feat=True,
        stereo_atten_feat=False,
        cat_img_feature=True,
        in_sem_channels=32,
        cv_channels=32,  # cost volume channels
        out_channels=32,  # out volume channels after conv/pool
        norm_cfg=dict(type='GN', num_groups=32,
                      requires_grad=True)  # use GN by default
    ):
        super(DfMBackbone, self).__init__()

        # general config
        self.GN = True  # TODO: replace it with norm_cfg

        # stereo config
        self.depth_cfg = depth_cfg
        self.voxel_cfg = voxel_cfg
        self.downsample_factor = downsample_factor
        self.downsampled_depth_offset = 0.5  # TODO: only use default value
        self.num_hg = num_hg

        # volume config
        self.num_3dconvs = num_3dconvs
        self.cv_channels = cv_channels
        self.out_channels = out_channels
        self.in_sem_channels = in_sem_channels

        # stereo network
        self.in_channels = in_channels
        self.dres0 = nn.Sequential(
            convbn_3d(2 * in_channels, self.cv_channels, 3, 1, 1, gn=self.GN),
            nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(
            convbn_3d(self.cv_channels, self.cv_channels, 3, 1, 1, gn=self.GN))
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
            convbn_3d(self.in_channels, self.cv_channels, 3, 1, 1, gn=self.GN))
        self.hg_mono = nn.ModuleList()
        for _ in range(self.num_hg):
            self.hg_mono.append(hourglass(self.cv_channels, gn=self.GN))

        # mono predictions
        self.pred_mono = nn.ModuleList()
        for _ in range(self.num_hg):
            self.pred_mono.append(self.build_depth_pred_module())

        # switch of stereo or mono predictions
        aggregate_out_dim = round(self.depth_cfg['num_bins'] //
                                  self.downsample_factor)
        self.aggregate_cost = nn.Conv2d(
            2 * aggregate_out_dim,
            aggregate_out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.upsample_cost = nn.Upsample(
            scale_factor=self.downsample_factor,
            mode='trilinear',
            align_corners=True)

        # aggregate features args
        self.sem_atten_feat = sem_atten_feat
        self.stereo_atten_feat = stereo_atten_feat
        self.cat_img_feature = cat_img_feature

        # conv layers for voxel feature volume (after grid sampling)
        voxel_channels = self.cv_channels
        if getattr(self, 'cat_img_feature', False):
            if self.cat_img_feature:
                voxel_channels += self.in_sem_channels
        else:
            self.cat_img_feature = False

        voxel_convs = []
        for i in range(self.num_3dconvs):
            voxel_convs.append(
                nn.Sequential(
                    convbn_3d(
                        voxel_channels if i == 0 else self.out_channels,
                        self.out_channels,
                        3,
                        1,
                        1,
                        gn=self.GN), nn.ReLU(inplace=True)))
        self.voxel_convs = nn.Sequential(*voxel_convs)
        self.voxel_pool = torch.nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1))
        self.num_3d_features = self.out_channels

        # prepare tensors
        self.prepare_depth(depth_cfg)
        self.prepare_coordinates_3d(voxel_cfg)
        self.init_params()

    def build_depth_pred_module(self):
        return nn.Sequential(
            convbn_3d(self.cv_channels, self.cv_channels, 3, 1, 1, gn=self.GN),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.cv_channels, 1, 3, 1, 1, bias=False))

    def prepare_depth(self, depth_cfg):
        assert depth_cfg['depth_min'] >= 0 and \
            depth_cfg['depth_max'] > depth_cfg['depth_min']
        depth_interval = (depth_cfg['depth_max'] -
                          depth_cfg['depth_min']) / depth_cfg['num_bins']
        print(f"stereo volume depth range: {depth_cfg['depth_min']} -> " +
              f"{depth_cfg['depth_max']}, interval {depth_interval}")
        # prepare downsampled depth
        self.downsampled_depth = torch.zeros(
            (depth_cfg['num_bins'] // self.downsample_factor),
            dtype=torch.float32)
        for i in range(depth_cfg['num_bins'] // self.downsample_factor):
            self.downsampled_depth[i] = (
                i + self.downsampled_depth_offset
            ) * self.downsample_factor * depth_interval + \
                depth_cfg['depth_min']
        # prepare depth
        self.depth = torch.zeros((depth_cfg['num_bins']), dtype=torch.float32)
        for i in range(depth_cfg['num_bins']):
            self.depth[i] = (i + 0.5) * depth_interval + depth_cfg['depth_min']

    def prepare_coordinates_3d(self, voxel_cfg, sample_rate=(1, 1, 1)):
        self.min_x, self.min_y, self.min_z = voxel_cfg['point_cloud_range'][:3]
        self.max_x, self.max_y, self.max_z = voxel_cfg['point_cloud_range'][3:]
        self.voxel_size_x, self.voxel_size_y, self.voxel_size_z = voxel_cfg[
            'voxel_size']
        grid_size = (
            np.array(voxel_cfg['point_cloud_range'][3:6], dtype=np.float32) -
            np.array(voxel_cfg['point_cloud_range'][0:3],
                     dtype=np.float32)) / np.array(voxel_cfg['voxel_size'])
        self.grid_size_x, self.grid_size_y, self.grid_size_z = (
            np.round(grid_size).astype(np.int64)).tolist()

        self.voxel_size_x /= sample_rate[0]
        self.voxel_size_y /= sample_rate[1]
        self.voxel_size_z /= sample_rate[2]

        self.grid_size_x *= sample_rate[0]
        self.grid_size_y *= sample_rate[1]
        self.grid_size_z *= sample_rate[2]

        zs = torch.linspace(
            self.min_z + self.voxel_size_z / 2.,
            self.max_z - self.voxel_size_z / 2.,
            self.grid_size_z,
            dtype=torch.float32)
        ys = torch.linspace(
            self.min_y + self.voxel_size_y / 2.,
            self.max_y - self.voxel_size_y / 2.,
            self.grid_size_y,
            dtype=torch.float32)
        xs = torch.linspace(
            self.min_x + self.voxel_size_x / 2.,
            self.max_x - self.voxel_size_x / 2.,
            self.grid_size_x,
            dtype=torch.float32)
        zs, ys, xs = torch.meshgrid(zs, ys, xs)
        coordinates_3d = torch.stack([xs, ys, zs], dim=-1)
        self.coordinates_3d = coordinates_3d.float()

    def init_params(self):
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

    def pred_depth(self, stereo_depth_conv_module, mono_depth_conv_module,
                   cost1, mono_cost1, img_shape):
        cost1 = stereo_depth_conv_module(cost1)
        mono_cost1 = mono_depth_conv_module(mono_cost1)
        # (1, 2*D, H, W)
        cost = torch.cat((cost1, mono_cost1), dim=1).flatten(
            start_dim=1, end_dim=2)
        # (1, 1, D, H, W)
        weight = self.aggregate_cost(cost).unsqueeze(dim=1).sigmoid()
        cost = weight * cost1 + (1 - weight) * mono_cost1
        # (1, 1, 4D, 4H, 4W)
        cost = self.upsample_cost(cost)
        cost = torch.squeeze(cost, 1)
        cost_softmax = F.softmax(cost, dim=1)
        return cost, cost_softmax

    def get_local_depth(self, d_prob):
        with torch.no_grad():
            d = self.depth.cuda()[None, :, None, None]
            d_mul_p = d * d_prob
            local_window = 5
            p_local_sum = 0
            for off in range(0, local_window):
                cur_p = d_prob[:, off:off + d_prob.shape[1] - local_window + 1]
                p_local_sum += cur_p
            max_indices = p_local_sum.max(1, keepdim=True).indices
            pd_local_sum_for_max = 0
            for off in range(0, local_window):
                # d_prob[:, off:off + d_prob.shape[1] - local_window + 1]
                cur_pd = torch.gather(d_mul_p, 1, max_indices + off).squeeze(1)
                pd_local_sum_for_max += cur_pd
            mean_d = pd_local_sum_for_max / torch.gather(
                p_local_sum, 1, max_indices).squeeze(1)
        return mean_d

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
        cam2imgs = torch.as_tensor(
            [img_meta['cam2img'] for img_meta in img_metas],
            dtype=torch.float32,
            device=cur_stereo_feats.device)
        # cur2prevs: (B, N-1, 4, 4)
        cur2prevs = torch.stack(
            [img_meta['cur2prevs'] for img_meta in img_metas])
        batch_size = len(img_metas)

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
            img_metas[0]['ori_shape'],
            img_metas[0].get('flip', False),
            img_metas[0]['crop_offset'],
            img_scale_factor=img_metas[0]['scale_factor'][0])

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
        if not self.training:
            depth_preds_local = []
        depth_volumes = []
        # depth_samples = self.depth.clone().detach().cuda()
        # upsample the 3d volume and predict depth by pred_depth (trilinear)
        for idx in range(len(all_costs)):
            upcost_i, cost_softmax_i = self.pred_depth(
                self.pred_stereo[idx], self.pred_mono[idx], all_costs[idx],
                all_costs_mono[idx], img_metas[0]['pad_shape'])
            depth_volumes.append(upcost_i)
            if not self.training:
                depth_preds_local.append(self.get_local_depth(cost_softmax_i))

        # beginning of 3d detection part
        out = all_costs[-1]  # use stereo feature for detection
        out_prob = cost_softmax_i

        # convert plane-sweep into 3d volume
        coordinates_3d = self.coordinates_3d.cuda()
        norm_coord_imgs = []
        coord_imgs = []
        valids2d = []
        for i in range(batch_size):
            c3d = coordinates_3d.view(-1, 3)
            # in pseudo lidar coord
            c3d = project_pseudo_lidar_to_rectcam(c3d)
            coord_img = project_rect_to_image(c3d, cam2imgs[i].float().cuda())

            coord_img = torch.cat([coord_img, c3d[..., 2:]], dim=-1)
            coord_img = coord_img.view(*self.coordinates_3d.shape[:3], 3)

            coord_imgs.append(coord_img)

            # TODO: to modify for bs>1
            pad_shape = img_metas[0]['pad_shape']
            valid_mask_2d = (coord_img[..., 0] >= 0) & (coord_img[
                ..., 0] <= pad_shape[1]) & (coord_img[..., 1] >= 0) & (
                    coord_img[..., 1] <= pad_shape[0])
            valids2d.append(valid_mask_2d)

            # TODO: check whether the shape is right here
            crop_x1, crop_x2 = 0, pad_shape[1]
            crop_y1, crop_y2 = 0, pad_shape[0]
            norm_coord_img = (coord_img - torch.as_tensor(
                [crop_x1, crop_y1, self.depth_cfg['depth_min']],
                device=coord_img.device)) / torch.as_tensor(
                    [
                        crop_x2 - 1 - crop_x1, crop_y2 - 1 - crop_y1,
                        self.depth_cfg['depth_max'] -
                        self.depth_cfg['depth_min']
                    ],
                    device=coord_img.device)
            norm_coord_img = norm_coord_img * 2. - 1.
            norm_coord_imgs.append(norm_coord_img)
        norm_coord_imgs = torch.stack(norm_coord_imgs, dim=0)
        coord_imgs = torch.stack(coord_imgs, dim=0)
        valids2d = torch.stack(valids2d, dim=0)

        valids = valids2d & (norm_coord_imgs[..., 2] >= -1.) & (
            norm_coord_imgs[..., 2] <= 1.)
        valids = valids.float()

        # Retrieve Voxel Feature from Cost Volume Feature
        Voxel = F.grid_sample(out, norm_coord_imgs, align_corners=True)
        Voxel = Voxel * valids[:, None, :, :, :]

        if (self.stereo_atten_feat
                or (self.sem_atten_feat and self.cat_img_feature)):
            pred_disp = F.grid_sample(
                out_prob.detach()[:, None],
                norm_coord_imgs,
                align_corners=True)
            pred_disp = pred_disp * valids[:, None, :, :, :]

            if self.stereo_atten_feat:
                Voxel = Voxel * pred_disp

        # Retrieve Voxel Feature from 2D Img Feature
        if self.cat_img_feature:
            norm_coord_imgs_2d = norm_coord_imgs.clone().detach()
            norm_coord_imgs_2d[..., 2] = 0
            Voxel_2D = F.grid_sample(
                cur_sem_feats.unsqueeze(2),
                norm_coord_imgs_2d,
                align_corners=True)
            Voxel_2D = Voxel_2D * valids2d.float()[:, None, :, :, :]

            if self.sem_atten_feat:
                Voxel_2D = Voxel_2D * pred_disp

            if Voxel is not None:
                Voxel = torch.cat([Voxel, Voxel_2D], dim=1)
            else:
                Voxel = Voxel_2D

        # (1, 64, 20, 304, 288)
        Voxel = self.voxel_convs(Voxel)
        # volume_features_nopool = Voxel

        # (1, 32, 20, 304, 288)
        Voxel = self.voxel_pool(
            Voxel)  # [B, C, Nz, Ny, Nx] in cam (not img frustum) view

        # (1, 32, 5, 304, 288)
        volume_features = Voxel

        return volume_features


def project_pseudo_lidar_to_rectcam(pts_3d):
    xs, ys, zs = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    return torch.stack([-ys, -zs, xs], dim=-1)


def project_rectcam_to_pseudo_lidar(pts_3d):
    xs, ys, zs = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    return torch.stack([zs, -xs, -ys], dim=-1)


def project_rect_to_image(pts_3d_rect, P):
    n = pts_3d_rect.shape[0]
    ones = torch.ones((n, 1), device=pts_3d_rect.device)
    pts_3d_rect = torch.cat([pts_3d_rect, ones], dim=1)
    pts_2d = torch.mm(pts_3d_rect, torch.transpose(P, 0, 1))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def unproject_image_to_rect(pts_image, P):
    pts_3d = torch.cat(
        [pts_image[..., :2],
         torch.ones_like(pts_image[..., 2:3])], -1)
    pts_3d = pts_3d * pts_image[..., 2:3]
    pts_3d = torch.cat([pts_3d, torch.ones_like(pts_3d[..., 2:3])], -1)
    P4x4 = torch.eye(4, dtype=P.dtype, device=P.device)
    P4x4[:3, :] = P
    invP = torch.inverse(P4x4)
    pts_3d = torch.matmul(pts_3d, torch.transpose(invP, 0, 1))
    return pts_3d[..., :3]


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
