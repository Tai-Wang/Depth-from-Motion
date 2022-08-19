# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmdet.models.builder import NECKS


@NECKS.register_module()
class FrustumToVoxel(BaseModule):

    def __init__(
        self,
        num_3dconvs=1,  # num of 3d conv layers before hourglass
        cv_channels=32,  # cost volume channels
        out_channels=32,  # out volume channels after conv/pool
        in_sem_channels=32,
        sem_atten_feat=True,
        stereo_atten_feat=False,
        cat_img_feature=True,
        norm_cfg=dict(type='GN', num_groups=32,
                      requires_grad=True)  # use GN by default
    ):
        super(FrustumToVoxel, self).__init__()

        # general config
        self.GN = True  # TODO: replace it with norm_cfg

        # volume config
        self.num_3dconvs = num_3dconvs
        self.cv_channels = cv_channels
        self.out_channels = out_channels
        self.in_sem_channels = in_sem_channels

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
                    ConvModule(
                        voxel_channels if i == 0 else self.out_channels,
                        self.out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=norm_cfg)))

        self.voxel_convs = nn.Sequential(*voxel_convs)
        self.voxel_pool = nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1))
        self.init_weights()

    def forward(self,
                stereo_feat,
                stereo_feat_softmax,
                img_metas,
                cur_sem_feats=None):
        cam2imgs = torch.as_tensor(
            [img_meta['cam2img'] for img_meta in img_metas],
            dtype=torch.float32,
            device=stereo_feat.device)
        batch_size = len(img_metas)
        # stereo_feat as the root of voxel_feat sampling
        # stereo_feat_softmax as the attention mask for lifting 2D sem_feat
        # to 3D space
        # 1. convert plane-sweep into 3d volume
        coordinates_3d = self.coordinates_3d.cuda()
        norm_coord_imgs = []
        coord_imgs = []
        valids2d = []
        for i in range(batch_size):
            c3d = coordinates_3d.view(-1, 3)
            # in pseudo lidar coord
            c3d = project_pseudo_lidar_to_rectcam(c3d)
            # coord_img = project_rect_to_image(
            #     c3d, cam2imgs[i].float().cuda())
            coord_img = project_rect_to_image(c3d,
                                              cam2imgs[i][:3].float().cuda())

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

        # 2. Retrieve Voxel Feature from Cost Volume Feature
        Voxel = F.grid_sample(stereo_feat, norm_coord_imgs, align_corners=True)
        Voxel = Voxel * valids[:, None, :, :, :]

        if (self.stereo_atten_feat
                or (self.sem_atten_feat and self.cat_img_feature)):
            pred_disp = F.grid_sample(
                stereo_feat_softmax.detach(),
                norm_coord_imgs,
                align_corners=True)
            pred_disp = pred_disp * valids[:, None, :, :, :]

            if self.stereo_atten_feat:
                Voxel = Voxel * pred_disp

        # 3. Retrieve Voxel Feature from 2D Img Feature
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


def project_rect_to_image(pts_3d_rect, P):
    n = pts_3d_rect.shape[0]
    ones = torch.ones((n, 1), device=pts_3d_rect.device)
    pts_3d_rect = torch.cat([pts_3d_rect, ones], dim=1)
    pts_2d = torch.mm(pts_3d_rect, torch.transpose(P, 0, 1))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]
