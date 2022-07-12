# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet3d.core import bbox3d2result
from mmdet.models.detectors import BaseDetector
from ..builder import DETECTORS, build_backbone, build_head, build_neck


@DETECTORS.register_module()
class DfM(BaseDetector):
    """Monocular 3D Object Detection with Depth from Motion."""

    def __init__(self,
                 backbone,
                 neck,
                 backbone_stereo,
                 backbone_3d,
                 bbox_head_3d,
                 neck_2d=None,
                 neck_3d=None,
                 feature_transformation=None,
                 bbox_head_2d=None,
                 depth_head_2d=None,
                 depth_head=None,
                 depth_cfg=None,
                 voxel_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        if backbone_stereo is not None:
            if depth_cfg is not None:
                backbone_stereo.update(depth_cfg=depth_cfg)
            self.backbone_stereo = build_backbone(backbone_stereo)
        if backbone_3d is not None:
            self.backbone_3d = build_backbone(backbone_3d)
        if neck_3d is not None:
            self.neck_3d = build_neck(neck_3d)
        if neck_2d is not None:
            self.neck_2d = build_neck(neck_2d)
        if feature_transformation is not None:
            feature_transformation.update(
                cat_img_feature=self.neck.cat_img_feature)
            feature_transformation.update(
                in_sem_channels=self.neck.sem_channels[-1])
            self.feature_transformation = build_neck(feature_transformation)
            assert self.neck.cat_img_feature == \
                self.feature_transformation.cat_img_feature
            assert self.neck.sem_channels[
                -1] == self.feature_transformation.in_sem_channels
        if bbox_head_2d is not None:
            self.bbox_head_2d = build_head(bbox_head_2d)
        if depth_head_2d is not None:
            self.depth_head_2d = build_head(depth_head_2d)
        if depth_head is not None:
            self.depth_head = build_head(depth_head)
        if depth_cfg is not None:
            # TODO: only use default value
            self.downsampled_depth_offset = 0.5
            self.downsample_factor = depth_cfg['downsample_factor']
            self.prepare_depth(depth_cfg)
            # TODO: remove the param dependency
            if feature_transformation is not None:
                self.feature_transformation.depth_cfg = depth_cfg
            if depth_head is not None:
                self.backbone_stereo.downsampled_depth = \
                    self.downsampled_depth
                self.depth_head.depth_samples = self.depth
                self.depth_head.downsample_factor = self.downsample_factor
                assert self.downsample_factor == \
                    self.depth_head.downsample_factor
        if voxel_cfg is not None:
            self.prepare_coordinates_3d(voxel_cfg)
            # TODO: remove the param dependency
            if feature_transformation is not None:
                self.feature_transformation.coordinates_3d = \
                    self.coordinates_3d
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        bbox_head_3d.update(train_cfg=train_cfg)
        bbox_head_3d.update(test_cfg=test_cfg)
        self.bbox_head_3d = build_head(bbox_head_3d)

    @property
    def with_backbone_3d(self):
        return hasattr(self, 'backbone_3d') and self.backbone_3d is not None

    @property
    def with_neck_3d(self):
        return hasattr(self, 'neck_3d') and self.neck_3d is not None

    @property
    def with_feature_transformation(self):
        return hasattr(self, 'feature_transformation') and \
            self.feature_transformation is not None

    @property
    def with_neck_2d(self):
        return hasattr(self, 'neck_2d') and self.neck_2d is not None

    @property
    def with_bbox_head_2d(self):
        return hasattr(self, 'bbox_head_2d') and self.bbox_head_2d is not None

    @property
    def with_depth_head_2d(self):
        return hasattr(self,
                       'depth_head_2d') and self.depth_head_2d is not None

    @property
    def with_depth_head(self):
        return hasattr(self, 'depth_head') and self.depth_head is not None

    def prepare_depth(self, depth_cfg):
        # TODO: support different depth division
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

    def extract_feat(self, img, img_metas):
        """
        Args:
            img (torch.Tensor): [B, N, C_in, H, W]
            img_metas (list): each element corresponds to a group of images.
                len(img_metas) == B.

        Returns:
            torch.Tensor: bev feature with shape [B, C_out, N_y, N_x].
        """
        # split input img into current and previous ones
        batch_size, N, C_in, H, W = img.shape
        cur_imgs = img[:, 0]
        prev_imgs = img[:, 1]  # TODO: to support multiple prev imgs
        # 2D backbone for feature extraction
        cur_feats = self.backbone(cur_imgs)
        cur_feats = [cur_imgs] + list(cur_feats)
        prev_feats = self.backbone(prev_imgs)
        prev_feats = [prev_imgs] + list(prev_feats)
        # SPP module as the feature neck
        cur_stereo_feat, cur_sem_feat = self.neck(cur_feats)
        prev_stereo_feat, prev_sem_feat = self.neck(prev_feats)
        # derive cur2prevs
        cur_pose = torch.tensor(
            [img_meta['cam2global'] for img_meta in img_metas],
            device=img.device)[:, None, :, :]  # (B, 1, 4, 4)
        prev_poses = []
        for img_meta in img_metas:
            sweep_img_metas = img_meta['sweep_img_metas']
            prev_poses.append([
                sweep_img_meta['cam2global']
                for sweep_img_meta in sweep_img_metas
            ])
        prev_poses = torch.tensor(prev_poses, device=img.device)
        pad_prev_cam2global = torch.eye(4)[None, None].expand(
            batch_size, N - 1, 4, 4).to(img.device)
        pad_prev_cam2global[:, :, :prev_poses.shape[-2], :prev_poses.
                            shape[-1]] = prev_poses
        pad_cur_cam2global = torch.eye(4)[None,
                                          None].expand(batch_size, 1, 4,
                                                       4).to(img.device)
        pad_cur_cam2global[:, :, :cur_pose.shape[-2], :cur_pose.
                           shape[-1]] = cur_pose
        # (B, N-1, 4, 4) * (B, 1, 4, 4) -> (B, N-1, 4, 4)
        # torch.linalg.solve is faster and more numerically stable
        # than torch.matmul(torch.linalg.inv(A), B)
        # empirical results show that torch.linalg.solve can derive
        # almost the same result with np.linalg.inv
        # while torch.linalg.inv can not
        cur2prevs = torch.linalg.solve(pad_prev_cam2global, pad_cur_cam2global)
        for meta_idx, img_meta in enumerate(img_metas):
            img_meta['cur2prevs'] = cur2prevs[meta_idx]
        # stereo backbone for depth estimation
        # stereo_feat: (batch_size, C, D, H, W)
        mono_stereo_costs, stereo_feats, mono_feats = self.backbone_stereo(
            cur_stereo_feat, prev_stereo_feat, img_metas)
        return mono_stereo_costs, stereo_feats, mono_feats, cur_sem_feat

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      depth_img=None,
                      depth_fgmask_img=None,
                      **kwargs):
        mono_stereo_costs, stereo_feats, mono_feats, cur_sem_feat = \
            self.extract_feat(img, img_metas)

        upsample_costs, upsample_costs_softmax, depth_preds = self.depth_head(
            mono_stereo_costs)
        # TODO: try to use mono_stereo_costs/feats to replace stereo_feats
        volume_feat = self.feature_transformation(stereo_feats,
                                                  upsample_costs_softmax,
                                                  img_metas, cur_sem_feat)
        # height compression
        _, Cv, Nz, Ny, Nx = volume_feat.shape
        bev_feat = volume_feat.view(-1, Cv * Nz, Ny, Nx)
        bev_feat_prehg, bev_feat = self.backbone_3d(bev_feat)
        # bbox_head takes a list of feature from different levels as input
        # so need [bev_feat]
        outs = self.bbox_head_3d([bev_feat])
        losses = self.bbox_head_3d.loss(*outs, gt_bboxes_3d, gt_labels_3d,
                                        img_metas)
        # TODO: loss_dense_depth, loss_2d, loss_imitation
        if self.with_depth_head and depth_img is not None:
            loss_depth = self.depth_head.loss(
                depth_preds.flatten(start_dim=0, end_dim=1),
                upsample_costs.flatten(start_dim=0, end_dim=1),
                depth_img.flatten(start_dim=0, end_dim=1),
                depth_fgmask_img=depth_fgmask_img.flatten(
                    start_dim=0, end_dim=1))
            losses['loss_dense_depth'] = loss_depth
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """Forward of testing.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.
        Returns:
            list[dict]: Predicted 3d boxes.
        """
        # not supporting aug_test for now
        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas):
        mono_stereo_costs, stereo_feats, mono_feats, cur_sem_feat = \
            self.extract_feat(img, img_metas)

        upsample_costs, upsample_costs_softmax, depth_preds = self.depth_head(
            mono_stereo_costs)
        # TODO: try to use mono_stereo_costs/feats to replace stereo_feats
        volume_feat = self.feature_transformation(stereo_feats,
                                                  upsample_costs_softmax,
                                                  img_metas, cur_sem_feat)
        # height compression
        _, Cv, Nz, Ny, Nx = volume_feat.shape
        bev_feat = volume_feat.view(-1, Cv * Nz, Ny, Nx)
        bev_feat_prehg, bev_feat = self.backbone_3d(bev_feat)
        # bbox_head takes a list of feature from different levels as input
        # so need [bev_feat]
        outs = self.bbox_head_3d([bev_feat])
        bbox_list = self.bbox_head_3d.get_bboxes(*outs, img_metas)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        # add pseudo-lidar label to each pred_dict for post-processing
        for bbox_result in bbox_results:
            bbox_result['pseudo_lidar'] = True
        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            imgs (list[torch.Tensor]): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
