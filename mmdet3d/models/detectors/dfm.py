# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.ops.points_in_boxes import points_in_boxes_part

from mmdet3d.core import bbox3d2result
from mmdet.models.detectors import BaseDetector
from ..builder import (DETECTORS, build_backbone, build_detector, build_head,
                       build_neck)
from ..dense_heads import LIGAATSSHead
from ..utils.common_utils import dist_reduce_mean
from .imitation_utils import NormalizeLayer, WeightedL2WithSigmaLoss


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
                 lidar_model=None,
                 depth_cfg=None,
                 voxel_cfg=None,
                 normalizer_clamp_value=10,
                 imitation_cfgs=None,
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
        self.imitation_cfgs = imitation_cfgs
        if lidar_model is not None and imitation_cfgs is not None:
            self.lidar_model = build_detector(lidar_model)
            for param in self.lidar_model.parameters():
                param.requires_grad_(False)
            self._init_imitation_layers()
            # TODO: replace this hard-code declaration
            self.loss_imitation = WeightedL2WithSigmaLoss()
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
        self.normalizer_clamp_value = normalizer_clamp_value
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        bbox_head_3d.update(train_cfg=train_cfg)
        bbox_head_3d.update(test_cfg=test_cfg)
        # TODO: remove this hack
        if bbox_head_3d['type'] == 'LIGAAnchor3DHead':
            bbox_head_3d.update(normalizer_clamp_value=normalizer_clamp_value)
        self.bbox_head_3d = build_head(bbox_head_3d)

    def train(self, mode=True):
        super(DfM, self).train(mode)
        # set lidar_model to eval mode by default
        if self.with_lidar_model:
            for m in self.lidar_model.modules():
                m.eval()

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

    @property
    def with_lidar_model(self):
        return hasattr(self, 'lidar_model') and self.lidar_model is not None

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

    def _init_imitation_layers(self):
        if self.imitation_cfgs is not None:
            cfgs = self.imitation_cfgs if isinstance(
                self.imitation_cfgs, list) else [self.imitation_cfgs]

            conv_imitation_layers = []
            self.norm_imitation = nn.ModuleDict()
            for cfg in cfgs:
                layers = []
                if cfg['layer'] == 'conv2d':
                    layers.append(
                        nn.Conv2d(
                            cfg['channel'],
                            cfg['channel'],
                            kernel_size=cfg['kernel_size'],
                            padding=cfg['kernel_size'] // 2,
                            stride=1,
                            groups=1))
                elif cfg['layer'] == 'conv3d':
                    layers.append(
                        nn.Conv3d(
                            cfg['channel'],
                            cfg['channel'],
                            kernel_size=cfg['kernel_size'],
                            padding=cfg['kernel_size'] // 2,
                            stride=1,
                            groups=1))
                else:
                    assert cfg['layer'] == 'none', \
                        f"invalid layer type {cfg['layer']}"
                if cfg['use_relu']:
                    layers.append(nn.ReLU())
                    assert cfg['normalize'] is None

                if cfg['normalize'] is not None:
                    self.norm_imitation[cfg['stereo_feature_layer']] = \
                        NormalizeLayer(cfg['normalize'], cfg['channel'])
                else:
                    self.norm_imitation[cfg['stereo_feature_layer']] = \
                        nn.Identity()

                if len(layers) <= 1:
                    conv_imitation_layers.append(layers[0])
                else:
                    conv_imitation_layers.append(nn.Sequential(*layers))

            if len(cfgs) > 1:
                self.conv_imitation = nn.ModuleList(conv_imitation_layers)
            else:
                self.conv_imitation = conv_imitation_layers[0]

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
        # torch version seems to be unstable than numpy
        cur2prevs = torch.tensor(
            [img_meta['cur2prevs'] for img_meta in img_metas],
            device=img.device,
            dtype=img.dtype)  # (B, N-1, 4, 4)
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
                      gt_bboxes=None,
                      depth_img=None,
                      depth_fgmask_img=None,
                      points=None,
                      centers2d=None,
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
        if self.with_bbox_head_2d and gt_bboxes is not None:
            if isinstance(self.bbox_head_2d, LIGAATSSHead):
                if self.bbox_head_2d.assigner.append_3d_centers:
                    for sample_idx in range(len(gt_bboxes)):
                        gt_bboxes[sample_idx] = torch.cat(
                            [gt_bboxes[sample_idx], centers2d[sample_idx]],
                            dim=-1)
            sem_feat_2d = self.neck_2d([cur_sem_feat])
            outs_2d = self.bbox_head_2d(sem_feat_2d)
            losses_bbox2d = self.bbox_head_2d.loss(
                *outs_2d,
                gt_bboxes,
                gt_labels=gt_labels_3d,
                img_metas=img_metas)
            for key in losses_bbox2d.keys():
                losses_bbox2d[key] = sum(losses_bbox2d[key])
            loss_bbox2d = sum([v for _, v in losses_bbox2d.items()])
            losses['loss_bbox2d'] = loss_bbox2d
        if self.with_depth_head and depth_img is not None:
            if depth_fgmask_img is not None:
                depth_fgmask_img = depth_fgmask_img.flatten(
                    start_dim=0, end_dim=1)
            loss_depth = self.depth_head.loss(
                depth_preds.flatten(start_dim=0, end_dim=1),
                upsample_costs.flatten(start_dim=0, end_dim=1),
                depth_img.flatten(start_dim=0, end_dim=1),
                depth_fgmask_img=depth_fgmask_img)
            losses['loss_dense_depth'] = loss_depth
        if self.with_lidar_model:
            lidar_volume_feat, lidar_bev_feat = self.extract_lidar_model_feat(
                points)
            raw_features = dict(
                lidar_features=dict(
                    spatial_features_2d=lidar_bev_feat,
                    volume_features=lidar_volume_feat),
                stereo_features=dict(
                    spatial_features_2d=bev_feat, volume_features=volume_feat))
            feature_pairs = self.construct_feature_pairs(raw_features)
            # TODO: construct feature pairs
            losses['loss_imitation'] = self.imitation_loss(
                feature_pairs, gt_bboxes_3d)
        return losses

    def extract_lidar_model_feat(self, points):
        """Extract features from points."""
        voxels, num_points, coors = self.lidar_model.voxelize(points)
        voxel_features = self.lidar_model.voxel_encoder(
            voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        spatial_feat, volume_feat = self.lidar_model.middle_encoder(
            voxel_features, coors, batch_size)
        bev_feat = self.lidar_model.backbone(spatial_feat)
        return volume_feat, bev_feat

    def construct_feature_pairs(self, raw_features):
        if self.imitation_cfgs is not None and self.training:
            imitation_features_pairs = []
            imitation_conv_layers = [self.conv_imitation] if len(
                self.imitation_cfgs) == 1 else self.conv_imitation
            for cfg, imitation_conv in zip(self.imitation_cfgs,
                                           imitation_conv_layers):
                lidar_feature_name = cfg['lidar_feature_layer']
                stereo_feature_name = cfg['stereo_feature_layer']
                imitation_features_pairs.append(
                    dict(
                        config=cfg,
                        stereo_feature_name=stereo_feature_name,
                        lidar_feature_name=lidar_feature_name,
                        gt=raw_features['lidar_features'][lidar_feature_name],
                        pred=imitation_conv(raw_features['stereo_features']
                                            [stereo_feature_name])))
        return imitation_features_pairs

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

    def imitation_loss(self, feature_pairs, gt_bboxes_3d):
        imitation_loss_list = []
        for feature_pair in feature_pairs:
            features_preds = feature_pair['pred']
            features_targets = feature_pair['gt']
            imitation_loss, _ = self.get_imitation_reg_layer_loss(
                features_preds=features_preds,
                features_targets=features_targets,
                imitation_cfg=feature_pair['config'],
                gt_bboxes_3d=gt_bboxes_3d)
            imitation_loss_list.append(imitation_loss)
        return imitation_loss_list

    def get_imitation_reg_layer_loss(self, features_preds, features_targets,
                                     imitation_cfg, gt_bboxes_3d):
        # TODO: check all the self attributes
        batch_size = int(features_preds.shape[0])
        features_preds = features_preds.permute(
            0, *range(2, len(features_preds.shape)), 1)
        features_targets = features_targets.permute(
            0, *range(2, len(features_targets.shape)), 1)

        if imitation_cfg['mode'] == 'inbox':
            anchors_xyz = self.bbox_head_3d.anchors[0][:, :, :, 0,
                                                       0, :3].clone()
            gt_boxes = torch.stack([
                gt_bbox_3d.tensor[..., :7].clone().to(features_preds.device)
                for gt_bbox_3d in gt_bboxes_3d
            ],
                                   dim=0)
            anchors_xyz[..., 2] = 0
            gt_boxes[..., 2] = 0
            positives = points_in_boxes_part(
                anchors_xyz.view(anchors_xyz.shape[0], -1, 3), gt_boxes)
            # output -1 by default for background
            positives = (positives >= 0).view(*anchors_xyz.shape[:3])
        elif imitation_cfg['mode'] == 'full':
            positives = features_preds.new_ones(*features_preds.shape[:3])
            if dist.get_rank() == 0:
                print('using full imitation mask')
        else:
            raise ValueError('wrong imitation mode')

        if len(features_targets.shape) == 5:
            # 3d feature
            positives = positives.unsqueeze(1).repeat(
                1, features_targets.shape[1], 1, 1)
        else:
            assert len(features_targets.shape) == 4

        positives = positives & torch.any(features_targets != 0, dim=-1)

        reg_weights = positives.float()
        pos_normalizer = positives.sum().float()
        pos_normalizer = dist_reduce_mean(pos_normalizer.mean())
        reg_weights /= torch.clamp(
            pos_normalizer, min=self.normalizer_clamp_value)

        pos_inds = reg_weights > 0
        pos_feature_preds = features_preds[pos_inds]
        pos_feature_targets = self.norm_imitation[
            imitation_cfg['stereo_feature_layer']](
                features_targets[pos_inds])

        imitation_loss_src = self.loss_imitation(
            pos_feature_preds,
            pos_feature_targets,
            weights=reg_weights[pos_inds])  # [N, M]
        imitation_loss_src = imitation_loss_src.mean(-1)

        imitation_loss = imitation_loss_src.sum() / batch_size
        imitation_loss = imitation_loss * imitation_cfg['loss_weight']

        tb_dict = {
            'rpn_loss_imitation': imitation_loss.item(),
        }

        if pos_inds.sum() > 0:
            rel_err = torch.median(
                torch.abs((pos_feature_preds - pos_feature_targets) /
                          pos_feature_targets))
            tb_dict['rel_err_imitation_feature'] = rel_err.item()
        else:
            tb_dict['rel_err_imitation_feature'] = 0.

        return imitation_loss, tb_dict
