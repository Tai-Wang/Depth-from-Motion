# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet3d.core import bbox3d2result, build_prior_generator
from mmdet3d.models.dense_heads import CenterHead
from mmdet3d.models.fusion_layers.point_fusion import (point_sample,
                                                       voxel_sample)
from ..builder import DETECTORS
from .dfm import DfM


@DETECTORS.register_module()
class MultiViewDfM(DfM):
    """Monocular 3D Object Detection with Depth from Motion."""

    def __init__(self,
                 backbone,
                 neck,
                 backbone_stereo,
                 backbone_3d,
                 neck_3d,
                 bbox_head_3d,
                 voxel_size,
                 anchor_generator,
                 neck_2d=None,
                 bbox_head_2d=None,
                 depth_head_2d=None,
                 depth_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 valid_sample=True,
                 temporal_aggregate='mean',
                 transform_depth=True,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            backbone_stereo=backbone_stereo,
            backbone_3d=backbone_3d,
            neck_3d=neck_3d,
            bbox_head_3d=bbox_head_3d,
            neck_2d=neck_2d,
            bbox_head_2d=bbox_head_2d,
            depth_head_2d=depth_head_2d,
            depth_head=depth_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.voxel_size = voxel_size
        self.voxel_range = anchor_generator['ranges'][0]
        self.n_voxels = [
            round((self.voxel_range[3] - self.voxel_range[0]) /
                  self.voxel_size[0]),
            round((self.voxel_range[4] - self.voxel_range[1]) /
                  self.voxel_size[1]),
            round((self.voxel_range[5] - self.voxel_range[2]) /
                  self.voxel_size[2])
        ]
        self.anchor_generator = build_prior_generator(anchor_generator)
        self.valid_sample = valid_sample
        self.temporal_aggregate = temporal_aggregate
        self.transform_depth = transform_depth

    def extract_feat(self, img, img_metas):
        """
        Args:
            img (torch.Tensor): [B, Nv, C_in, H, W]
            img_metas (list): each element corresponds to a group of images.
                len(img_metas) == B.

        Returns:
            torch.Tensor: bev feature with shape [B, C_out, N_y, N_x].
        """
        # TODO: Nt means the number of frames temporally
        # num_views means the number of views of a frame
        batch_size, _, C_in, H, W = img.shape
        num_views = img_metas[0]['num_views']
        num_ref_frames = img_metas[0]['num_ref_frames']
        if num_ref_frames > 0:
            num_frames = num_ref_frames + 1
        else:
            num_frames = 1
        input_shape = img.shape[-2:]
        # NOTE: input_shape is the largest pad_shape of the batch of images
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)
        if num_ref_frames > 0:
            cur_imgs = img[:, :num_views].reshape(-1, C_in, H, W)
            prev_imgs = img[:, num_views:].reshape(-1, C_in, H, W)
            cur_feats = self.backbone(cur_imgs)
            cur_feats = self.neck(cur_feats)[0]
            with torch.no_grad():
                prev_feats = self.backbone(prev_imgs)
                prev_feats = self.neck(prev_feats)[0]
            _, C_feat, H_feat, W_feat = cur_feats.shape
            cur_feats = cur_feats.view(batch_size, -1, C_feat, H_feat, W_feat)
            prev_feats = prev_feats.view(batch_size, -1, C_feat, H_feat,
                                         W_feat)
            batch_feats = torch.cat([cur_feats, prev_feats], dim=1)
        else:
            batch_imgs = img.view(-1, C_in, H, W)
            batch_feats = self.backbone(batch_imgs)
            # TODO: support SPP module neck
            batch_feats = self.neck(batch_feats)[0]
            _, C_feat, H_feat, W_feat = batch_feats.shape
            batch_feats = batch_feats.view(batch_size, -1, C_feat, H_feat,
                                           W_feat)
        # transform the feature to voxel & stereo space
        transform_feats = self.feature_transformation(batch_feats, img_metas,
                                                      num_views, num_frames)
        if self.with_depth_head_2d:
            transform_feats += (batch_feats[:, :num_views], )

        return transform_feats

    def feature_transformation(self, batch_feats, img_metas, num_views,
                               num_frames):
        # TODO: support more complicated 2D feature sampling
        points = self.anchor_generator.grid_anchors(
            [self.n_voxels[::-1]], device=batch_feats.device)[0][:, :3]
        volumes = []
        img_scale_factors = []
        img_flips = []
        img_crop_offsets = []
        for feature, img_meta in zip(batch_feats, img_metas):
            if 'scale_factor' in img_meta:
                if isinstance(
                        img_meta['scale_factor'],
                        np.ndarray) and len(img_meta['scale_factor']) >= 2:
                    img_scale_factor = (
                        points.new_tensor(img_meta['scale_factor'][:2]))
                else:
                    img_scale_factor = (
                        points.new_tensor(img_meta['scale_factor']))
            else:
                img_scale_factor = (1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (
                points.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            img_scale_factors.append(img_scale_factor)
            img_flips.append(img_flip)
            img_crop_offsets.append(img_crop_offset)
            # TODO: remove feature sampling from back
            # TODO: support different scale_factors/flip/crop_offset for
            # different views
            frame_volume = []
            frame_valid_nums = []
            for frame_idx in range(num_frames):
                volume = []
                valid_flags = []
                for view_idx in range(num_views):
                    sample_idx = frame_idx * num_views + view_idx
                    sample_results = point_sample(
                        img_meta,
                        img_features=feature[sample_idx][None, ...],
                        points=points,
                        proj_mat=points.new_tensor(
                            img_meta['ori_lidar2img'][sample_idx]),
                        coord_type='LIDAR',
                        img_scale_factor=img_scale_factor,
                        img_crop_offset=img_crop_offset,
                        img_flip=img_flip,
                        img_pad_shape=img_meta['input_shape'],
                        img_shape=img_meta['img_shape'][sample_idx][:2],
                        aligned=False,
                        valid_flag=self.valid_sample)
                    if self.valid_sample:
                        volume.append(sample_results[0])
                        valid_flags.append(sample_results[1])
                    else:
                        volume.append(sample_results)
                    # TODO: save valid flags, more reasonable feat fusion
                if self.valid_sample:
                    valid_nums = torch.stack(
                        valid_flags, dim=0).sum(0)  # (N, )
                    volume = torch.stack(volume, dim=0).sum(0)
                    valid_mask = valid_nums > 0
                    volume[~valid_mask] = 0
                    frame_valid_nums.append(valid_nums)
                else:
                    volume = torch.stack(volume, dim=0).mean(0)
                frame_volume.append(volume)
            if self.valid_sample:
                if self.temporal_aggregate == 'mean':
                    frame_volume = torch.stack(frame_volume, dim=0).sum(0)
                    frame_valid_nums = torch.stack(
                        frame_valid_nums, dim=0).sum(0)
                    frame_valid_mask = frame_valid_nums > 0
                    frame_volume[~frame_valid_mask] = 0
                    frame_volume = frame_volume / torch.clamp(
                        frame_valid_nums[:, None], min=1)
                elif self.temporal_aggregate == 'concat':
                    frame_valid_nums = torch.stack(frame_valid_nums, dim=1)
                    frame_volume = torch.stack(frame_volume, dim=1)
                    frame_valid_mask = frame_valid_nums > 0
                    frame_volume[~frame_valid_mask] = 0
                    frame_volume = (frame_volume / torch.clamp(
                        frame_valid_nums[:, :, None], min=1)).flatten(
                            start_dim=1, end_dim=2)
            else:
                frame_volume = torch.stack(frame_volume, dim=0).mean(0)
            volumes.append(
                frame_volume.reshape(self.n_voxels[::-1] + [-1]).permute(
                    3, 2, 1, 0))
        volume_feat = torch.stack(volumes)  # (B, C, N_x, N_y, N_z)
        if self.with_backbone_3d:
            outputs = self.backbone_3d(volume_feat)
            volume_feat = outputs[0]
            if self.backbone_3d.output_bev:
                # use outputs[0] if len(outputs) == 1
                # use outputs[1] if len(outputs) == 2
                # TODO: unify the output formats
                bev_feat = outputs[-1]
        # grid_sample stereo features from the volume feature
        # TODO: also support temporal modeling for depth head
        if self.with_depth_head:
            batch_stereo_feats = []
            for batch_idx in range(volume_feat.shape[0]):
                stereo_feat = []
                for view_idx in range(num_views):
                    img_scale_factor = img_scale_factors[batch_idx] \
                        if self.transform_depth else points.new_tensor(
                            [1., 1.])
                    img_crop_offset = img_crop_offsets[batch_idx] \
                        if self.transform_depth else points.new_tensor(
                            [0., 0.])
                    img_flip = img_flips[batch_idx] if self.transform_depth \
                        else False
                    img_pad_shape = img_meta['input_shape'] \
                        if self.transform_depth else img_meta['ori_shape'][:2]
                    stereo_feat.append(
                        voxel_sample(
                            volume_feat[batch_idx][None],
                            voxel_range=self.voxel_range,
                            voxel_size=self.voxel_size,
                            depth_samples=volume_feat.new_tensor(
                                self.depth_samples),
                            proj_mat=points.new_tensor(
                                img_metas[batch_idx]['ori_lidar2img']
                                [view_idx]),
                            downsample_factor=self.depth_head.
                            downsample_factor,
                            img_scale_factor=img_scale_factor,
                            img_crop_offset=img_crop_offset,
                            img_flip=img_flip,
                            img_pad_shape=img_pad_shape,
                            img_shape=img_metas[batch_idx]['img_shape']
                            [view_idx][:2],
                            aligned=True))  # TODO: study the aligned setting
                batch_stereo_feats.append(torch.cat(stereo_feat))
            # cat (N, C, D, H, W) -> (B*N, C, D, H, W)
            batch_stereo_feats = torch.cat(batch_stereo_feats)
        if self.with_neck_3d:
            if self.with_backbone_3d and self.backbone_3d.output_bev:
                spatial_features = self.neck_3d(bev_feat)
                # TODO: unify the outputs of neck_3d
                volume_feat = spatial_features[1]
            else:
                volume_feat = self.neck_3d(volume_feat)[0]
        # TODO: unify the output format of neck_3d
        transform_feats = (volume_feat, )
        if self.with_depth_head:
            transform_feats += (batch_stereo_feats, )
        return transform_feats

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      depth_img=None,
                      **kwargs):
        feats = self.extract_feat(img, img_metas)
        bev_feat = feats[0]
        outs = self.bbox_head_3d([bev_feat])
        if not isinstance(self.bbox_head_3d, CenterHead):
            losses = self.bbox_head_3d.loss(*outs, gt_bboxes_3d, gt_labels_3d,
                                            img_metas)
        else:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.bbox_head_3d.loss(*loss_inputs)
        if self.with_depth_head_2d:
            fv_feat = feats[-1]
            depth_volumes, _, depth_preds = self.depth_head_2d(
                fv_feat.flatten(start_dim=0, end_dim=1))
            loss_depth_2d = self.depth_head_2d.loss(
                depth_preds.flatten(start_dim=0, end_dim=1),
                depth_volumes.flatten(start_dim=0, end_dim=1),
                depth_img.flatten(start_dim=0, end_dim=1),
                depth_fgmask_imgs=None)
            losses['loss_dense_depth_2d'] = loss_depth_2d
        if self.with_depth_head:
            stereo_feat = feats[1]
            depth_volumes, _, depth_preds = self.depth_head(stereo_feat)
            loss_depth = self.depth_head.loss(
                depth_preds.flatten(start_dim=0, end_dim=1),
                depth_volumes.flatten(start_dim=0, end_dim=1),
                depth_img.flatten(start_dim=0, end_dim=1),
                depth_fgmask_imgs=None)
            losses['loss_dense_depth'] = loss_depth
        # TODO: loss_dense_depth, loss_2d, loss_imitation
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
        feats = self.extract_feat(img, img_metas)
        # bbox_head takes a list of feature from different levels as input
        # so need [bev_feat]
        bev_feat = feats[0]
        outs = self.bbox_head_3d([bev_feat])
        """
        if self.with_depth_head:
            stereo_feat = feats[1]
            depth_volumes, _, depth_preds = self.depth_head(stereo_feat)
        """
        if not isinstance(self.bbox_head_3d, CenterHead):
            bbox_list = self.bbox_head_3d.get_bboxes(*outs, img_metas)
        else:
            bbox_list = self.bbox_head_3d.get_bboxes(
                outs, img_metas, rescale=False)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
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
