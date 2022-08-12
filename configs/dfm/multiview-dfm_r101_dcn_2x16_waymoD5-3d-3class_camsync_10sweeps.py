model = dict(
    type='MultiViewDfM',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa
            '/mnt/lustre/wangtai/pretrained_models/resnet101-63fe2227.pth'),
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4),
    neck_2d=None,
    bbox_head_2d=None,
    backbone_stereo=None,
    depth_head=None,
    backbone_3d=None,
    neck_3d=dict(
        type='DfMNeck', in_channels=64, out_channels=256,
        num_frames=2),  # num_frames = num_ref_frames+1
    valid_sample=True,
    temporal_aggregate='concat',
    voxel_size=(0.5, 0.5, 0.5),  # n_voxels=[240, 300, 12]
    anchor_generator=dict(
        type='AlignedAnchor3DRangeGenerator',
        ranges=[[-35.0, -75.0, -2, 75.0, 75.0, 4]],
        rotations=[.0]),
    bbox_head_3d=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-35.0, -75.0, 0, 75.0, 75.0, 0],
                    [-35.0, -75.0, -0.1188, 75.0, 75.0, -0.1188],
                    [-35.0, -75.0, -0.0345, 75.0, 75.0, -0.0345]],
            sizes=[
                [0.91, 0.84, 1.74],  # pedestrian
                [1.81, 0.84, 1.77],  # cyclist
                [4.73, 2.08, 1.77]  # car
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1)
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.05,
        score_thr=0.001,
        min_bbox_size=0,
        nms_pre=500,
        max_num=100))

dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=False, use_camera=True)
point_cloud_range = [-35.0, -75.0, -2, 75.0, 75.0, 4]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        './data/waymo/':
        's3://openmmlab/datasets/detection3d/waymo/',
        'data/waymo/':
        's3://openmmlab/datasets/detection3d/waymo/'
    }))
# Explore RandomFlip3D Aug
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        file_client_args=file_client_args,
        num_views=5,
        num_ref_frames=1,
        test_mode=False),
    dict(type='MultiViewImagePhotoMetricDistortion'),
    dict(
        type='MultiViewImageResize3D',
        img_scale=(1248, 832),
        ratio_range=(0.95, 1.05),
        keep_ratio=True,
        multiscale_mode='range'),
    dict(type='MultiViewImageCrop3D', crop_size=(720, 1080)),
    dict(type='MultiViewRandomFlip3D', flip_ratio=0.5, direction='horizontal'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='MultiViewImageNormalize', **img_norm_cfg),
    dict(type='MultiViewImagePad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        file_client_args=file_client_args,
        num_views=5,
        num_ref_frames=1,
        test_mode=True),
    dict(
        type='MultiViewImageResize3D',
        img_scale=(1248, 832),
        keep_ratio=True,
        multiscale_mode='range'),
    dict(type='MultiViewImageNormalize', **img_norm_cfg),
    dict(type='MultiViewImagePad', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_train.pkl',
        split='training',
        classes=class_names,
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR',
        load_interval=5,
        load_mode='lidar_frame',
        max_sweeps=10,
        cam_sync=True,
        file_client_args=file_client_args),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_val.pkl',
        split='training',
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        load_mode='lidar_frame',
        max_sweeps=10,
        cam_sync=True,
        file_client_args=file_client_args),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_val.pkl',
        split='training',
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        load_mode='lidar_frame',
        max_sweeps=10,
        cam_sync=True,
        file_client_args=file_client_args))

optimizer = dict(
    type='AdamW',
    lr=0.0005,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))
# although grad_clip is set in original code, it is not used
# optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[16, 22])
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
evaluation = dict(interval=24)
dist_params = dict(backend='nccl')
find_unused_parameters = True  # only 1 of 4 FPN outputs is used
log_level = 'INFO'
# load_from = '/mnt/lustre/wangtai/mmdet3d-DfM/work_dirs/2x16-D3-pgd-mv3d/epoch_24.pth'  # noqa
# load_from = '/mnt/lustre/wangtai/mmdet3d-DfM/work_dirs/3x16-pgd-fixed-baseline/epoch_24.pth'  # noqa
load_from = '/mnt/lustre/wangtai/mmdet3d-DfM/work_dirs/3x16-pgd-fixed-baseline-full-smallimg/epoch_24.pth'  # noqa
resume_from = None
workflow = [('train', 1)]
