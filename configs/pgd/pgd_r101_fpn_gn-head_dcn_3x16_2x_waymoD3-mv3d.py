_base_ = [
    '../_base_/datasets/waymoD3-mv3d.py', '../_base_/models/pgd.py',
    '../_base_/schedules/mmdet_schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    neck=dict(num_outs=3),
    bbox_head=dict(
        num_classes=3,
        bbox_code_size=7,
        pred_attrs=False,
        pred_velo=False,
        pred_bbox2d=True,
        use_onlyreg_proj=True,
        strides=(8, 16, 32),
        regress_ranges=((-1, 128), (128, 256), (256, 1e8)),
        group_reg_dims=(2, 1, 3, 1, 16,
                        4),  # offset, depth, size, rot, kpts, bbox2d
        reg_branch=(
            (256, ),  # offset
            (256, ),  # depth
            (256, ),  # size
            (256, ),  # rot
            (256, ),  # kpts
            (256, )  # bbox2d
        ),
        centerness_branch=(256, ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        use_depth_classifier=True,
        depth_branch=(256, ),
        depth_range=(0, 50),
        depth_unit=10,
        division='uniform',
        depth_bins=6,
        pred_keypoints=True,
        weight_dim=1,
        loss_depth=dict(
            type='UncertainSmoothL1Loss', alpha=1.0, beta=3.0,
            loss_weight=1.0),
        loss_bbox2d=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.0),
        loss_consistency=dict(type='GIoULoss', loss_weight=0.0),
        bbox_coder=dict(
            type='PGDBBoxCoder',
            base_depths=((41.01, 18.44), ),
            base_dims=((0.91, 1.74, 0.84), (1.81, 1.77, 0.84), (4.73, 1.77,
                                                                2.08)),
            code_size=7)),
    # set weight 1.0 for base 7 dims (offset, depth, size, rot)
    # 0.2 for 16-dim keypoint offsets and 1.0 for 4-dim 2D distance targets
    train_cfg=dict(code_weight=[
        1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0
    ]),
    test_cfg=dict(nms_pre=100, nms_thr=0.05, score_thr=0.001, max_per_img=20))

class_names = ['Pedestrian', 'Cyclist', 'Car']
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
train_pipeline = [
    dict(type='LoadImageFromFileMono3D', file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize3D', img_scale=(1248, 832), ratio_range=(0.95, 1.05)),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d', 'gt_labels_3d',
            'centers2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=0.65,
        flip=False,
        transforms=[
            dict(type='Resize3D', img_scale=(1248, 832), keep_ratio=True),
            dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(pipeline=train_pipeline, classes=class_names, cam_sync=True),
    val=dict(pipeline=test_pipeline, classes=class_names, cam_sync=True),
    test=dict(pipeline=test_pipeline, classes=class_names, cam_sync=True))
# optimizer
optimizer = dict(
    lr=0.008, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=24)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
dist_params = dict(backend='nccl', port=29510)
