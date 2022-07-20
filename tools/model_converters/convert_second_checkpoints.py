# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from mmcv import Config
from mmcv.runner import load_state_dict

from mmdet3d.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D version of SECOND teacher checkpoint')
    parser.add_argument('config', help='path of the config file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='path of the output checkpoint file')
    args = parser.parse_args()
    return args


def main():
    """Convert SECOND teacher checkpoints to mmdet3d-style."""
    args = parse_args()
    checkpoint = torch.load(args.checkpoint)
    cfg = Config.fromfile(args.config)
    # Build the model and load checkpoint
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    ori_ckpt = checkpoint['model_state']
    converted_ckpt = ori_ckpt.copy()

    RENAME_PREFIX = {
        'backbone_3d.conv1': 'middle_encoder.encoder_layers.encoder_layer1',
        'backbone_3d.conv2': 'middle_encoder.encoder_layers.encoder_layer2',
        'backbone_3d.conv3': 'middle_encoder.encoder_layers.encoder_layer3',
        'backbone_3d.conv4': 'middle_encoder.encoder_layers.encoder_layer4',
        'backbone_3d': 'middle_encoder',
        'backbone_2d.rpn3d_conv2': 'backbone.compress_conv',
        'backbone_2d.rpn3d_conv3': 'backbone.bev_hourglass',
        'dense_head.rpn3d_cls_convs': 'bbox_head.cls_convs',
        'dense_head.rpn3d_bbox_convs': 'bbox_head.reg_convs',
        'dense_head.conv_cls': 'bbox_head.conv_cls',
        'dense_head.conv_box': 'bbox_head.conv_reg',
        'dense_head.conv_dir_cls': 'bbox_head.conv_dir_cls'
    }

    # Delete some useless keys
    DELETE_KEYS = list()
    for old_key in converted_ckpt.keys():
        if 'global_step' in old_key:
            DELETE_KEYS.append(old_key)
    for delete_key in DELETE_KEYS:
        converted_ckpt.pop(delete_key)

    # Rename keys with specific prefix
    RENAME_KEYS = dict()
    for old_key in converted_ckpt.keys():
        for rename_prefix in RENAME_PREFIX.keys():
            if rename_prefix in old_key:
                new_key = old_key.replace(rename_prefix,
                                          RENAME_PREFIX[rename_prefix])
                RENAME_KEYS[new_key] = old_key
                break  # avoid duplicate replace

    for new_key, old_key in RENAME_KEYS.items():
        converted_ckpt[new_key] = converted_ckpt.pop(old_key)

    checkpoint['state_dict'] = converted_ckpt
    torch.save(checkpoint, args.out)
    # Check the converted checkpoint by loading to the model
    load_state_dict(model, converted_ckpt, strict=True)


if __name__ == '__main__':
    main()
