# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet3d.core.bbox import points_img2cam
from . import roiaware_pool3d_ext


def points_in_boxes_part(points, boxes):
    """Find the box in which each point is (CUDA).

    Args:
        points (torch.Tensor): [B, M, 3], [x, y, z] in LiDAR/DEPTH coordinate
        boxes (torch.Tensor): [B, T, 7],
            num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz] in
            LiDAR/DEPTH coordinate, (x, y, z) is the bottom center

    Returns:
        box_idxs_of_pts (torch.Tensor): (B, M), default background = -1
    """
    assert points.shape[0] == boxes.shape[0], \
        f'Points and boxes should have the same batch size, ' \
        f'got {points.shape[0]} and {boxes.shape[0]}'
    assert boxes.shape[2] == 7, \
        f'boxes dimension should be 7, ' \
        f'got unexpected shape {boxes.shape[2]}'
    assert points.shape[2] == 3, \
        f'points dimension should be 3, ' \
        f'got unexpected shape {points.shape[2]}'
    batch_size, num_points, _ = points.shape

    box_idxs_of_pts = points.new_zeros((batch_size, num_points),
                                       dtype=torch.int).fill_(-1)

    # If manually put the tensor 'points' or 'boxes' on a device
    # which is not the current device, some temporary variables
    # will be created on the current device in the cuda op,
    # and the output will be incorrect.
    # Therefore, we force the current device to be the same
    # as the device of the tensors if it was not.
    # Please refer to https://github.com/open-mmlab/mmdetection3d/issues/305
    # for the incorrect output before the fix.
    points_device = points.get_device()
    assert points_device == boxes.get_device(), \
        'Points and boxes should be put on the same device'
    if torch.cuda.current_device() != points_device:
        torch.cuda.set_device(points_device)

    roiaware_pool3d_ext.points_in_boxes_part(boxes.contiguous(),
                                             points.contiguous(),
                                             box_idxs_of_pts)

    return box_idxs_of_pts


def points_in_boxes_cpu(points, boxes):
    """Find all boxes in which each point is (CPU). The CPU version of
    :meth:`points_in_boxes_all`.

    Args:
        points (torch.Tensor): [B, M, 3], [x, y, z] in
            LiDAR/DEPTH coordinate
        boxes (torch.Tensor): [B, T, 7],
            num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz],
            (x, y, z) is the bottom center.

    Returns:
        box_idxs_of_pts (torch.Tensor): (B, M, T), default background = 0.
    """
    assert points.shape[0] == boxes.shape[0], \
        f'Points and boxes should have the same batch size, ' \
        f'got {points.shape[0]} and {boxes.shape[0]}'
    assert boxes.shape[2] == 7, \
        f'boxes dimension should be 7, ' \
        f'got unexpected shape {boxes.shape[2]}'
    assert points.shape[2] == 3, \
        f'points dimension should be 3, ' \
        f'got unexpected shape {points.shape[2]}'
    batch_size, num_points, _ = points.shape
    num_boxes = boxes.shape[1]

    point_indices = points.new_zeros((batch_size, num_boxes, num_points),
                                     dtype=torch.int)
    for b in range(batch_size):
        roiaware_pool3d_ext.points_in_boxes_cpu(boxes[b].float().contiguous(),
                                                points[b].float().contiguous(),
                                                point_indices[b])
    point_indices = point_indices.transpose(1, 2)

    return point_indices


def points_in_boxes_all(points, boxes):
    """Find all boxes in which each point is (CUDA).

    Args:
        points (torch.Tensor): [B, M, 3], [x, y, z] in LiDAR/DEPTH coordinate
        boxes (torch.Tensor): [B, T, 7],
            num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz],
            (x, y, z) is the bottom center.

    Returns:
        box_idxs_of_pts (torch.Tensor): (B, M, T), default background = 0.
    """
    assert boxes.shape[0] == points.shape[0], \
        f'Points and boxes should have the same batch size, ' \
        f'got {boxes.shape[0]} and {boxes.shape[0]}'
    assert boxes.shape[2] == 7, \
        f'boxes dimension should be 7, ' \
        f'got unexpected shape {boxes.shape[2]}'
    assert points.shape[2] == 3, \
        f'points dimension should be 3, ' \
        f'got unexpected shape {points.shape[2]}'
    batch_size, num_points, _ = points.shape
    num_boxes = boxes.shape[1]

    box_idxs_of_pts = points.new_zeros((batch_size, num_points, num_boxes),
                                       dtype=torch.int).fill_(0)

    # Same reason as line 25-32
    points_device = points.get_device()
    assert points_device == boxes.get_device(), \
        'Points and boxes should be put on the same device'
    if torch.cuda.current_device() != points_device:
        torch.cuda.set_device(points_device)

    roiaware_pool3d_ext.points_in_boxes_all(boxes.contiguous(),
                                            points.contiguous(),
                                            box_idxs_of_pts)

    return box_idxs_of_pts


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def points_in_boxes_cpu_idmap(points, boxes):
    """Original version of LIGA-DfM.

    Note here we expect the first dimension of input points and boxes is not
    batch_size. So we need to add [None] to be compatible with
    points_in_boxes_cpu in mmdet3d.
    """
    points, is_numpy = check_numpy_to_torch(points)
    boxes, is_numpy = check_numpy_to_torch(boxes)
    points, boxes = points[None], boxes[None]

    if len(boxes) > 0 and len(points) > 0:
        point_indices = points_in_boxes_cpu(points[None], boxes[None])[0]
        point_indices[point_indices == 0] = -1
        for i in range(boxes.shape[0]):
            point_indices[i, point_indices[i] == 1] = i
        point_indices = point_indices.max(0).values
    else:
        point_indices = np.empty([len(points)], dtype=np.int32)
        point_indices[:] = -1
        point_indices = torch.from_numpy(point_indices)

    return point_indices.numpy() if is_numpy else point_indices


def depth_map_in_boxes_cpu(depth_map,
                           boxes,
                           cam2img,
                           expand_ratio=1.0,
                           expand_distance=0.):
    """boxes should be in lidar format."""
    mask = depth_map > 0
    u = np.arange(depth_map.shape[1])
    v = np.arange(depth_map.shape[0])
    u, v = np.meshgrid(u, v)
    img_points = np.stack([u[mask], v[mask], depth_map[mask]],
                          axis=-1).reshape(-1, 3)
    cam_points = points_img2cam(img_points, cam2img)
    pseudo2cam_T = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float32)
    pseudo_points = (cam_points @ np.linalg.inv(pseudo2cam_T))[:, :3]

    fgmask_img = np.zeros(depth_map.shape[:2], dtype=np.int32)
    if boxes.shape[0] > 0:
        boxes = boxes.copy()
        boxes[:, [3, 4, 5]] *= expand_ratio
        boxes[:, [3, 4, 5]] += expand_distance
        point_in_boxes_ids = points_in_boxes_cpu_idmap(pseudo_points, boxes)
        fgmask_img[mask] = point_in_boxes_ids + 1
    return fgmask_img
