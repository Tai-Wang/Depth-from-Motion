# Copyright (c) OpenMMLab. All rights reserved.
# yapf: disable
import copy
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmcv.ops.points_in_boxes import points_in_boxes_cpu
from nuscenes.utils.geometry_utils import view_points
from shapely.geometry import MultiPoint, box

from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, box_np_ops,
                               points_img2cam)
from mmdet3d.core.camera.calibration import Calibration
from mmdet3d.datasets.pipelines import (Collect3D, DefaultFormatBundle3D,
                                        LoadAnnotations3D,
                                        LoadImageFromFileMono3D,
                                        LoadMultiViewImageFromFiles,
                                        LoadPointsFromFile,
                                        LoadPointsFromMultiSweeps,
                                        MultiScaleFlipAug3D,
                                        PointSegClassMapping)
# yapf: enable
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadImageFromFile, MultiScaleFlipAug


def is_loading_function(transform):
    """Judge whether a transform function is a loading function.

    Note: `MultiScaleFlipAug3D` is a wrapper for multiple pipeline functions,
    so we need to search if its inner transforms contain any loading function.

    Args:
        transform (dict | :obj:`Pipeline`): A transform config or a function.

    Returns:
        bool: Whether it is a loading function. None means can't judge.
            When transform is `MultiScaleFlipAug3D`, we return None.
    """
    # TODO: use more elegant way to distinguish loading modules
    loading_functions = (LoadImageFromFile, LoadPointsFromFile,
                         LoadAnnotations3D, LoadMultiViewImageFromFiles,
                         LoadPointsFromMultiSweeps, DefaultFormatBundle3D,
                         Collect3D, LoadImageFromFileMono3D,
                         PointSegClassMapping)
    if isinstance(transform, dict):
        obj_cls = PIPELINES.get(transform['type'])
        if obj_cls is None:
            return False
        if obj_cls in loading_functions:
            return True
        if obj_cls in (MultiScaleFlipAug3D, MultiScaleFlipAug):
            return None
    elif callable(transform):
        if isinstance(transform, loading_functions):
            return True
        if isinstance(transform, (MultiScaleFlipAug3D, MultiScaleFlipAug)):
            return None
    return False


def get_loading_pipeline(pipeline):
    """Only keep loading image, points and annotations related configuration.

    Args:
        pipeline (list[dict] | list[:obj:`Pipeline`]):
            Data pipeline configs or list of pipeline functions.

    Returns:
        list[dict] | list[:obj:`Pipeline`]): The new pipeline list with only
            keep loading image, points and annotations related configuration.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadPointsFromFile',
        ...         coord_type='LIDAR', load_dim=4, use_dim=4),
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations3D',
        ...         with_bbox=True, with_label_3d=True),
        ...    dict(type='Resize',
        ...         img_scale=[(640, 192), (2560, 768)], keep_ratio=True),
        ...    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        ...    dict(type='PointsRangeFilter',
        ...         point_cloud_range=point_cloud_range),
        ...    dict(type='ObjectRangeFilter',
        ...         point_cloud_range=point_cloud_range),
        ...    dict(type='PointShuffle'),
        ...    dict(type='Normalize', **img_norm_cfg),
        ...    dict(type='Pad', size_divisor=32),
        ...    dict(type='DefaultFormatBundle3D', class_names=class_names),
        ...    dict(type='Collect3D',
        ...         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadPointsFromFile',
        ...         coord_type='LIDAR', load_dim=4, use_dim=4),
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations3D',
        ...         with_bbox=True, with_label_3d=True),
        ...    dict(type='DefaultFormatBundle3D', class_names=class_names),
        ...    dict(type='Collect3D',
        ...         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ...    ]
        >>> assert expected_pipelines == \
        ...        get_loading_pipeline(pipelines)
    """
    loading_pipeline = []
    for transform in pipeline:
        is_loading = is_loading_function(transform)
        if is_loading is None:  # MultiScaleFlipAug3D
            # extract its inner pipeline
            if isinstance(transform, dict):
                inner_pipeline = transform.get('transforms', [])
            else:
                inner_pipeline = transform.transforms.transforms
            loading_pipeline.extend(get_loading_pipeline(inner_pipeline))
        elif is_loading:
            loading_pipeline.append(transform)
    assert len(loading_pipeline) > 0, \
        'The data pipeline in your config file must include ' \
        'loading step.'
    return loading_pipeline


def extract_result_dict(results, key):
    """Extract and return the data corresponding to key in result dict.

    ``results`` is a dict output from `pipeline(input_dict)`, which is the
        loaded data from ``Dataset`` class.
    The data terms inside may be wrapped in list, tuple and DataContainer, so
        this function essentially extracts data from these wrappers.

    Args:
        results (dict): Data loaded using pipeline.
        key (str): Key of the desired data.

    Returns:
        np.ndarray | torch.Tensor: Data term.
    """
    if key not in results.keys():
        return None
    # results[key] may be data or list[data] or tuple[data]
    # data may be wrapped inside DataContainer
    data = results[key]
    if isinstance(data, (list, tuple)):
        data = data[0]
    if isinstance(data, mmcv.parallel.DataContainer):
        data = data._data
    return data


def get_waymo_2d_boxes(info, occluded, mono3d=True):
    """Get the 2D annotation records for a given info.

    This function is used to get 2D annotations when loading annotations from
    a dataset class. The original version in the data converter will be
    deprecated in the future.

    Args:
        info: Information of the given sample data.
        occluded: Integer (0, 1, 2, 3) indicating occlusion state: \
            0 = fully visible, 1 = partly occluded, 2 = largely occluded, \
            3 = unknown, -1 = DontCare
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """
    # Get calibration information
    P0 = info['calib']['P0']

    repro_recs = []
    # if no annotations in info (test dataset), then return
    if 'annos' not in info:
        return repro_recs

    # Get all the annotation with the specified visibilties.
    # filter the annotation bboxes by occluded attributes
    ann_dicts = info['annos']
    mask = [(ocld in occluded) for ocld in ann_dicts['occluded']]
    for k in ann_dicts.keys():
        ann_dicts[k] = ann_dicts[k][mask]

    # convert dict of list to list of dict
    ann_recs = []
    for i in range(len(ann_dicts['occluded'])):
        ann_rec = {}
        for k in ann_dicts.keys():
            ann_rec[k] = ann_dicts[k][i]
        ann_recs.append(ann_rec)

    for ann_idx, ann_rec in enumerate(ann_recs):
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = \
            f"{info['image']['image_idx']}.{ann_idx}"
        ann_rec['sample_data_token'] = info['image']['image_idx']
        sample_data_token = info['image']['image_idx']

        loc = ann_rec['location'][np.newaxis, :]
        dim = ann_rec['dimensions'][np.newaxis, :]
        rot = ann_rec['rotation_y'][np.newaxis, np.newaxis]

        # transform the center from [0.5, 1.0, 0.5] to [0.5, 0.5, 0.5]
        dst = np.array([0.5, 0.5, 0.5])
        src = np.array([0.5, 1.0, 0.5])
        loc = loc + dim * (dst - src)
        loc_3d = np.copy(loc)
        gt_bbox_3d = np.concatenate([loc, dim, rot], axis=1).astype(np.float32)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box_np_ops.center_to_corner_box3d(
            gt_bbox_3d[:, :3],
            gt_bbox_3d[:, 3:6],
            gt_bbox_3d[:, 6], [0.5, 0.5, 0.5],
            axis=1)
        corners_3d = corners_3d[0].T  # (1, 8, 3) -> (3, 8)
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        camera_intrinsic = P0
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(
            corner_coords,
            imsize=(info['image']['image_shape'][1],
                    info['image']['image_shape'][0]))

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token,
                                    info['image']['image_path'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            repro_rec['bbox_cam3d'] = np.concatenate(
                [loc_3d, dim, rot],
                axis=1).astype(np.float32).squeeze().tolist()
            repro_rec['velo_cam3d'] = -1  # no velocity in KITTI

            center3d = np.array(loc).reshape([1, 3])
            center2d = box_np_ops.points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            repro_rec['attribute_name'] = -1  # no attribute in KITTI
            repro_rec['attribute_id'] = -1

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(corner_coords, imsize):
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec, x1, y1, x2, y2, sample_data_token, filename):
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.

    The original version in the data converter will be deprecated in the
    future.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, x_size, y_size of 2d box
            - iscrowd (int): whether the area is crowd
    """
    kitti_categories = ('Pedestrian', 'Cyclist', 'Car')
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    key_mapping = {
        'name': 'category_name',
        'num_points_in_gt': 'num_lidar_pts',
        'sample_annotation_token': 'sample_annotation_token',
        'sample_data_token': 'sample_data_token',
    }

    for key, value in ann_rec.items():
        if key in key_mapping.keys():
            repro_rec[key_mapping[key]] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in kitti_categories:
        return None
    cat_name = repro_rec['category_name']
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = kitti_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec


def convert_annos(info, cam_idx):
    """Convert front-cam anns to i-th camera (KITTI-style info).

    Args:
        info (dict): The sample info in dict format, in the referenced camera
            coordinate system. The reference is set to cam-0 by default.
        cam_idx (int): The index of the camera whose info needs to be
            converted.

    Returns:
        dict: Converted infos in the coordinate system of i-th camera.
    """
    rect = info['calib']['R0_rect'].astype(np.float32)
    lidar2cam0 = info['calib']['Tr_velo_to_cam'].astype(np.float32)
    lidar2cami = info['calib'][f'Tr_velo_to_cam{cam_idx}'].astype(np.float32)
    annos = info['annos']
    converted_annos = copy.deepcopy(annos)
    loc = annos['location']
    dims = annos['dimensions']
    rots = annos['rotation_y']
    gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                  axis=1).astype(np.float32)
    # convert gt_bboxes_3d to velodyne coordinates
    gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
        Box3DMode.LIDAR, np.linalg.inv(rect @ lidar2cam0), correct_yaw=True)
    # convert gt_bboxes_3d to cam coordinates
    gt_bboxes_3d = gt_bboxes_3d.convert_to(
        Box3DMode.CAM, rect @ lidar2cami, correct_yaw=True).tensor.numpy()
    converted_annos['location'] = gt_bboxes_3d[:, :3]
    converted_annos['dimensions'] = gt_bboxes_3d[:, 3:6]
    converted_annos['rotation_y'] = gt_bboxes_3d[:, 6]
    return converted_annos


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


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
    img_pts = np.stack([u[mask], v[mask], depth_map[mask]], axis=1)
    rect_pts = points_img2cam(img_pts, cam2img)
    rect_pts = np.hstack(
        (rect_pts, np.ones((rect_pts.shape[0], 1), dtype=np.float32)))
    T = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                 dtype=np.float32)
    pts = np.dot(rect_pts, np.linalg.inv(T))[:, :3]
    fgmask_img = np.zeros(depth_map.shape[:2], dtype=np.int32)
    if boxes.shape[0] > 0:
        boxes = boxes.copy()
        boxes[:, [3, 4, 5]] *= expand_ratio
        boxes[:, [3, 4, 5]] += expand_distance
        point_in_boxes_ids = points_in_boxes_cpu_idmap(pts, boxes)
        fgmask_img[mask] = point_in_boxes_ids + 1
    return fgmask_img


def points_in_boxes_cpu_idmap(points, boxes):
    if len(boxes.shape) == 2 and len(points.shape) == 2:
        single_batch = True
    points, is_numpy = check_numpy_to_torch(points)
    boxes, is_numpy = check_numpy_to_torch(boxes)

    if len(boxes) > 0 and len(points) > 0:
        # points_in_boxes_cpu in mmcv outputs the point_indices
        # in shape (num_points, num_boxes)
        # so transpose here
        if single_batch:
            point_indices = points_in_boxes_cpu(points[None],
                                                boxes[None])[0].transpose(
                                                    0, 1)
        else:
            raise NotImplementedError
        point_indices[point_indices == 0] = -1
        for i in range(boxes.shape[0]):
            point_indices[i, point_indices[i] == 1] = i
        point_indices = point_indices.max(0).values
    else:
        point_indices = np.empty([len(points)], dtype=np.int32)
        point_indices[:] = -1
        point_indices = torch.from_numpy(point_indices)

    return point_indices.numpy() if is_numpy else point_indices


def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords,
        see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array(
        [l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2],
        dtype=np.float32).T
    z_corners = np.array(
        [w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.],
        dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(
            4, axis=1)  # (N, 8)
    else:
        y_corners = np.array([
            h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.
        ],
                             dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(
        ry.size, dtype=np.float32), np.ones(
            ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros], [np.sin(ry), zeros,
                                                np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(
        -1, 8, 1), y_corners.reshape(-1, 8, 1), z_corners.reshape(-1, 8, 1)),
                                  axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], \
        rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate(
        (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)),
        axis=2)

    return corners.astype(np.float32)


def boxes3d_to_grid3d_kitti_camera(boxes3d,
                                   size=28,
                                   bottom_center=True,
                                   surface=False):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords,
        see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners, y_corners, z_corners = np.meshgrid(
        np.linspace(-0.5, 0.5, size), np.linspace(-0.5, 0.5, size),
        np.linspace(-0.5, 0.5, size))
    if surface:
        surface_mask = (np.abs(x_corners)
                        == 0.5) | (np.abs(y_corners) == 0.5) | (
                            np.abs(z_corners) == 0.5)
        x_corners = x_corners[surface_mask]
        y_corners = y_corners[surface_mask]
        z_corners = z_corners[surface_mask]
    x_corners = x_corners.reshape([1, -1]) * l.reshape([-1, 1])
    y_corners = y_corners.reshape([1, -1]) * h.reshape([-1, 1])
    z_corners = z_corners.reshape([1, -1]) * w.reshape([-1, 1])
    if bottom_center:
        y_corners -= h.reshape([-1, 1]) / 2

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(
        ry.size, dtype=np.float32), np.ones(
            ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros], [np.sin(ry), zeros,
                                                np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.stack([x_corners, y_corners, z_corners],
                            axis=-1)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], \
        rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners
    y = y_loc.reshape(-1, 1) + y_corners
    z = z_loc.reshape(-1, 1) + z_corners

    corners = np.stack([x, y, z], axis=-1)

    return corners.astype(np.float32)


def boxes3d_lidar_to_kitti_camera(boxes3d_lidar,
                                  calib=None,
                                  pseudo_lidar=False,
                                  pseduo_cam2_view=False):
    """
    :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading],
        (x, y, z) is the box center
    :param calib:
    :return:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    """
    # TODO: will modify original boxes3d_lidar
    xyz_lidar = boxes3d_lidar[:, 0:3].copy()
    l, w, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], \
        boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    if not pseudo_lidar:
        assert calib is not None, 'calib can only be None in pseudo_lidar mode'
        xyz_cam = calib.lidar_to_rect(xyz_lidar)
    else:
        # transform xyz from pseudo-lidar to camera view
        xyz_cam = Calibration.lidar_pseudo_to_rect(xyz_lidar)
        if pseduo_cam2_view:
            xyz_cam = xyz_cam - calib.txyz
    # xyz_cam[:, 1] += h.reshape(-1) / 2
    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)


def boxes3d_kitti_camera_to_imageboxes(boxes3d,
                                       calib,
                                       image_shape=None,
                                       return_neg_z_mask=False,
                                       fix_neg_z_bug=False):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    :param calib:
    :return:
        box_2d_preds: (N, 4) [x1, y1, x2, y2]
    """
    if not fix_neg_z_bug:
        raise NotImplementedError
        corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
        pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
        corners_in_image = pts_img.reshape(-1, 8, 2)

        min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
        max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
        boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
        if image_shape is not None:
            boxes2d_image[:, 0] = np.clip(
                boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 1] = np.clip(
                boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
            boxes2d_image[:, 2] = np.clip(
                boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 3] = np.clip(
                boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

        if not return_neg_z_mask:
            return boxes2d_image
        else:
            return boxes2d_image, np.all(corners3d[:, :, 2] >= 0.01, axis=1)
    else:
        num_boxes = boxes3d.shape[0]
        corners3d = boxes3d_to_grid3d_kitti_camera(
            boxes3d, size=7, surface=False)
        if num_boxes != 0:
            num_points = corners3d.shape[1]
            pts_img, pts_depth = calib.rect_to_img(corners3d.reshape(-1, 3))
            corners_in_image = pts_img.reshape(num_boxes, num_points, 2)
            depth_in_image = pts_depth.reshape(num_boxes, num_points)
            min_uv = []
            max_uv = []
            for idx, (x,
                      d) in enumerate(zip(corners_in_image, depth_in_image)):
                if not np.all(corners3d[idx, :, 2] >= 0.01, axis=0):
                    continue
                uv = x[d >= 0]
                if len(uv) > 0:
                    min_uv.append(np.min(uv, axis=0))
                    max_uv.append(np.max(uv, axis=0))
            min_uv = np.array(min_uv)  # (N, 2)
            max_uv = np.array(max_uv)  # (N, 2)
            boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
        else:
            boxes2d_image = np.zeros([0, 4], dtype=np.float32)

        if image_shape is not None:
            boxes2d_image[:, 0] = np.clip(
                boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 1] = np.clip(
                boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
            boxes2d_image[:, 2] = np.clip(
                boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 3] = np.clip(
                boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

        if not return_neg_z_mask:
            return boxes2d_image
        else:
            return boxes2d_image, np.all(corners3d[:, :, 2] >= 0.01, axis=1)


def boxes3d_kitti_camera_to_imagecenters(boxes3d, calib, image_shape=None):
    centers3d = boxes3d_to_corners3d_kitti_camera(boxes3d).mean(1)
    pts_img, _ = calib.rect_to_img(centers3d)
    return pts_img
