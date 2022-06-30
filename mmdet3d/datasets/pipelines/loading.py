# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import numpy as np
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import (Compose, LoadAnnotations,
                                      LoadImageFromFile)
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk'),
                 num_views=5,
                 num_ref_frames=-1,
                 test_mode=False,
                 set_default_scale=True):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.num_views = num_views
        # num_ref_frames is used for multi-sweep loading
        self.num_ref_frames = num_ref_frames
        # when test_mode=False, we randomly select previous frames
        # otherwise, select the earliest one
        self.test_mode = test_mode
        self.set_default_scale = set_default_scale

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # TODO: consider split the multi-sweep part out of this pipeline
        # Derive the mask and transform for loading of multi-sweep data
        if self.num_ref_frames > 0:
            # init choice with the current frame
            init_choice = np.array([0], dtype=np.int64)
            num_frames = len(results['img_filename']) // self.num_views - 1
            if num_frames == 0:  # no previous frame, then copy cur frames
                choices = np.random.choice(
                    1, self.num_ref_frames, replace=True)
            elif num_frames >= self.num_ref_frames:
                # NOTE: suppose the info is saved following the order
                # from latest to earlier frames
                if self.test_mode:
                    choices = np.arange(num_frames - self.num_ref_frames,
                                        num_frames) + 1
                # NOTE: +1 is for selecting previous frames
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=False) + 1
            elif num_frames > 0 and num_frames < self.num_ref_frames:
                if self.test_mode:
                    base_choices = np.arange(num_frames) + 1
                    random_choices = np.random.choice(
                        num_frames,
                        self.num_ref_frames - num_frames,
                        replace=True) + 1
                    choices = np.concatenate([base_choices, random_choices])
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=True) + 1
            else:
                raise NotImplementedError
            choices = np.concatenate([init_choice, choices])
            select_filename = []
            for choice in choices:
                select_filename += results['img_filename'][choice *
                                                           self.num_views:
                                                           (choice + 1) *
                                                           self.num_views]
            results['img_filename'] = select_filename
            for key in ['lidar2img', 'cam2img', 'lidar2cam']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += results[key][choice *
                                                       self.num_views:(choice +
                                                                       1) *
                                                       self.num_views]
                    results[key] = select_results
            for key in ['ego2global']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += [results[key][choice]]
                    results[key] = select_results
            # Transform lidar2img and lidar2cam to
            # [cur_lidar]2[prev_img] and [cur_lidar]2[prev_cam]
            for key in ['lidar2img', 'lidar2cam']:
                if key in results:
                    # only change matrices of previous frames
                    for choice_idx in range(1, len(choices)):
                        pad_prev_ego2global = np.eye(4)
                        prev_ego2global = results['ego2global'][choice_idx]
                        pad_prev_ego2global[:prev_ego2global.
                                            shape[0], :prev_ego2global.
                                            shape[1]] = prev_ego2global
                        pad_cur_ego2global = np.eye(4)
                        cur_ego2global = results['ego2global'][0]
                        pad_cur_ego2global[:cur_ego2global.
                                           shape[0], :cur_ego2global.
                                           shape[1]] = cur_ego2global
                        cur2prev = np.linalg.inv(pad_prev_ego2global).dot(
                            pad_cur_ego2global)
                        for result_idx in range(choice_idx * self.num_views,
                                                (choice_idx + 1) *
                                                self.num_views):
                            results[key][result_idx] = \
                                results[key][result_idx].dot(cur2prev)
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        filename = results['img_filename']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [self.file_client.get(name) for name in filename]
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        if 'cam2img' in results:
            results['ori_cam2img'] = copy.deepcopy(results['cam2img'])
        if 'lidar2img' in results:
            results['ori_lidar2img'] = copy.deepcopy(results['lidar2img'])
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f'num_views={self.num_views}, '
        repr_str += f'num_ref_frames={self.num_ref_frames}, '
        repr_str += f'test_mode={self.test_mode})'
        return repr_str


@PIPELINES.register_module()
class LoadMultiViewDepthFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
    """

    def __init__(self,
                 to_float32=False,
                 file_client_args=dict(backend='disk'),
                 with_transform=True):
        self.to_float32 = to_float32
        self.file_client_args = file_client_args
        self.file_client = None
        self.with_transform = with_transform

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        # NOTE: we only load the depth_img of current frame for supervision
        img_filenames = results['img_filename'][:results['num_views']]
        depth_filenames = [
            img_filename.replace('image_', 'depth_image_')
            for img_filename in img_filenames
        ]

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # depth_img is of shape (h, w, num_views)
        # h and w can be different for different views
        depth_bytes = [self.file_client.get(name) for name in depth_filenames]
        depth_imgs = [
            mmcv.imfrombytes(depth_byte, flag='grayscale')
            for depth_byte in depth_bytes
        ]
        # handle the image with different shape
        depth_img_shapes = np.stack(
            [depth_img.shape for depth_img in depth_imgs], axis=0)
        depth_img_shape_max = np.max(depth_img_shapes, axis=0)
        depth_img_shape_min = np.min(depth_img_shapes, axis=0)
        assert depth_img_shape_min[-1] == depth_img_shape_max[-1]
        if not np.all(depth_img_shape_max == depth_img_shape_min):
            pad_shape = results['img_shape']
        else:
            pad_shape = None
        if pad_shape is not None:
            depth_imgs = [
                mmcv.impad(depth_img, shape=pad_shape, pad_val=0)
                for depth_img in depth_imgs
            ]
        depth_img = np.stack(depth_imgs, axis=-1)
        if self.to_float32:
            depth_img = depth_img.astype(np.float32)

        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['depth_img'] = [
            depth_img[..., i] for i in range(depth_img.shape[-1])
        ]
        # hack the depth_fields with seg_fields given their similarity
        # TODO: distinguish seg_fields and depth_fields
        if self.with_transform:
            # add into seg_fields to apply transforms same as imgs
            results['seg_fields'].append('depth_img')
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32})'
        return repr_str


@PIPELINES.register_module()
class LoadDepthFromFile(object):
    """Load depth maps."""

    def __init__(self,
                 to_float32=False,
                 file_client_args=dict(backend='disk'),
                 with_transform=True):
        self.to_float32 = to_float32
        self.file_client_args = file_client_args
        self.file_client = None
        self.with_transform = with_transform

    def __call__(self, results):
        """Call function to load depth maps from files.

        Args:
            results (dict): Result dict containing multi-frame image filenames.
        Returns:
            dict: The result dict containing the multi-frame image data.
                Added keys are deducted from all pipelines from
                self.transforms.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        img_filename = results['filename']

        if 'samples' in img_filename:  # nuscenes
            depth_filename = img_filename.replace('samples', 'depths').replace(
                'jpg', 'png')
            depth_img = np.load(depth_filename)['velodyne_depth']
        elif 'kitti_format' in img_filename:  # waymo
            depth_filename = img_filename.replace('image_', 'depth_image_')
            depth_bytes = self.file_client.get(depth_filename)
            depth_img = mmcv.imfrombytes(depth_bytes, flag='grayscale')
        else:  # KITTI
            depth_filename = img_filename.replace('image_2', 'depth_2')
            depth_bytes = self.file_client.get(depth_filename)
            depth_img = mmcv.imfrombytes(depth_bytes, flag='grayscale')

        if self.to_float32:
            depth_img = depth_img.astype(np.float32)

        results['depth_img'] = depth_img
        # hack the depth_fields with seg_fields given their similarity
        # TODO: distinguish seg_fields and depth_fields
        if self.with_transform:
            # add into seg_fields to apply transforms same as imgs
            results['seg_fields'].append('depth_img')

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        return repr_str


@PIPELINES.register_module()
class VideoPipeline(object):
    """Load and transform multi-frame images.

    Args:
        transforms (list[dict]): Transforms to apply for each image.
            The list of transforms for MultiViewPipeline and
            MultiScaleFlipAug differs. Here LoadImageFromFile is
            required as the first transform. Other transforms as usual
            can include Normalize, Pad, Resize etc.
        num_ref_imgs (int): Number of referenced images. Defaults to -1.
    """

    def __init__(self,
                 transforms,
                 num_ref_imgs=-1,
                 random=True,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'pad_shape', 'scale_factor', 'flip',
                            'cam_intrinsic', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'rect', 'Trv2c', 'P2', 'pcd_trans',
                            'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                            'pts_filename', 'transformation_3d_flow',
                            'cam2global', 'ori_cam2img')):
        self.transforms = Compose(transforms)
        self.num_ref_imgs = num_ref_imgs
        self.random = random
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to load multi-frame image from files.

        Args:
            results (dict): Result dict containing multi-frame image filenames.
        Returns:
            dict: The result dict containing the multi-frame image data.
                Added keys are deducted from all pipelines from
                self.transforms.
        """
        # To deprecate: assert len(results['img_info']['sweeps'])

        # sample self.num_ref_imgs from all images
        if self.num_ref_imgs > 0 and len(results['img_info']['sweeps']):
            ids = np.arange(len(results['img_info']['sweeps']))
            if self.random:
                replace = True if self.num_ref_imgs > len(ids) else False
                ids = np.random.choice(ids, self.num_ref_imgs, replace=replace)
            else:
                ids = ids[-self.num_ref_imgs:]
        else:
            ids = np.arange(0)

        # make a backup for original results
        org_results = copy.deepcopy(results)

        base_results = copy.deepcopy(results)
        # extract the transformation matrix from camera to global
        if 'cam2global' not in base_results['img_info']:  # nuScenes
            cam2ego_rot = np.array(
                Quaternion(base_results['img_info']
                           ['cam2ego_rotation']).transformation_matrix)
            cam2ego = cam2ego_rot.copy()
            cam2ego[0:3, 3] += np.array(
                base_results['img_info']['cam2ego_translation']).T
            ego2global_rot = np.array(
                Quaternion(base_results['img_info']
                           ['ego2global_rotation']).transformation_matrix)
            ego2global = ego2global_rot.copy()
            ego2global[0:3, 3] += np.array(
                base_results['img_info']['ego2global_translation']).T
            cam2global = np.dot(ego2global, cam2ego).astype(np.float32)
            base_results['cam2global'] = cam2global
        else:
            base_results['cam2global'] = base_results['img_info']['cam2global']

        base_results = self.transforms(base_results)
        multi_imgs = [base_results['img']]
        """
        # collect image metas
        img_metas = {}
        for key in self.meta_keys:
            if key in base_results:
                img_metas[key] = base_results[key]
        """
        # only save prev_img_metas in sweep_img_metas
        sweep_img_metas = []
        # apply self.transforms to referenced images
        for i in ids.tolist():
            ref_results = copy.deepcopy(org_results)
            ref_results['img_info']['filename'] = \
                results['img_info']['sweeps'][i]['data_path']
            # set flip value to make flip transformation consistent
            if 'flip' in base_results.keys():
                ref_results['flip'] = base_results['flip']
                ref_results['flip_direction'] = base_results['flip_direction']
            # set scale value to make resize consistent
            # use scale instead of img_scale = img_shape[:2]
            # to avoid some corner cases (inconsistent size with
            # base_results caused by the keep_ratio choice)
            if 'scale' in base_results.keys():
                ref_results['scale'] = base_results['scale']
            # extract the transformation matrix from camera to global
            if 'cam2global' not in ref_results['img_info']['sweeps'][i]:
                cam2ego_rot = np.array(
                    Quaternion(ref_results['img_info']['sweeps'][i]
                               ['sensor2ego_rotation']).transformation_matrix)
                cam2ego = cam2ego_rot.copy()
                cam2ego[0:3, 3] += np.array(ref_results['img_info']['sweeps']
                                            [i]['sensor2ego_translation']).T
                ego2global_rot = np.array(
                    Quaternion(ref_results['img_info']['sweeps'][i]
                               ['ego2global_rotation']).transformation_matrix)
                ego2global = ego2global_rot.copy()
                ego2global[0:3,
                           3] += np.array(ref_results['img_info']['sweeps'][i]
                                          ['ego2global_translation']).T
                cam2global = np.dot(ego2global, cam2ego).astype(np.float32)
                ref_results['cam2global'] = cam2global
            else:
                ref_results['cam2global'] = \
                    ref_results['img_info']['sweeps'][i]['cam2global']

            ref_results = self.transforms(ref_results)
            if 'flip' in ref_results.keys():
                assert ref_results['flip'] == base_results['flip']
            if 'pad_shape' in ref_results.keys():
                assert ref_results['pad_shape'] == base_results['pad_shape']
            multi_imgs.append(ref_results['img'])
            # collect image metas
            img_metas = {}
            for key in self.meta_keys:
                if key in ref_results:
                    img_metas[key] = ref_results[key]
            sweep_img_metas.append(img_metas)

        base_results['img'] = multi_imgs
        base_results['sweep_img_metas'] = sweep_img_metas

        # copy the contents in base_results to results
        for key in base_results.keys():
            results[key] = base_results[key]
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'num_ref_imgs={self.num_ref_imgs})'
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        if 'cam2img' not in results.keys():  # Mono3D dataset class
            results['cam2img'] = results['img_info']['cam_intrinsic']
        # save the original cam2img before augmentation
        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])
        return results


@PIPELINES.register_module()
class DepthPipeline(LoadImageFromFileMono3D):
    """Load and transform depth maps.

    Args:
        transforms (list[dict]): Transforms to apply for the depth map.
            The list of transforms for MultiViewPipeline and
            MultiScaleFlipAug differs. Here LoadImageFromFile is
            required as the first transform. Other transforms as usual
            can include Normalize, Pad, Resize etc.
    """

    def __init__(self,
                 transforms=[],
                 file_client_args=dict(backend='disk'),
                 **kwargs):
        super().__init__(file_client_args=file_client_args, **kwargs)
        self.transforms = Compose(transforms)

    def __call__(self, results):
        """Call function to load depth maps from files.

        Args:
            results (dict): Result dict containing multi-frame image filenames.
        Returns:
            dict: The result dict containing the multi-frame image data.
                Added keys are deducted from all pipelines from
                self.transforms.
        """
        super().__call__(results)
        # apply self.transforms to the depth map
        img_filename = results['filename']
        if 'samples' in img_filename:  # nuscenes
            depth_filename = img_filename.replace('samples', 'depths').replace(
                'jpg', 'png')
            depth_map = np.load(depth_filename)['velodyne_depth']
        else:  # KITTI
            depth_filename = img_filename.replace('image_2', 'depth_2')
            depth_bytes = self.file_client.get(depth_filename)
            depth_map = mmcv.imfrombytes(depth_bytes, flag='grayscale')
            """
            # only use the part smaller than (375, 1242)
            if depth_map.shape[0] > 375:
                depth_map = depth_map[:375, :]
            if depth_map.shape[1] > 1242:
                depth_map = depth_map[:, :1242]
            """
        results['depth_map'] = depth_map.astype(np.float32)
        results['seg_fields'].append('depth_map')

        results = self.transforms(results)
        # avoid the transforms out of this pipeline also applying to depth_map
        # TODO: consider categorizing depth_map into seg_fields
        # results['img_fields'].remove('depth_map')

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int, optional): Number of sweeps. Defaults to 10.
        load_dim (int, optional): Dimension number of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimension to use.
            Defaults to [0, 1, 2, 4].
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool, optional): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool, optional): Whether to remove close points.
            Defaults to False.
        test_mode (bool, optional): If `test_mode=True`, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk'),
                 pseudo_lidar=False):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pseudo_lidar = pseudo_lidar

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.pseudo_lidar:
            lidar2cam = results['lidar2cam']
            points_xyz = points[:, :3]
            homo_points_xyz = np.concatenate(
                [points_xyz, np.ones([points.shape[0], 1])], axis=-1)
            cam_points = homo_points_xyz @ lidar2cam.T
            # cam points -> pseudo lidar points
            pseudo2cam_T = np.array(
                [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                dtype=np.float32)
            points[:, :3] = (cam_points @ np.linalg.inv(pseudo2cam_T))[:, :3]

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def __call__(self, results):
        assert 'points' in results
        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype=np.int64,
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str
