import argparse

try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-1-0==1.2.0" '
        'to install the official devkit first.')

from glob import glob
from os.path import join

import mmcv
import tensorflow as tf
from waymo_open_dataset.protos import metrics_pb2


def save_label(frame, objects, version='multi-view', cam_sync=True):
    """Parse and save gt bin file for camera-only 3D detection on Waymo.

    Args:
        frame (:obj:`Frame`): Open dataset frame proto.
        objects (:obj:`Object`): Ground truths in waymo dataset Object proto.
        version (str): Version of gt bin file. Choices include 'multi-view'
            and 'front-view'. Defaults to 'multi-view'.
        cam_sync (bool): Whether to generate camera synced gt bin. Defaults to
            True.
    """
    id_to_bbox = dict()
    id_to_name = dict()
    for labels in frame.projected_lidar_labels:
        name = labels.name  # 0 unknown, 1-5 corresponds to 5 cameras
        for label in labels.labels:
            # TODO: need a workaround as bbox may not belong to front cam
            bbox = [
                label.box.center_x - label.box.length / 2,
                label.box.center_y - label.box.width / 2,
                label.box.center_x + label.box.length / 2,
                label.box.center_y + label.box.width / 2
            ]
            # object id in one frame
            id_to_bbox[label.id] = bbox
            id_to_name[label.id] = name - 1
    if version == 'multi-view':
        cam_list = [
            '_FRONT', '_FRONT_LEFT', '_FRONT_RIGHT', '_SIDE_LEFT',
            '_SIDE_RIGHT'
        ]
    elif version == 'front-view':
        cam_list = ['_FRONT']
    else:
        raise NotImplementedError
    for obj in frame.laser_labels:
        bounding_box = None
        name = None
        id = obj.id
        for cam in cam_list:
            if id + cam in id_to_bbox:
                bounding_box = id_to_bbox.get(id + cam)
                name = str(id_to_name.get(id + cam))
                break
        num_pts = obj.num_lidar_points_in_box

        if cam_sync:
            if obj.most_visible_camera_name:
                box3d = obj.camera_synced_box
            else:
                continue
        else:
            box3d = obj.box

        if bounding_box is not None and obj.type > 0 and num_pts >= 1:
            o = metrics_pb2.Object()
            o.context_name = frame.context.name
            o.frame_timestamp_micros = frame.timestamp_micros
            o.score = 0.5
            o.object.CopyFrom(obj)
            o.object.box.CopyFrom(box3d)
            objects.objects.append(o)


parser = argparse.ArgumentParser(description='Waymo GT Generator arg parser')
parser.add_argument(
    '--root-path',
    type=str,
    default='data/waymo/waymo_format/validation',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='multi-view',
    required=False,
    help='specify the gt version')
parser.add_argument(
    '--out-dir',
    type=str,
    default='data/waymo/waymo_format/',
    required=False,
    help='name of info pkl')
parser.add_argument(
    '--backend',
    type=str,
    default='disk',
    required=False,
    help='file backend for data loading')
args = parser.parse_args()

if __name__ == '__main__':
    load_dir = args.root_path
    version = args.version
    out_dir = args.out_dir
    if args.backend == 'disk':
        file_client_args = dict(backend='disk')
    elif args.backend == 'petrel':
        file_client_args = dict(
            backend='petrel',
            path_mapping=dict({
                './data/waymo/':
                's3://openmmlab/datasets/detection3d/waymo-v1.3.1/',
                'data/waymo/':
                's3://openmmlab/datasets/detection3d/waymo-v1.3.1/'
            }))
    tfrecord_pathnames = glob(join(load_dir, '*.tfrecord'))
    if file_client_args['backend'] == 'disk':
        tfrecord_pathnames = sorted(glob(join(load_dir, '*.tfrecord')))
    else:
        file_client = mmcv.FileClient(**file_client_args)
        load_dir = file_client.client._map_path(load_dir)
        from petrel_client.client import Client
        client = Client()
        contents = client.list(load_dir)
        tfrecord_pathnames = list()
        for content in sorted(list(contents)):
            if content.endswith('tfrecord'):
                tfrecord_pathnames.append(join(load_dir, content))

    objects = metrics_pb2.Objects()
    progress_bar = mmcv.ProgressBar(len(tfrecord_pathnames))
    for i in range(len(tfrecord_pathnames)):
        pathname = tfrecord_pathnames[i]
        segment_filename = pathname.split('/')[-1]
        context_name = segment_filename[8:-28]
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')
        for frame_idx, data in enumerate(dataset):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            save_label(frame, objects, version)
        progress_bar.update()

    # Write objects to a file.
    f = open(join(out_dir, 'cam_gt.bin'), 'wb')
    f.write(objects.SerializeToString())
    f.close()
