# Depth from Motion (DfM)

This repository is the official implementation for DfM and MV-FCOS3D++.

## Introduction

This is an official release of the paper **Monocular 3D Object Detection with Depth from Motion** & **MV-FCOS3D++: Multi-View Camera-Only 4D Object Detection with Pretrained Monocular Backbones**.

The code is still going through large refactoring. We plan to re-organize this repo as a combination of core codes for this project and mmdet3d requirement finally.

Please stay tuned for the clean release of all the configs and models.

Note: We will also release the refactored code in the official [mmdet3d](https://github.com/open-mmlab/mmdetection3d) soon.

> **Monocular 3D Object Detection with Depth from Motion**,            
> Tai Wang, Jiangmiao Pang, Dahua Lin            
> In: Proc. European Conference on Computer Vision (ECCV), 2022          
> \[arXiv][[Bibtex](https://github.com/Tai-Wang/Depth-from-Motion#citation)]
> **MV-FCOS3D++: Multi-View Camera-Only 4D Object Detection with Pretrained Monocular Backbones**,            
> Tai Wang, Qing Lian, Chenming Zhu, Xinge Zhu, Wenwei Zhang            
> In: arxiv, 2022          
> \[arXiv][[Bibtex](https://github.com/Tai-Wang/Depth-from-Motion#citation)]

## Results

### DfM

The results of DfM and its corresponding config are shown as below.
We have released the preliminary model for reproducing the results on the KITTI validation set.
The complete model checkpoints and logs will be released soon.

|  Backbone | Lr schd | Mem (GB) | Inf time (fps) |  Easy  | Moderate | Hard | Download|
| :-------: | :-----: | :------: | :------------: | :----: | :------: | :--: | :-----: |
| [ResNet34](./configs/dfm/dfm.py) | - | - | - |   |  |  | [model]()|

### MV-FCOS3D++

The results of MV-FCOS3D++ (baseline version) and its corresponding config are shown as below.
We have released the preliminary config for reproducing the results on the Waymo validation set.
(To comply the license agreement of Waymo dataset, the pre-trained models on Waymo dataset are not released.)
The complete model configs and logs will be released soon.

|  Backbone | Lr schd | Mem (GB) | Inf time (fps) |  mAPL  | mAP | mAPH | Download |
| :-------: | :-----: | :------: | :------------: | :----: | :------: | :--: | :-----: |
| [ResNet101+DCN](./configs/dfm/https://github.com/Tai-Wang/Depth-from-Motion/blob/main/configs/dfm/multiview-dfm_r101_dcn_2x16_waymoD5-3d-3class_camsync.py) | - | - | - |   |  |  |-|


## Installation

It requires the following OpenMMLab packages:

- MMCV-full >= v1.6.0 (recommended for the latest iou3d computation)
- MMDetection >= v2.24.0
- MMSegmentation >= v0.20.0

All the above versions are recommended except mmcv. Lower version of mmdet and mmseg may also work but I do not test them temporarily.

Example commands are shown as follows.

```bash
conda create --name dfm python=3.7 -y
conda activate dfm
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install mmcv-full==1.6.0
pip install mmdet==2.24.0
pip install mmsegmentation==0.20.0
git clone https://github.com/Tai-Wang/Depth-from-Motion.git
cd Depth-from-Motion
pip install -v -e .
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Usage

### Data preparation

First prepare the raw data of KITTI and Waymo data following [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

Then we prepare the data related to temporally consecutive frames. **(still unstable and details under modifying & testing)**

For KITTI, we need to additionally download the pose and label files of the raw data [here](https://www.cse.msu.edu/computervision/Kinematic3D-raw_extra.zip) and the official mapping (between the raw data and the 3D detection benchmark split) [here](https://github.com/garrickbrazil/kinematic3d/tree/master/data/kitti_split1/devkit/mapping). Then we can run the data converter script:

```
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

For Waymo, we need to additionally generate the ground truth bin file for camera-only setting (only boxes covered by the perception range of cameras are considered). Besides, we recommend use the latest waymo dataset, which includes the camera synced annotations tailored to this setting.

```
python tools/create_waymo_gt_bin.py
```

The final data structure looks like below:

```text
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── kitti
│   │   ├── ImageSets
│   │   ├── testing
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── prev_2
│   │   │   ├── velodyne
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── prev_2
│   │   │   ├── label_2
│   │   │   ├── velodyne
│   │   ├── raw
│   │   │   ├── 2011_09_26_drive_0001_sync
│   │   │   ├── xxxx (other raw data files)
│   │   ├── devkit
│   │   │   ├── mapping
│   │   │   │   ├── train_mapping.txt
│   │   │   │   ├── train_rand.txt
│   ├── waymo
│   │   ├── waymo_format
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── testing
│   │   │   ├── gt.bin
│   │   │   ├── cam_gt.bin
│   │   ├── kitti_format
│   │   │   ├── ImageSets

```

### Training and testing

For training and testing, you can follow the standard command in mmdet to train and test the model

```bash
# train DfM on KITTI
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

For simple inference and evaluation, you can use the command below:

```bash
# evaluate DfM on KITTI and MV-FCOS3D++ on Waymo
./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CKPT_PATH} --eval mAP
```

## Citation

```bibtex
@inproceedings{wang2022dfm,
    title={Monocular 3D Object Detection with Depth from Motion},
    author={Wang, Tai and Pang, Jiangmiao and Lin, Dahua},
    year={2022},
    booktitle={European Conference on Computer Vision (ECCV)},
}
@article{wang2022mvfcos3d++,
  title={{MV-FCOS3D++: Multi-View} Camera-Only 4D Object Detection with Pretrained Monocular Backbones},
  author={Wang, Tai and Lian, Qing and Zhu, Chenming and Zhu, Xinge and Zhang, Wenwei},
  journal={arXiv preprint},
  year={2022}
}
```
