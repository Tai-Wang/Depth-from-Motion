# Depth from Motion (DfM)

This repository is the official implementation for DfM and MV-FCOS3D++.

![pv-demo](https://user-images.githubusercontent.com/30491025/181146351-876d8800-7261-4725-aeb1-b42e416eed01.gif)

![3d-demo-318](https://user-images.githubusercontent.com/30491025/181148417-915d9dd0-4f04-49fb-8106-4217e9d27e2a.gif) ![3d-demo2-318](https://user-images.githubusercontent.com/30491025/181148429-1d51bb92-68e2-4ab6-ac67-224822444b1d.gif)

## Introduction

This is an official release of the paper: `Monocular 3D Object Detection with Depth from Motion` and `MV-FCOS3D++: Multi-View Camera-Only 4D Object Detection with Pretrained Monocular Backbones`.

The code is still going through large refactoring. We plan to re-organize this repo as a combination of core codes for this project and mmdet3d requirement finally.

Please stay tuned for the clean release of all the configs and models.

Note: We will also release the refactored code in the official [mmdet3d](https://github.com/open-mmlab/mmdetection3d) soon.

> **Monocular 3D Object Detection with Depth from Motion**,            
> Tai Wang, Jiangmiao Pang, Dahua Lin            
> In: Proc. European Conference on Computer Vision (ECCV), 2022          
> [[arXiv](https://arxiv.org/abs/2207.12988)][[Bibtex](https://github.com/Tai-Wang/Depth-from-Motion#citation)]
>
> **MV-FCOS3D++: Multi-View Camera-Only 4D Object Detection with Pretrained Monocular Backbones**,            
> Tai Wang, Qing Lian, Chenming Zhu, Xinge Zhu, Wenwei Zhang            
> In: arxiv, 2022          
> [[arXiv](https://arxiv.org/abs/2207.12716)][[Bibtex](https://github.com/Tai-Wang/Depth-from-Motion#citation)]

## Results

### DfM

The results of DfM and its corresponding config are shown as below.

We have released the preliminary model for reproducing the results on the KITTI validation set.

The complete model checkpoints and logs will be released soon.

|  Backbone | Lr schd | Mem (GB) | Inf time (fps) |  Easy  | Moderate | Hard | Download|
| :-------: | :-----: | :------: | :------------: | :----: | :------: | :--: | :-----: |
| [ResNet34](./configs/dfm/dfm_r34_1x8_kitti-3d-3class.py) | - | - | - | 29.3569 | 19.5483 | 17.1763<sup>1</sup> | [model](https://download.openmmlab.com/mim-example/dfm/dfm_r34_1x8_kitti-3d-3class/epoch_60.pth)\|[log](https://download.openmmlab.com/mim-example/dfm/dfm_r34_1x8_kitti-3d-3class/20220819_205904.log.json)|

[1] This reproduced performance may have some degree of fluctuation due to the limited training samples and sensitive metrics. From my experience of multiple runs, the average performance may vary from 26/18/16 to 29/20/17, depending on the effect of corner cases (caused by matrix inverse computation or other reasons). Please stay tuned for a more stable version.

### MV-FCOS3D++

The results of MV-FCOS3D++ (baseline version) and its corresponding config are shown as below.

We have released the preliminary config for reproducing the results on the Waymo validation set.

(To comply the license agreement of Waymo dataset, the pre-trained models on Waymo dataset are not released.)

The complete model configs and logs will be released soon.

#### Pretrained FCOS3D++ (without customized finetuning)

|  Backbone | Lr schd | Mem (GB) | Inf time (fps) |  mAPL  | mAP | mAPH | Download |
| :-------: | :-----: | :------: | :------------: | :----: | :------: | :--: | :-----: |
| [ResNet101+DCN](./configs/pgd/pgd_r101_fpn_gn-head_dcn_3x16_2x_waymoD3-mv3d.py) | - | - | - | 0.2041 | 0.286 | 0.2701 | [log](https://download.openmmlab.com/mim-example/dfm/pgd_r101_fpn_gn-head_dcn_3x16_2x_waymoD3-mv3d/20220808_221519.log.json) |
| above @ Car | - | - | - | 0.4105 | 0.5574 | 0.5483 | |
| above @ Pedestrian | - | - | - | 0.1877 | 0.2785 | 0.2421 | |
| above @ Cyclist | - | - | - | 0.0143 | 0.0221 | 0.02 | |

#### MV-FCOS3D++ with Pretrained FCOS3D++

|  Backbone | Lr schd | Mem (GB) | Inf time (fps) |  mAPL  | mAP | mAPH | Download |
| :-------: | :-----: | :------: | :------------: | :----: | :------: | :--: | :-----: |
| [ResNet101+DCN](./configs/dfm/multiview-dfm_r101_dcn_2x16_waymoD5-3d-3class_camsync.py) | - | - | - | 0.338 | 0.4665 | 0.4425| [log](https://download.openmmlab.com/mim-example/dfm/multiview-dfm_r101_dcn_2x16_waymoD5-3d-3class_camsync/20220807_153735.log.json) |
| above @ Car | - | - | - | 0.5269 | 0.6836 | 0.6747 | |
| above @ Pedestrian | - | - | - | 0.2682 | 0.3847 | 0.341 | |
| above @ Cyclist | - | - | - | 0.219 | 0.3311 | 0.3116 | |
| [ResNet101+DCN+10 sweeps](./configs/dfm/multiview-dfm_r101_dcn_2x16_waymoD5-3d-3class_camsync_10sweeps.py) | - | - | - | 0.3514| 0.4798 | 0.4549 | [log1](https://download.openmmlab.com/mim-example/dfm/multiview-dfm_r101_dcn_2x16_waymoD5-3d-3class_camsync_10sweeps/20220808_170010.log.json) \| [log2](https://download.openmmlab.com/mim-example/dfm/multiview-dfm_r101_dcn_2x16_waymoD5-3d-3class_camsync_10sweeps/20220809_093358.log.json) |
| above @ Car | - | - | - | 0.5544 | 0.7072 | 0.6979 | |
| above @ Pedestrian | - | - | - | 0.276 | 0.395 | 0.351 | |
| above @ Cyclist | - | - | - | 0.2239 | 0.3372 | 0.3159 | |
| [ResNet101+DCN (slow infer)<sup>2</sup>](./configs/dfm/multiview-dfm_r101_dcn_2x16_waymoD5-3d-3class_camsync.py) | - | - | - | 0.379 | 0.5215 | 0.4884| |
| above @ Car | - | - | - | 0.5624 | 0.7315 | 0.7207 | |
| above @ Pedestrian | - | - | - | 0.346 | 0.4901 | 0.4225 | |
| above @ Cyclist | - | - | - | 0.2284 | 0.3429 | 0.3218 | |

[2] "slow infer" refers to changing the nms setting to `nms_pre=4096` and `max_num=500` to increase the number of predictions such that the inference can get a better recall performance. It will slow down the inference procedure but significantly improves the final performance under the Waymo metric. **The same trick can also be applied to the 10-sweep config and other models.**

## Installation

It requires the following OpenMMLab packages:

- MMCV-full >= v1.6.0 (recommended for the latest iou3d computation)
- MMDetection >= v2.24.0
- MMSegmentation >= v0.20.0

All the above versions are recommended except mmcv. Lower version of mmdet and mmseg may also work but are not tested temporarily.

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

### Pretrained models

For the KITTI implementation of DfM, we keep the LIGA-Stereo setting that has a LiDAR-based teacher for better supervision during training. Please download the teacher checkpoint (has been converted to mmdet3d-style) [here](https://download.openmmlab.com/mim-example/dfm/pretrained_models/mmdet3d-second-teacher.pth). It can make this network converge faster and bring ~1 AP performance gain. We will consider to replace it with other more direct supervision for simpler usage in the near future.

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

## Acknowledgement

This codebase is based on [MMDet3D](https://github.com/open-mmlab/mmdetection3d) and it benefits a lot from [LIGA-Stereo](https://github.com/xy-guo/LIGA-Stereo).

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
