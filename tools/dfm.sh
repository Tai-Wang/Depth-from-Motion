#!/usr/bin/env bash

set -x

CKPT_PATH=/mnt/lustre/wangtai.vendor/mmdet3d-prerelease/work_dirs
PARTITION=robot
JOB_NAME=dfm-depth-noaug
TASK=dfm-depth-noaug
CONFIG=configs/dfm/dfm-baseline-depth.py
WORK_DIR=${CKPT_PATH}/${TASK}
CKPT=${CKPT_PATH}/${TASK}/latest.pth
GPUS=1
GPUS_PER_NODE=1
XNODE=SH-IDC1-10-140-0-[224,232,240,242,245,247,255],SH-IDC1-10-140-1-[24,28,41,78,98,65,112]
PORT=29301

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} -x ${XNODE} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    --quotatype=auto \
    python -Xfaulthandler -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm"
