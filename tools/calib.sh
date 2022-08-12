#!/usr/bin/env bash

set -x

CKPT_PATH=/mnt/lustre/wangtai/mmdet3d-prerelease/work_dirs
PARTITION=mm_det
JOB_NAME=calib
TASK=calib
CONFIG=configs/dfm/calib.py
WORK_DIR=${CKPT_PATH}/${TASK}
CKPT=${CKPT_PATH}/${TASK}/latest.pth
GPUS=1
GPUS_PER_NODE=1
XNODE=SH-IDC1-10-140-0-[137,168],SH-IDC1-10-140-1-[61]
PORT=29301

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} -x ${XNODE} -w SH-IDC1-10-140-1-120 \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    --quotatype=auto \
    python -Xfaulthandler -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" --seed 0 --deterministic
