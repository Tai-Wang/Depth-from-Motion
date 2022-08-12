#!/usr/bin/env bash

set -x

CKPT_PATH=/mnt/lustre/wangtai/mmdet3d-prerelease/work_dirs
PARTITION=mm_det
JOB_NAME=test-mvfcos3d-10sweeps-full-pretrain
TASK=test-mvfcos3d-10sweeps-full-pretrain
CONFIG=configs/dfm/multiview-dfm_r101_dcn_2x16_waymoD5-3d-3class_camsync_10sweeps.py
WORK_DIR=${CKPT_PATH}/${TASK}
CKPT=${CKPT_PATH}/${TASK}/latest.pth
GPUS=16
GPUS_PER_NODE=8
XNODE=SH-IDC1-10-140-0-[137,168,224],SH-IDC1-10-140-1-[61,114]
PORT=29301

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} -x ${XNODE} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    --quotatype=reserved \
    python -Xfaulthandler -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm"
