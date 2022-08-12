#!/usr/bin/env bash

set -x

CKPT_PATH=/mnt/lustre/wangtai/mmdet3d-prerelease/work_dirs
PARTITION=mm_det
JOB_NAME=eval
TASK=test-fcos3d-waymo
CONFIG=configs/pgd/pgd_r101_fpn_gn-head_dcn_3x16_2x_waymoD3-mv3d.py
WORK_DIR=${CKPT_PATH}/${TASK}
CKPT=${CKPT_PATH}/${TASK}/latest.pth
GPUS=4
GPUS_PER_NODE=4
XNODE=SH-IDC1-10-140-0-[230,232,240,255],SH-IDC1-10-140-1-[45,53,64,65,78]
PORT=29504

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
TF_CPP_MIN_LOG_LEVEL=3 \
srun -p ${PARTITION} -x ${XNODE}\
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    --quotatype=reserved \
    python -u tools/test.py ${CONFIG} $CKPT --out ${WORK_DIR}/val.pkl \
    --eval waymo --options 'pklfile_prefix='${WORK_DIR}'/kitti_results' \
    --launcher='slurm'
