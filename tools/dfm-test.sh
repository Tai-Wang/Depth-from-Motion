#!/usr/bin/env bash

set -x

CKPT_PATH=/mnt/lustre/wangtai.vendor/mmdet3d-prerelease/work_dirs
PARTITION=robot
JOB_NAME=dfm-lq
TASK=dfm-lq
CONFIG=configs/dfm/dfm-baseline-lq.py
WORK_DIR=${CKPT_PATH}/${TASK}
CKPT=${CKPT_PATH}/${TASK}/mmdet3d-dfm.pth
GPUS=8
GPUS_PER_NODE=8
XNODE=SH-IDC1-10-140-0-[232,240,255],SH-IDC1-10-140-1-65
PORT=29510

mkdir logs/${TASK}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
LD_PRELOAD=/mnt/cache/share/spring/envs/r0.3.0/lib/libpython3.6m.so.1.0:$LD_PRELOAD \
srun -p ${PARTITION} -x ${XNODE}\
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    --quotatype=auto \
    python -Xfaulthandler -u tools/test.py ${CONFIG} $CKPT --out ${WORK_DIR}/refactor.pkl \
    --eval mAP --options 'pklfile_prefix='${WORK_DIR}'/refactor' --launcher='slurm' \
    2>&1 | tee logs/${TASK}/logs_tee.txt
