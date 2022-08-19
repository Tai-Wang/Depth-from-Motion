n=1
n1=1
part=mm_det
name=convert
mkdir -p logs/${name}
PYTHONPATH="$(dirname $0)":$PYTHONPATH \
MMCV_WITH_OPS=1 srun --mpi=pmi2 --gres=gpu:${n1} -w SH-IDC1-10-140-1-87 \
        -p $part -n${n} \
        --ntasks-per-node=${n1} \
        -J $name -K \
        --quotatype=reserved \
        python tools/model_converters/convert_dfm_checkpoints.py \
            configs/dfm/dfm_r34_1x8_kitti-3d-3class.py \
            /mnt/lustre/wangtai/LIGA-DfM/outputs/configs_stereo_kitti_models/best/ckpt/checkpoint_epoch_54.pth \
            --out /mnt/lustre/wangtai/mmdet3d-prerelease/work_dirs/dfm/mmdet3d-dfm-final.pth
