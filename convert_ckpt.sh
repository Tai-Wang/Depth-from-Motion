n=1
n1=1
part=mm_det
name=convert
mkdir -p logs/${name}
PYTHONPATH="$(dirname $0)":$PYTHONPATH \
MMCV_WITH_OPS=1 srun --mpi=pmi2 --gres=gpu:${n1} \
        -p $part -n${n} -w SH-IDC1-10-140-1-117\
        --ntasks-per-node=${n1} \
        -J $name -K \
        --quotatype=spot \
        python tools/model_converters/convert_dfm_checkpoints.py \
            configs/dfm/dfm-baseline-depth-syncbn-fgmask-usevan-imit.py \
            /mnt/lustre/wangtai.vendor/mmdet3d-prerelease/work_dirs/dfm/liga-dfm.pth \
            --out /mnt/lustre/wangtai.vendor/mmdet3d-prerelease/work_dirs/dfm/mmdet3d-dfm-imit.pth
