n=1
n1=1
part=mm_det
name=convert
mkdir -p logs/${name}
PYTHONPATH="$(dirname $0)":$PYTHONPATH \
MMCV_WITH_OPS=1 srun --mpi=pmi2 --gres=gpu:${n1} \
        -p $part -n${n} \
        --ntasks-per-node=${n1} \
        -J $name -K \
        --quotatype=spot \
        python tools/model_converters/convert_dfm_checkpoints.py configs/dfm/dfm-baseline.py checkpoints/dfm/liga-dfm.pth \
            --out checkpoints/dfm/mmdet3d-dfm.pth
