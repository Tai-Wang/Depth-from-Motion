n=1
n1=1
part=mm_det
name=install
mkdir -p logs/${name}
PYTHONPATH="$(dirname $0)":$PYTHONPATH \
LD_PRELOAD=/mnt/cache/share/spring/envs/r0.3.0/lib/libpython3.6m.so.1.0:$LD_PRELOAD \
MMCV_WITH_OPS=1 srun --mpi=pmi2 --gres=gpu:${n1} \
        -p $part -n${n} \
        --ntasks-per-node=${n1} \
        -J $name -K \
        --quotatype=spot \
        python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
