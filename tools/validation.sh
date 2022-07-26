n=1
n1=1
part=mm_det
name=install
mkdir -p logs/${name}
PYTHONPATH="$(dirname $0)":$PYTHONPATH \
MMCV_WITH_OPS=1 srun --mpi=pmi2 --gres=gpu:${n1} \
        -p $part -n${n} \
        --ntasks-per-node=${n1} \
        -J $name -K \
        --quotatype=spot \
        python demo/pcd_demo.py demo/data/kitti/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth
