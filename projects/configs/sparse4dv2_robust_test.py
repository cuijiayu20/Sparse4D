"""
Sparse4Dv2 Robustness Test Config
用于丢帧测试和外参扰动测试。

使用方法：
  # Clean baseline (无噪声)
  bash local_test.sh sparse4dv2_robust_test ckpts/sparse4dv2_r50_HInf_256x704.pth

  # 丢帧测试 (ratio=10,20,...,90)
  bash local_test.sh sparse4dv2_robust_test ckpts/sparse4dv2_r50_HInf_256x704.pth \
    --cfg-options data.test.drop_frames=True data.test.drop_ratio=10 \
    data.test.noise_nuscenes_ann_file=data/nuscenes/nuscenes_infos_val_with_noise_Drop.pkl

  # 外参扰动 - 单摄像头 (level=L1,L2,L3,L4)
  bash local_test.sh sparse4dv2_robust_test ckpts/sparse4dv2_r50_HInf_256x704.pth \
    --cfg-options data.test.extrinsics_noise=True \
    data.test.extrinsics_noise_type=single \
    data.test.extrinsics_noise_level=L1 \
    data.test.noise_nuscenes_ann_file=data/nuscenes/nuscenes_infos_val_with_noise.pkl

  # 外参扰动 - 多摄像头 (level=L1,L2,L3,L4)
  bash local_test.sh sparse4dv2_robust_test ckpts/sparse4dv2_r50_HInf_256x704.pth \
    --cfg-options data.test.extrinsics_noise=True \
    data.test.extrinsics_noise_type=all \
    data.test.extrinsics_noise_level=L1 \
    data.test.noise_nuscenes_ann_file=data/nuscenes/nuscenes_infos_val_with_noise.pkl
"""
_base_ = ['./sparse4dv2_r50_HInf_256x704.py']

# ================== Override test data config ========================
data = dict(
    test=dict(
        ann_file='data/nuscenes/nuscenes2d_temporal_infos_val.pkl',
        # ===== Robustness test parameters =====
        noise_nuscenes_ann_file='',
        extrinsics_noise=False,
        extrinsics_noise_type='single',
        extrinsics_noise_level=None,
        drop_frames=False,
        drop_ratio=0,
        drop_type='discrete',
        noise_sensor_type='camera',
    ),
)
