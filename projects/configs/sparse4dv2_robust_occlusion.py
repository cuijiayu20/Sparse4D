"""
Sparse4Dv2 Occlusion Robustness Test Config
用于遮挡测试 (exp1, exp2, exp3, exp5)。

遮挡级别说明：
  exp1 (occlusion_level=1): S1 轻微遮挡
  exp2 (occlusion_level=2): S2 中等遮挡
  exp3 (occlusion_level=3): S3 严重遮挡
  exp5 (occlusion_level=5): S4 极端遮挡

使用方法：
  bash local_test.sh sparse4dv2_robust_occlusion ckpts/sparse4dv2_r50_HInf_256x704.pth \
    --cfg-options occlusion_level=1
"""
_base_ = ['./sparse4dv2_r50_HInf_256x704.py']

# ================== Occlusion parameters ========================
occlusion_level = 3
noise_pkl_file = 'data/nuscenes/nuscenes_infos_val_with_noise.pkl'
mask_dir = 'robust_benchmark/Occlusion_mask'

# ================== Override test pipeline ========================
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    # 遮挡 mask 叠加 (在 Resize 之前)
    dict(
        type="LoadMaskMultiViewImage",
        noise_nuscenes_ann_file=noise_pkl_file,
        mask_file=mask_dir,
        occlusion_level=occlusion_level,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="DefaultFormatBundle3D",
        class_names=[
            "car", "truck", "construction_vehicle", "bus", "trailer",
            "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
        ],
        with_label=False,
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect3D",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]

data = dict(
    test=dict(
        ann_file='data/nuscenes/nuscenes2d_temporal_infos_val.pkl',
        pipeline=test_pipeline,
    ),
)
