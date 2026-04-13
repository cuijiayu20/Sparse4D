#!/bin/bash
# ==============================================================
# Sparse4Dv2 鲁棒性测试 - 自动化运行脚本
# ==============================================================
# 使用方法:
#   bash run_robust_test.sh [test_type]
#
# 参数:
#   test_type:  测试类型 (可选)
#     all       - 运行所有测试 (默认)
#     baseline  - 仅运行 baseline
#     drop      - 仅运行丢帧测试
#     noise     - 仅运行外参扰动测试
#     occlusion - 仅运行遮挡测试
#
# 示例:
#   bash run_robust_test.sh all
#   bash run_robust_test.sh drop
# ==============================================================

set -e
export PYTHONPATH=$PYTHONPATH:./

TEST_TYPE=${1:-all}

# ==================== 路径配置 ====================
CHECKPOINT="ckpts/sparse4dv2_r50_HInf_256x704.pth"

# 配置文件
CONFIG_ROBUST="projects/configs/sparse4dv2_robust_test.py"
CONFIG_OCCLUSION="projects/configs/sparse4dv2_robust_occlusion.py"

# 噪声 PKL 文件 (位于 data/nuscenes/ 下)
NOISE_PKL="data/nuscenes/nuscenes_infos_val_with_noise.pkl"
DROP_PKL="data/nuscenes/nuscenes_infos_val_with_noise_Drop.pkl"

# GPU 配置
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
gpus=(${CUDA_VISIBLE_DEVICES//,/ })
gpu_num=${#gpus[@]}

# 结果输出目录
RESULT_DIR="work_dirs/robust_results"
mkdir -p ${RESULT_DIR}

# ==================== 测试函数 ====================

run_test() {
    local config=$1
    local result_name=$2
    shift 2
    local cfg_options="$@"

    local result_dir="${RESULT_DIR}/${result_name}"
    mkdir -p ${result_dir}

    echo "=========================================="
    echo "[TEST] ${result_name}"
    echo "  Config: ${config}"
    echo "  Checkpoint: ${CHECKPOINT}"
    echo "  Result dir: ${result_dir}"
    if [ -n "$cfg_options" ]; then
        echo "  Options: ${cfg_options}"
    fi
    echo "=========================================="

    if [ ${gpu_num} -gt 1 ]; then
        bash ./tools/dist_test.sh \
            ${config} \
            ${CHECKPOINT} \
            ${gpu_num} \
            --eval bbox \
            --eval-options jsonfile_prefix=${result_dir} \
            ${cfg_options:+--cfg-options ${cfg_options}}
    else
        python ./tools/test.py \
            ${config} \
            ${CHECKPOINT} \
            --eval bbox \
            --eval-options jsonfile_prefix=${result_dir} \
            ${cfg_options:+--cfg-options ${cfg_options}}
    fi

    echo "[DONE] ${result_name}"
    echo ""
}

# ==================== Baseline 测试 ====================
run_baseline() {
    echo "############################################"
    echo "# Running Baseline (Clean) Test"
    echo "############################################"
    run_test ${CONFIG_ROBUST} "baseline"
}

# ==================== 丢帧测试 ====================
run_drop_tests() {
    echo "############################################"
    echo "# Running Frame Drop Tests (10%-90%)"
    echo "############################################"

    for ratio in 10 20 30 40 50 60 70 80 90; do
        run_test ${CONFIG_ROBUST} \
            "drop_${ratio}" \
            "data.test.drop_frames=True data.test.drop_ratio=${ratio} data.test.drop_type=discrete data.test.noise_nuscenes_ann_file=${DROP_PKL}"
    done
}

# ==================== 外参扰动测试 ====================
run_noise_tests() {
    echo "############################################"
    echo "# Running Extrinsic Noise Tests (L1-L4)"
    echo "############################################"

    # 单摄像头扰动 (single)
    echo "--- Single Camera Noise ---"
    for level in L1 L2 L3 L4; do
        run_test ${CONFIG_ROBUST} \
            "noise_single_${level}" \
            "data.test.extrinsics_noise=True data.test.extrinsics_noise_type=single data.test.extrinsics_noise_level=${level} data.test.noise_nuscenes_ann_file=${NOISE_PKL}"
    done

    # 多摄像头扰动 (all)
    echo "--- All Camera Noise ---"
    for level in L1 L2 L3 L4; do
        run_test ${CONFIG_ROBUST} \
            "noise_all_${level}" \
            "data.test.extrinsics_noise=True data.test.extrinsics_noise_type=all data.test.extrinsics_noise_level=${level} data.test.noise_nuscenes_ann_file=${NOISE_PKL}"
    done
}

# ==================== 遮挡测试 ====================
run_occlusion_tests() {
    echo "############################################"
    echo "# Running Occlusion Tests (exp1,2,3,5)"
    echo "############################################"

    # exp1 (S1 轻微遮挡, power=1)
    run_test ${CONFIG_OCCLUSION} "occlusion_exp1" "occlusion_level=1"

    # exp2 (S2 中等遮挡, power=2)
    run_test ${CONFIG_OCCLUSION} "occlusion_exp2" "occlusion_level=2"

    # exp3 (S3 严重遮挡, power=3)
    run_test ${CONFIG_OCCLUSION} "occlusion_exp3" "occlusion_level=3"

    # exp5 (S4 极端遮挡, power=5)
    run_test ${CONFIG_OCCLUSION} "occlusion_exp5" "occlusion_level=5"
}

# ==================== 主流程 ====================
echo "============================================================"
echo " Sparse4Dv2 Robustness Benchmark"
echo " Checkpoint: ${CHECKPOINT}"
echo " Test type:  ${TEST_TYPE}"
echo " GPUs:       ${gpu_num} (${CUDA_VISIBLE_DEVICES})"
echo " Result dir: ${RESULT_DIR}"
echo "============================================================"
echo ""

case ${TEST_TYPE} in
    baseline)
        run_baseline
        ;;
    drop)
        run_drop_tests
        ;;
    noise)
        run_noise_tests
        ;;
    occlusion)
        run_occlusion_tests
        ;;
    all)
        run_baseline
        run_drop_tests
        run_noise_tests
        run_occlusion_tests
        ;;
    *)
        echo "Unknown test type: ${TEST_TYPE}"
        echo "Valid options: all | baseline | drop | noise | occlusion"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo " All ${TEST_TYPE} tests completed!"
echo " Results saved to: ${RESULT_DIR}"
echo ""
echo " To collect results:"
echo "   python collect_robust_results.py --result-dir ${RESULT_DIR}"
echo "============================================================"
