#!/usr/bin/env python3
"""
Sparse4D 鲁棒性测试结果收集与 RDRR 计算脚本

使用方法:
    python collect_robust_results.py [--result-dir RESULT_DIR] [--baseline-dir BASELINE_DIR]

输出:
    1. 丢帧测试结果表 (表2)
    2. 外参扰动测试结果表 - 单摄像头 (表3/4)
    3. 外参扰动测试结果表 - 多摄像头 (表3/4)
    4. 遮挡测试结果表 (表5)
"""

import os
import json
import argparse
from collections import OrderedDict


def load_metrics(result_dir):
    """从 metrics_summary.json 加载 NDS 和 mAP"""
    metrics_file = os.path.join(result_dir, 'metrics_summary.json')
    if not os.path.exists(metrics_file):
        # 尝试在子目录中查找
        for root, dirs, files in os.walk(result_dir):
            if 'metrics_summary.json' in files:
                metrics_file = os.path.join(root, 'metrics_summary.json')
                break

    if not os.path.exists(metrics_file):
        return None, None

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    nds = metrics.get('nd_score', None)
    mAP = metrics.get('mean_ap', None)
    return nds, mAP


def compute_rdrr(our_metric, our_baseline, baseline_metric, baseline_baseline):
    """
    计算 RDRR (Relative Degradation Rate Reduction)

    RDRR = ((baseline_baseline - baseline_metric) - (our_baseline - our_metric))
           / (baseline_baseline - baseline_metric) * 100%

    简化: RDRR = 1 - (our_baseline - our_metric) / (baseline_baseline - baseline_metric)
    """
    baseline_deg = baseline_baseline - baseline_metric
    our_deg = our_baseline - our_metric

    if abs(baseline_deg) < 1e-8:
        return 0.0

    rdrr = (baseline_deg - our_deg) / baseline_deg * 100.0
    return rdrr


def print_table(title, headers, rows, col_widths=None):
    """打印格式化表格"""
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows if i < len(r)))
                      for i, h in enumerate(headers)]
        col_widths = [w + 2 for w in col_widths]

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    # Header
    header_str = " | ".join(str(h).center(w) for h, w in zip(headers, col_widths))
    print(f"  {header_str}")
    print(f"  {'-+-'.join('-'*w for w in col_widths)}")

    # Rows
    for row in rows:
        row_str = " | ".join(str(v).center(w) for v, w in zip(row, col_widths))
        print(f"  {row_str}")

    print()


def collect_frame_drop_results(result_dir, baseline_nds=None, baseline_mAP=None):
    """收集丢帧测试结果"""
    print("\n" + "="*80)
    print("  表2: 丢帧鲁棒性测试结果 (nuScenes val set)")
    print("="*80)

    headers = ["Frame Drop Rate", "Our_mAP↑", "Our_NDS↑",
               "Baseline_NDS↓", "Baseline_mAP",
               "NDS_RDRR", "mAP_RDRR"]

    rows = []

    # 0% (baseline)
    our_nds_0, our_mAP_0 = load_metrics(os.path.join(result_dir, 'baseline'))
    if our_nds_0 is not None:
        rows.append([
            "0%",
            f"{our_mAP_0:.4f}" if our_mAP_0 else "N/A",
            f"{our_nds_0:.4f}" if our_nds_0 else "N/A",
            f"{baseline_nds:.4f}" if baseline_nds else "N/A",
            f"{baseline_mAP:.4f}" if baseline_mAP else "N/A",
            "0.00%",
            "0.00%",
        ])

    for ratio in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        nds, mAP = load_metrics(os.path.join(result_dir, f'drop_{ratio}'))
        if nds is None:
            continue

        nds_rdrr = "--"
        mAP_rdrr = "--"
        if baseline_nds and our_nds_0 and baseline_mAP and our_mAP_0:
            # 需要 baseline 模型在相同丢帧率下的结果来计算 RDRR
            # 这里先显示绝对值，RDRR 需要 baseline 对比模型的数据
            nds_rdrr = "N/A"
            mAP_rdrr = "N/A"

        rows.append([
            f"{ratio}%",
            f"{mAP:.4f}",
            f"{nds:.4f}",
            "N/A",  # baseline model results needed
            "N/A",
            nds_rdrr,
            mAP_rdrr,
        ])

    if rows:
        col_widths = [16, 10, 10, 14, 14, 10, 10]
        print_table("丢帧测试结果", headers, rows, col_widths)
    else:
        print("  [WARNING] 未找到丢帧测试结果")

    return rows


def collect_noise_results(result_dir, noise_type='single'):
    """收集外参扰动测试结果"""
    type_name = "单摄像头" if noise_type == 'single' else "多摄像头"
    print(f"\n  外参扰动 - {type_name}")

    headers = ["Level", "旋转噪声(°)", "平移噪声(cm)", "mAP↑", "NDS↑"]

    noise_levels = {
        'L0': ('0.0', '0.0'),
        'L1': ('0.5', '0.3'),
        'L2': ('1.0', '0.5'),
        'L3': ('2.0', '1.0'),
        'L4': ('5.0', '2.0'),
    }

    rows = []

    # L0 = baseline
    nds_0, mAP_0 = load_metrics(os.path.join(result_dir, 'baseline'))
    if nds_0 is not None:
        rows.append([
            'L0', '0.0', '0.0',
            f"{mAP_0:.4f}", f"{nds_0:.4f}",
        ])

    for level in ['L1', 'L2', 'L3', 'L4']:
        dir_name = f'noise_{noise_type}_{level}'
        nds, mAP = load_metrics(os.path.join(result_dir, dir_name))
        if nds is None:
            continue

        rot, trans = noise_levels[level]
        rows.append([
            level, rot, trans,
            f"{mAP:.4f}", f"{nds:.4f}",
        ])

    if rows:
        col_widths = [8, 12, 12, 10, 10]
        print_table(f"外参扰动 - {type_name}", headers, rows, col_widths)
    else:
        print(f"  [WARNING] 未找到 {type_name} 扰动测试结果")

    return rows


def collect_occlusion_results(result_dir):
    """收集遮挡测试结果"""
    headers = ["类型", "Our_mAP↑", "Our_NDS↑",
               "Baseline_NDS↓", "Baseline_mAP",
               "NDS_RDRR", "mAP_RDRR"]

    occlusion_types = OrderedDict([
        ('baseline', 'S0 无遮挡'),
        ('occlusion_exp1', 'S1 轻微遮挡'),
        ('occlusion_exp2', 'S2 中等遮挡'),
        ('occlusion_exp3', 'S3 严重遮挡'),
        ('occlusion_exp5', 'S4 极端遮挡'),
    ])

    rows = []
    for dir_name, label in occlusion_types.items():
        nds, mAP = load_metrics(os.path.join(result_dir, dir_name))
        if nds is None:
            continue

        rows.append([
            label,
            f"{mAP:.4f}",
            f"{nds:.4f}",
            "N/A",  # baseline model
            "N/A",
            "N/A",
            "N/A",
        ])

    if rows:
        col_widths = [14, 10, 10, 14, 14, 10, 10]
        print_table("遮挡测试结果", headers, rows, col_widths)
    else:
        print("  [WARNING] 未找到遮挡测试结果")

    return rows


def main():
    parser = argparse.ArgumentParser(description='Collect robustness test results')
    parser.add_argument('--result-dir', type=str,
                        default='work_dirs/robust_results',
                        help='Root directory of test results')
    parser.add_argument('--baseline-nds', type=float, default=None,
                        help='Baseline model NDS (for RDRR computation)')
    parser.add_argument('--baseline-mAP', type=float, default=None,
                        help='Baseline model mAP (for RDRR computation)')
    args = parser.parse_args()

    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        print(f"[ERROR] Result directory not found: {result_dir}")
        return

    print("=" * 80)
    print("  Sparse4D 鲁棒性测试结果汇总")
    print("=" * 80)

    # 加载 baseline 结果
    baseline_nds, baseline_mAP = load_metrics(
        os.path.join(result_dir, 'baseline'))
    if baseline_nds:
        print(f"\n  Our Baseline: NDS={baseline_nds:.4f}, mAP={baseline_mAP:.4f}")
    if args.baseline_nds:
        print(f"  Comparison Baseline: NDS={args.baseline_nds:.4f}, "
              f"mAP={args.baseline_mAP:.4f}")

    # 收集各类测试结果
    print("\n" + "#" * 80)
    print("# 1. 丢帧鲁棒性测试")
    print("#" * 80)
    collect_frame_drop_results(result_dir, args.baseline_nds, args.baseline_mAP)

    print("\n" + "#" * 80)
    print("# 2. 外参扰动鲁棒性测试 - 单摄像头")
    print("#" * 80)
    collect_noise_results(result_dir, 'single')

    print("\n" + "#" * 80)
    print("# 3. 外参扰动鲁棒性测试 - 多摄像头")
    print("#" * 80)
    collect_noise_results(result_dir, 'all')

    print("\n" + "#" * 80)
    print("# 4. 遮挡鲁棒性测试")
    print("#" * 80)
    collect_occlusion_results(result_dir)


if __name__ == '__main__':
    main()
