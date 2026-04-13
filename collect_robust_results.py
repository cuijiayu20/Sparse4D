#!/usr/bin/env python3
"""
Sparse4D 鲁棒性测试结果收集与 RDRR 计算脚本

使用方法:
    python collect_robust_results.py [--result-dir RESULT_DIR] [--baseline-dir BASELINE_DIR]

输出:
    1. 终端打印表格形式结果
    2. 导出完整详细指标到 CSV 文件 (包含各类 mAP, mATE, mASE 等和逐类别指标)
"""

import os
import json
import argparse
import csv
from collections import OrderedDict


def load_full_metrics(result_dir):
    """从 metrics_summary.json 加载完整的评估指标"""
    metrics_file = os.path.join(result_dir, 'metrics_summary.json')
    if not os.path.exists(metrics_file):
        for root, dirs, files in os.walk(result_dir):
            if 'metrics_summary.json' in files:
                metrics_file = os.path.join(root, 'metrics_summary.json')
                break

    if not os.path.exists(metrics_file):
        return None

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    return metrics


def extract_key_metrics(metrics):
    """提取终端打印用到的核心指标"""
    if not metrics:
        return None, None
    nds = metrics.get('nd_score', None)
    mAP = metrics.get('mean_ap', None)
    return nds, mAP

def compute_rdrr(our_metric, our_baseline, baseline_metric, baseline_baseline):
    """计算 RDRR"""
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

    header_str = " | ".join(str(h).center(w) for h, w in zip(headers, col_widths))
    print(f"  {header_str}")
    print(f"  {'-+-'.join('-'*w for w in col_widths)}")

    for row in rows:
        row_str = " | ".join(str(v).center(w) for v, w in zip(row, col_widths))
        print(f"  {row_str}")
    print()

def export_to_csv(all_results, csv_path):
    """将所有搜集到的结果导出为 CSV (Excel可打开)"""
    if not all_results:
        return
        
    print(f"\n正在将详细结果导出到 Excel(CSV): {csv_path}")
    
    # 类别列表 (NuScenes 标准类别)
    classes = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
               'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
               
    # 准备 CSV Header
    headers = [
        'Experiment', 'Sub-Level', 'NDS', 'mAP', 
        'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE'
    ]
    # 加上逐类别 AP
    for cls in classes:
        headers.append(f'{cls}_AP')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for exp_group_name, exps in all_results.items():
            for exp_name, metrics in exps.items():
                if metrics is None:
                    continue
                
                nds = metrics.get('nd_score', 0)
                mAP = metrics.get('mean_ap', 0)
                tp_errs = metrics.get('tp_errors', {})
                
                mATE = tp_errs.get('trans_err', 0)
                mASE = tp_errs.get('scale_err', 0)
                mAOE = tp_errs.get('orient_err', 0)
                mAVE = tp_errs.get('vel_err', 0)
                mAAE = tp_errs.get('attr_err', 0)
                
                row = [
                    exp_group_name, exp_name, 
                    f"{nds:.4f}", f"{mAP:.4f}",
                    f"{mATE:.4f}", f"{mASE:.4f}", f"{mAOE:.4f}", f"{mAVE:.4f}", f"{mAAE:.4f}"
                ]
                
                label_aps = metrics.get('label_aps', {})
                for cls in classes:
                    # 获取各类别的平均 AP (通常对各个距离阈值求均值)
                    cls_ap_dict = label_aps.get(cls, {})
                    if cls_ap_dict:
                        avg_ap = sum(cls_ap_dict.values()) / len(cls_ap_dict)
                    else:
                        avg_ap = 0.0
                    row.append(f"{avg_ap:.4f}")
                
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Collect robustness test results')
    parser.add_argument('--result-dir', type=str,
                        default='work_dirs/robust_results',
                        help='Root directory of test results')
    parser.add_argument('--baseline-nds', type=float, default=None)
    parser.add_argument('--baseline-mAP', type=float, default=None)
    parser.add_argument('--csv-out', type=str, default='work_dirs/robust_results/detailed_results.csv',
                        help='Output path for the detailed CSV file')
    args = parser.parse_args()

    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        print(f"[ERROR] Result directory not found: {result_dir}")
        return

    print("=" * 80)
    print("  Sparse4D 鲁棒性测试结果汇总")
    print("=" * 80)

    # 保存全量数据用于写入 CSV
    all_experiments_data = OrderedDict()

    # 1. Baseline
    baseline_metrics = load_full_metrics(os.path.join(result_dir, 'baseline'))
    all_experiments_data['Baseline'] = {'baseline': baseline_metrics}
    
    baseline_nds, baseline_mAP = extract_key_metrics(baseline_metrics)
    if baseline_nds:
        print(f"\n  Our Baseline: NDS={baseline_nds:.4f}, mAP={baseline_mAP:.4f}")

    # 2. Frame Drop
    all_experiments_data['Frame Drop'] = OrderedDict()
    drop_rows = []
    if baseline_nds:
        drop_rows.append(['0%', f"{baseline_mAP:.4f}", f"{baseline_nds:.4f}", "N/A", "N/A", "0.00%", "0.00%"])
    for ratio in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        metrics = load_full_metrics(os.path.join(result_dir, f'drop_{ratio}'))
        all_experiments_data['Frame Drop'][f'drop_{ratio}'] = metrics
        nds, mAP = extract_key_metrics(metrics)
        if nds:
            drop_rows.append([f"{ratio}%", f"{mAP:.4f}", f"{nds:.4f}", "N/A", "N/A", "N/A", "N/A"])
    if drop_rows:
        print_table("表2: 丢帧鲁棒性测试结果", ["Frame Drop Rate", "Our_mAP↑", "Our_NDS↑", "Baseline_NDS↓", "Baseline_mAP", "NDS_RDRR", "mAP_RDRR"], drop_rows)


    # 3. Noise Single
    all_experiments_data['Noise Single'] = OrderedDict()
    ns_rows = []
    if baseline_nds:
        ns_rows.append(['L0', '0.0', '0.0', f"{baseline_mAP:.4f}", f"{baseline_nds:.4f}"])
    noise_mapping = {'L1': ('0.5', '0.3'), 'L2': ('1.0', '0.5'), 'L3': ('2.0', '1.0'), 'L4': ('5.0', '2.0')}
    for level in ['L1', 'L2', 'L3', 'L4']:
        metrics = load_full_metrics(os.path.join(result_dir, f'noise_single_{level}'))
        all_experiments_data['Noise Single'][f'L{level}'] = metrics
        nds, mAP = extract_key_metrics(metrics)
        if nds:
            ns_rows.append([level, noise_mapping[level][0], noise_mapping[level][1], f"{mAP:.4f}", f"{nds:.4f}"])
    if ns_rows:
        print_table("外参扰动 - 单摄像头", ["Level", "旋转噪声(°)", "平移噪声(cm)", "mAP↑", "NDS↑"], ns_rows)


    # 4. Noise All
    all_experiments_data['Noise All'] = OrderedDict()
    na_rows = []
    if baseline_nds:
        na_rows.append(['L0', '0.0', '0.0', f"{baseline_mAP:.4f}", f"{baseline_nds:.4f}"])
    for level in ['L1', 'L2', 'L3', 'L4']:
        metrics = load_full_metrics(os.path.join(result_dir, f'noise_all_{level}'))
        all_experiments_data['Noise All'][f'L{level}'] = metrics
        nds, mAP = extract_key_metrics(metrics)
        if nds:
            na_rows.append([level, noise_mapping[level][0], noise_mapping[level][1], f"{mAP:.4f}", f"{nds:.4f}"])
    if na_rows:
        print_table("外参扰动 - 多摄像头", ["Level", "旋转噪声(°)", "平移噪声(cm)", "mAP↑", "NDS↑"], na_rows)


    # 5. Occlusion
    all_experiments_data['Occlusion'] = OrderedDict()
    occ_rows = []
    occ_map = {'occlusion_exp1': 'S1 轻微遮挡', 'occlusion_exp2': 'S2 中等遮挡', 'occlusion_exp3': 'S3 严重遮挡', 'occlusion_exp5': 'S4 极端遮挡'}
    if baseline_nds:
        occ_rows.append(['S0 无遮挡', f"{baseline_mAP:.4f}", f"{baseline_nds:.4f}", "N/A", "N/A", "N/A", "N/A"])
    for k, v in occ_map.items():
        metrics = load_full_metrics(os.path.join(result_dir, k))
        all_experiments_data['Occlusion'][v] = metrics
        nds, mAP = extract_key_metrics(metrics)
        if nds:
            occ_rows.append([v, f"{mAP:.4f}", f"{nds:.4f}", "N/A", "N/A", "N/A", "N/A"])
    if occ_rows:
        print_table("遮挡测试结果", ["类型", "Our_mAP↑", "Our_NDS↑", "Baseline_NDS↓", "Baseline_mAP", "NDS_RDRR", "mAP_RDRR"], occ_rows)
        
    # 导出到 CSV
    export_to_csv(all_experiments_data, args.csv_out)
    print(f"\n[完成] 详细指标(包含 mATE/mASE/Per-Class AP)已存入 {args.csv_out}")

if __name__ == '__main__':
    main()
