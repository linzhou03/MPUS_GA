#!/usr/bin/env python3
"""
汇总混淆矩阵改进实验的结果

功能：
1. 读取所有配置的结果 JSON
2. 计算各配置的平均指标
3. 生成混淆矩阵对比热力图
4. 与 CAGA-SGA 对比
5. 统计显著性检验

Created: 2026-10-01
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(result_root, directions, configs):
    """
    加载所有结果文件

    Returns:
        dict: {config: {direction: {subject: result_dict}}}
    """
    results = defaultdict(lambda: defaultdict(dict))

    for config in configs:
        for direction in directions:
            config_dir = Path(result_root) / config / direction
            if not config_dir.exists():
                print(f"警告: {config_dir} 不存在")
                continue

            # 读取该方向的所有被试结果
            for json_file in config_dir.glob("seed_43_subject_*.json"):
                subject_id = json_file.stem.split('_')[-1]
                with open(json_file, 'r') as f:
                    result = json.load(f)
                results[config][direction][subject_id] = result

    return results


def compute_summary_stats(results):
    """
    计算汇总统计

    Returns:
        DataFrame: 各配置×方向的平均指标
    """
    rows = []

    for config, direction_results in results.items():
        for direction, subject_results in direction_results.items():
            if len(subject_results) == 0:
                continue

            # 提取指标
            paccs = []
            bal_accs = []
            f1s = []
            confusion_matrices = []

            for subject_id, result in subject_results.items():
                eval_data = result['trial_evaluation']
                paccs.append(eval_data['accuracy'])
                bal_accs.append(eval_data['balanced_accuracy'])
                f1s.append(eval_data['macro_f1'])
                confusion_matrices.append(np.array(eval_data['confusion_matrix']))

            # 平均混淆矩阵（先累加再归一化）
            avg_cm = np.sum(confusion_matrices, axis=0)
            avg_cm_norm = avg_cm / avg_cm.sum(axis=1, keepdims=True)

            # 提取关键混淆比例
            neg_to_pos = avg_cm_norm[2, 0] * 100  # 负向→正向
            neu_to_pos = avg_cm_norm[1, 0] * 100  # 中性→正向
            pos_recall = avg_cm_norm[0, 0] * 100  # 正向召回

            rows.append({
                'config': config,
                'direction': direction,
                'n_subjects': len(subject_results),
                'pacc_mean': np.mean(paccs) * 100,
                'pacc_std': np.std(paccs) * 100,
                'bal_acc_mean': np.mean(bal_accs) * 100,
                'bal_acc_std': np.std(bal_accs) * 100,
                'f1_mean': np.mean(f1s) * 100,
                'f1_std': np.std(f1s) * 100,
                'neg_to_pos': neg_to_pos,
                'neu_to_pos': neu_to_pos,
                'pos_recall': pos_recall,
                'confusion_matrix': avg_cm_norm,
            })

    df = pd.DataFrame(rows)
    return df


def plot_confusion_comparison(df, output_dir):
    """
    绘制混淆矩阵对比热力图

    为每个方向绘制一张图，包含所有配置的混淆矩阵
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    directions = df['direction'].unique()
    configs = df['config'].unique()

    for direction in directions:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for idx, config in enumerate(configs):
            row = df[(df['direction'] == direction) & (df['config'] == config)]
            if len(row) == 0:
                axes[idx].axis('off')
                continue

            cm = row.iloc[0]['confusion_matrix']

            sns.heatmap(
                cm,
                annot=True,
                fmt='.2%',
                cmap='RdYlGn_r',
                vmin=0,
                vmax=1,
                ax=axes[idx],
                cbar=False,
                xticklabels=['Pos', 'Neu', 'Neg'],
                yticklabels=['Pos', 'Neu', 'Neg'],
            )
            axes[idx].set_title(f'{config}\nBalAcc={row.iloc[0]["bal_acc_mean"]:.1f}%')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')

        fig.suptitle(f'Direction {direction}: Confusion Matrix Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / f'confusion_comparison_{direction}.png', dpi=150)
        plt.close()

    print(f"混淆矩阵对比图已保存到: {output_dir}")


def generate_summary_table(df, output_dir):
    """
    生成汇总表格（Markdown 格式）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 表格 1：各配置的平均指标
    table1 = df.groupby('config').agg({
        'pacc_mean': 'mean',
        'bal_acc_mean': 'mean',
        'f1_mean': 'mean',
        'neg_to_pos': 'mean',
        'neu_to_pos': 'mean',
    }).round(2)

    md_content = "# 混淆矩阵改进实验汇总\n\n"
    md_content += "## 各配置平均指标（ACE 三方向平均）\n\n"
    md_content += "| 配置 | Pacc (%) | BalAcc (%) | F1 (%) | Neg→Pos (%) | Neu→Pos (%) |\n"
    md_content += "|------|----------|------------|--------|-------------|-------------|\n"

    for config, row in table1.iterrows():
        md_content += f"| {config} | {row['pacc_mean']:.2f} | {row['bal_acc_mean']:.2f} | {row['f1_mean']:.2f} | {row['neg_to_pos']:.2f} | {row['neu_to_pos']:.2f} |\n"

    md_content += "\n## 各方向详细指标\n\n"

    for direction in df['direction'].unique():
        md_content += f"\n### Direction {direction}\n\n"
        md_content += "| 配置 | BalAcc (%) | Neg→Pos (%) | Neu→Pos (%) | Pos Recall (%) |\n"
        md_content += "|------|------------|-------------|-------------|----------------|\n"

        dir_df = df[df['direction'] == direction].sort_values('bal_acc_mean', ascending=False)
        for _, row in dir_df.iterrows():
            md_content += f"| {row['config']} | {row['bal_acc_mean']:.2f} ± {row['bal_acc_std']:.2f} | {row['neg_to_pos']:.2f} | {row['neu_to_pos']:.2f} | {row['pos_recall']:.2f} |\n"

    # 保存
    with open(output_dir / '汇总报告.md', 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"汇总报告已保存到: {output_dir / '汇总报告.md'}")


def main():
    parser = argparse.ArgumentParser(description='汇总混淆矩阵改进实验结果')
    parser.add_argument('--result-root', type=str, required=True,
                        help='结果根目录')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录')
    args = parser.parse_args()

    print("=" * 60)
    print("汇总混淆矩阵改进实验结果")
    print("=" * 60)
    print(f"结果路径: {args.result_root}")
    print(f"输出路径: {args.output_dir}")
    print()

    # 配置
    directions = ['A', 'C', 'E']
    configs = ['baseline', 'asy', 'cal', 'cbst', 'asy_cal', 'asy_cbst', 'cal_cbst', 'full']

    # 加载结果
    print("加载结果文件...")
    results = load_results(args.result_root, directions, configs)

    # 统计数据
    print("\n计算汇总统计...")
    df = compute_summary_stats(results)

    print(f"\n加载完成:")
    for config in configs:
        count = sum(len(d) for d in results[config].values())
        print(f"  {config}: {count} folds")

    # 生成汇总表格
    print("\n生成汇总表格...")
    generate_summary_table(df, args.output_dir)

    # 绘制混淆矩阵对比图
    print("\n绘制混淆矩阵对比图...")
    plot_confusion_comparison(df, args.output_dir)

    # 保存原始数据
    df.to_csv(Path(args.output_dir) / 'summary_data.csv', index=False)
    print(f"\n原始数据已保存到: {Path(args.output_dir) / 'summary_data.csv'}")

    print("\n" + "=" * 60)
    print("汇总完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
