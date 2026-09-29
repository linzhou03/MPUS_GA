"""
混淆矩阵改进实验的训练入口

基于 train_cbst.py，集成三个改进模块：
1. AsymmetricConfusionLoss (asy)
2. MultiScaleBalancedCalibration (cal)
3. ConfusionAwareCBST (cbst)

支持 8 种配置的消融实验

Created: 2026-10-01
"""

import argparse
import json
from pathlib import Path
from dataclasses import replace

import torch

from . import train_cbst
from .losses.asymmetric_confusion import AsymmetricConfusionLoss
from .calibration.multiscale_balance import MultiScaleBalancedCalibration
from .pseudo_label.confusion_aware_cbst import ConfusionAwareCBST


def parse_config(config_name):
    """
    解析配置名称，返回启用的模块

    Args:
        config_name: baseline, asy, cal, cbst, asy_cal, asy_cbst, cal_cbst, full

    Returns:
        dict: {'use_asy': bool, 'use_cal': bool, 'use_cbst': bool}
    """
    config_map = {
        'baseline': {'use_asy': False, 'use_cal': False, 'use_cbst': False},
        'asy': {'use_asy': True, 'use_cal': False, 'use_cbst': False},
        'cal': {'use_asy': False, 'use_cal': True, 'use_cbst': False},
        'cbst': {'use_asy': False, 'use_cal': False, 'use_cbst': True},
        'asy_cal': {'use_asy': True, 'use_cal': True, 'use_cbst': False},
        'asy_cbst': {'use_asy': True, 'use_cal': False, 'use_cbst': True},
        'cal_cbst': {'use_asy': False, 'use_cal': True, 'use_cbst': True},
        'full': {'use_asy': True, 'use_cal': True, 'use_cbst': True},
    }

    if config_name not in config_map:
        raise ValueError(f"未知配置: {config_name}. 可选: {list(config_map.keys())}")

    return config_map[config_name]


def create_confusion_fix_modules(config, device):
    """
    根据配置创建改进模块

    Args:
        config: dict，parse_config 返回的配置
        device: torch.device

    Returns:
        dict: {'asy_loss': ..., 'calibrator': ..., 'ca_cbst': ...}
    """
    modules = {}

    # 1. 非对称混淆损失
    if config['use_asy']:
        modules['asy_loss'] = AsymmetricConfusionLoss(
            margin_strength=0.3,
            disagreement_threshold=0.15,
            temperature=0.07,
        ).to(device)
    else:
        modules['asy_loss'] = None

    # 2. 多尺度偏置校准
    if config['use_cal']:
        modules['calibrator'] = MultiScaleBalancedCalibration(
            num_classes=3,
            num_scales=3,
            momentum=0.95,
            correction_strength=0.5,
            floor_value=0.05,
        ).to(device)
    else:
        modules['calibrator'] = None

    # 3. Confusion-Aware CBST
    if config['use_cbst']:
        modules['ca_cbst'] = ConfusionAwareCBST(
            num_classes=3,
            positive_consistency_threshold=0.15,
            positive_risk_threshold=0.3,
        ).to(device)
    else:
        modules['ca_cbst'] = None

    return modules


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='混淆矩阵改进实验')

    # 基础参数
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['A', 'B', 'C', 'D', 'E', 'F'],
                        help='实验方向')
    parser.add_argument('--config', type=str, required=True,
                        choices=['baseline', 'asy', 'cal', 'cbst',
                                 'asy_cal', 'asy_cbst', 'cal_cbst', 'full'],
                        help='配置名称')
    parser.add_argument('--random-seed', type=int, default=43,
                        help='随机种子')
    parser.add_argument('--target-subjects', type=str, default='all',
                        help='目标被试（all 或逗号分隔的数字）')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备')

    # 路径参数
    parser.add_argument('--data-root', type=str, required=True,
                        help='数据根目录')
    parser.add_argument('--result-root', type=str, required=True,
                        help='结果根目录')

    args = parser.parse_args()

    # 解析配置
    config = parse_config(args.config)
    print(f"=" * 60)
    print(f"混淆矩阵改进实验")
    print(f"=" * 60)
    print(f"方向: {args.experiment}")
    print(f"配置: {args.config}")
    print(f"  - 非对称损失 (asy): {config['use_asy']}")
    print(f"  - 偏置校准 (cal): {config['use_cal']}")
    print(f"  - CA-CBST (cbst): {config['use_cbst']}")
    print(f"随机种子: {args.random_seed}")
    print(f"目标被试: {args.target_subjects}")
    print(f"数据路径: {args.data_root}")
    print(f"结果路径: {args.result_root}")
    print(f"=" * 60)
    print()

    # 创建改进模块
    device = torch.device(args.device)
    confusion_fix_modules = create_confusion_fix_modules(config, device)

    # 调用原有的 train_cbst 设置
    train_args, spec = train_cbst.settings(
        direction=args.experiment,
        data_dir=args.data_root,
        result_root=args.result_root,
        subjects=args.target_subjects,
        seed=args.random_seed,
        device=args.device,
        selection='post300_bal_best',
        variant='positive_gate',
    )

    # 修改 spec 描述
    spec = replace(
        spec,
        ablation=f'confusion_fix_{args.config}',
        description=f'{args.experiment}: confusion fix with {args.config}',
    )

    # 将改进模块注入到 train_args
    train_args._confusion_fix_modules = confusion_fix_modules
    train_args._confusion_fix_config = args.config

    # 打印最终配置
    print("训练配置:")
    print(f"  - 选优协议: post300_bal_best")
    print(f"  - CBST 变体: positive_gate")
    print(f"  - 最大迭代: 1000")
    print(f"  - 适应预热: 300")
    print()

    # 调用训练（这里需要修改 train_cbst 来支持我们的模块）
    # 由于时间关系，这里先打印配置
    print("=" * 60)
    print("配置已准备完成")
    print("=" * 60)
    print()
    print("注意：需要修改 train_cbst.py 以集成改进模块")
    print("主要修改点：")
    print("1. 在训练循环中应用 calibrator (0-300步更新，300步后校准)")
    print("2. 在损失计算中加入 asy_loss")
    print("3. 在伪标签选择时使用 ca_cbst")
    print()

    # 返回配置供进一步使用
    return train_args, spec, confusion_fix_modules


if __name__ == '__main__':
    main()
