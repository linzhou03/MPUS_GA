"""
增强偏置校准模块

针对目标域类别预测偏置（特别是 positive 被过度预测）的校准机制

设计思路：
1. 每个尺度独立估计目标域 class prior
2. 使用源域混淆矩阵 + 目标域软概率进行 label shift 估计
3. 如果三尺度都估计出 positive 被高估，则强力校正
4. 温和校正（强度 0.5）避免 overcorrect

Created: 2026-10-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleBalancedCalibration(nn.Module):
    """
    多尺度类别平衡校准
    """

    def __init__(
        self,
        num_classes=3,
        num_scales=3,
        momentum=0.95,
        correction_strength=0.5,
        floor_value=0.05,
    ):
        """
        Args:
            num_classes: 类别数（默认 3）
            num_scales: 尺度数（默认 3）
            momentum: EMA 动量
            correction_strength: 校正强度（0.5 为温和）
            floor_value: prior 最小值
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_scales = num_scales
        self.momentum = momentum
        self.correction_strength = correction_strength
        self.floor_value = floor_value

        # 每个尺度的源域混淆矩阵 [num_scales, num_classes, num_classes]
        # confusion[s, i, j] = P(pred=j | true=i) on source domain
        self.register_buffer(
            'source_confusion',
            torch.eye(num_classes).unsqueeze(0).repeat(num_scales, 1, 1)
        )

        # 每个尺度估计的目标域 prior [num_scales, num_classes]
        self.register_buffer(
            'target_prior',
            torch.ones(num_scales, num_classes) / num_classes
        )

        # 是否已完成源域 warmup
        self.register_buffer('warmup_done', torch.tensor(False))

    def update_source_confusion(self, scale_logits_list, source_labels):
        """
        在源域 warmup 期间（0-300步）更新源域混淆矩阵

        Args:
            scale_logits_list: list of [N, num_classes]，三尺度的 logits
            source_labels: [N] 源域真实标签
        """
        if self.warmup_done:
            return

        with torch.no_grad():
            for s, scale_logits in enumerate(scale_logits_list):
                # 预测概率
                probs = F.softmax(scale_logits, dim=1)  # [N, num_classes]

                # 按真实类别分组，计算平均预测分布
                for c in range(self.num_classes):
                    mask = (source_labels == c)
                    if mask.sum() == 0:
                        continue

                    # 该类别样本的平均预测分布
                    avg_pred = probs[mask].mean(dim=0)  # [num_classes]

                    # EMA 更新混淆矩阵的第 c 行
                    self.source_confusion[s, c] = (
                        self.momentum * self.source_confusion[s, c] +
                        (1 - self.momentum) * avg_pred
                    )

    def finish_warmup(self):
        """标记 warmup 完成，冻结源域混淆矩阵"""
        self.warmup_done = torch.tensor(True)
        # 归一化混淆矩阵的每一行
        self.source_confusion = self.source_confusion / (
            self.source_confusion.sum(dim=2, keepdim=True) + 1e-8
        )

    def estimate_target_prior(self, scale_probs_list):
        """
        估计目标域 class prior（使用 Black Box Shift Estimation）

        Args:
            scale_probs_list: list of [N, num_classes]，三尺度的目标域预测概率

        Returns:
            target_prior: [num_scales, num_classes]
        """
        with torch.no_grad():
            for s, probs in enumerate(scale_probs_list):
                # 目标域的平均预测分布
                p = probs.mean(dim=0)  # [num_classes]

                # 求解 C @ q = p，其中 C 是源域混淆矩阵，q 是目标 prior
                # 使用最小二乘求解
                C = self.source_confusion[s]  # [num_classes, num_classes]
                try:
                    q = torch.linalg.lstsq(C.T, p).solution
                except:
                    # 如果求解失败，使用均匀分布
                    q = torch.ones(self.num_classes, device=p.device) / self.num_classes

                # Floor 和归一化
                q = torch.clamp(q, min=self.floor_value)
                q = q / q.sum()

                # 更新
                self.target_prior[s] = q

    def calibrate_logits(self, scale_logits_list):
        """
        根据估计的 target prior 校准每个尺度的 logits

        Args:
            scale_logits_list: list of [N, num_classes]，三尺度的原始 logits

        Returns:
            calibrated_logits_list: list of [N, num_classes]，校准后的 logits
        """
        calibrated = []

        for s, logits in enumerate(scale_logits_list):
            # 当前预测分布
            probs = F.softmax(logits, dim=1)  # [N, num_classes]
            current_dist = probs.mean(dim=0)  # [num_classes]

            # 期望分布（估计的 target prior）
            target_dist = self.target_prior[s]  # [num_classes]

            # 计算偏置（log 空间）
            bias = torch.log(current_dist / (target_dist + 1e-8) + 1e-8)  # [num_classes]

            # 应用校正（温和版本）
            calibrated_logits = logits - self.correction_strength * bias.unsqueeze(0)

            calibrated.append(calibrated_logits)

        return calibrated

    def get_consensus_calibration_strength(self):
        """
        如果三尺度都估计出 positive 被高估，返回更强的校正信号

        Returns:
            consensus_strength: [num_classes]，每个类别的共识校正强度
        """
        # 均匀分布作为参考
        uniform = 1.0 / self.num_classes

        # 每个尺度的偏置方向
        bias_signs = torch.sign(self.target_prior - uniform)  # [num_scales, num_classes]

        # 计算共识（三尺度符号一致）
        consensus = (bias_signs.sum(dim=0).abs() == self.num_scales).float()  # [num_classes]

        return consensus

    def forward(self, scale_logits_list, iteration=None, is_source=False, source_labels=None):
        """
        前向传播

        Args:
            scale_logits_list: list of [N, num_classes]
            iteration: 当前迭代步数（用于判断是否 warmup）
            is_source: 是否源域数据
            source_labels: 源域标签（仅 is_source=True 时需要）

        Returns:
            calibrated_logits_list: list of [N, num_classes]
            diagnostics: dict
        """
        # 0-300 步：更新源域混淆矩阵
        if iteration is not None and iteration < 300:
            if is_source and source_labels is not None:
                self.update_source_confusion(scale_logits_list, source_labels)
            return scale_logits_list, {}

        # 第 300 步：完成 warmup
        if iteration == 300 and not self.warmup_done:
            self.finish_warmup()

        # 300 步之后：估计 target prior 并校准
        if not self.warmup_done:
            return scale_logits_list, {}

        # 估计目标域 prior
        scale_probs_list = [F.softmax(logits, dim=1) for logits in scale_logits_list]
        self.estimate_target_prior(scale_probs_list)

        # 校准 logits
        calibrated = self.calibrate_logits(scale_logits_list)

        # 诊断信息
        consensus = self.get_consensus_calibration_strength()
        diagnostics = {
            'cal_target_prior': self.target_prior.cpu().numpy(),
            'cal_consensus_strength': consensus.cpu().numpy(),
        }

        return calibrated, diagnostics


def test_calibration():
    """单元测试"""
    print("=" * 60)
    print("测试 MultiScaleBalancedCalibration")
    print("=" * 60)

    # 创建模块
    calibrator = MultiScaleBalancedCalibration(
        num_classes=3,
        num_scales=3,
        momentum=0.95,
        correction_strength=0.5,
    )

    # 模拟源域 warmup（0-300步）
    print("\n阶段 1: 源域 warmup")
    for iter in range(300):
        # 模拟源域数据
        source_logits_list = [torch.randn(32, 3) for _ in range(3)]
        source_labels = torch.randint(0, 3, (32,))

        _, diag = calibrator(
            source_logits_list,
            iteration=iter,
            is_source=True,
            source_labels=source_labels
        )

    print(f"源域混淆矩阵（尺度 0）:\n{calibrator.source_confusion[0]}")

    # 第 300 步：完成 warmup
    print("\n阶段 2: 完成 warmup")
    calibrator.finish_warmup()
    print(f"Warmup 完成: {calibrator.warmup_done}")

    # 300 步之后：目标域校准
    print("\n阶段 3: 目标域校准")
    target_logits_list = [torch.randn(32, 3) for _ in range(3)]

    # 模拟 positive 被过度预测
    for logits in target_logits_list:
        logits[:, 0] += 0.5  # boost positive

    calibrated, diag = calibrator(
        target_logits_list,
        iteration=301,
        is_source=False
    )

    print(f"\n估计的目标域 prior:")
    print(calibrator.target_prior)
    print(f"\n共识校正强度:")
    print(diag['cal_consensus_strength'])

    # 验证校准效果
    original_dist = F.softmax(target_logits_list[0], dim=1).mean(dim=0)
    calibrated_dist = F.softmax(calibrated[0], dim=1).mean(dim=0)

    print(f"\n校准前分布: {original_dist}")
    print(f"校准后分布: {calibrated_dist}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_calibration()
