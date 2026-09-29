"""
Confusion-Aware CBST 伪标签选择模块

在原有 CBST（类别平衡自训练）基础上加入混淆风险评估

设计思路：
1. 不仅看置信度，还看混淆风险（基于三尺度分歧）
2. 对 positive 类候选施加更严格的一致性要求
3. 对 neg/neu 类候选，如果某个尺度给了 positive 高概率，判定为高风险

Created: 2026-10-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfusionAwareCBST(nn.Module):
    """
    Confusion-Aware 类别平衡自训练
    """

    def __init__(
        self,
        num_classes=3,
        positive_consistency_threshold=0.15,
        positive_risk_threshold=0.3,
    ):
        """
        Args:
            num_classes: 类别数
            positive_consistency_threshold: positive 类的一致性要求
            positive_risk_threshold: 非 positive 类的风险阈值
        """
        super().__init__()
        self.num_classes = num_classes
        self.positive_consistency_threshold = positive_consistency_threshold
        self.positive_risk_threshold = positive_risk_threshold

    def select_pseudo_labels(
        self,
        fused_probs,      # [N, num_classes] 融合后的概率
        scale_probs_list, # list of [N, num_classes]，三尺度独立概率
        quota_per_class,  # 每个类别选择的样本数
    ):
        """
        选择伪标签样本

        Args:
            fused_probs: [N, num_classes] 融合预测概率
            scale_probs_list: list of [N, num_classes]，三尺度概率
            quota_per_class: int，每类选择样本数

        Returns:
            selected_indices: [K] 被选中的样本索引
            selected_labels: [K] 对应的伪标签
            diagnostics: dict
        """
        N = fused_probs.size(0)
        device = fused_probs.device

        # 融合预测类别
        fused_preds = fused_probs.argmax(dim=1)  # [N]

        # 堆叠三尺度概率 [N, 3, num_classes]
        scale_probs = torch.stack(scale_probs_list, dim=1)

        selected_indices = []
        selected_labels = []
        diagnostics = {
            'cbst_risk_scores': {c: [] for c in range(self.num_classes)},
            'cbst_selected_counts': {c: 0 for c in range(self.num_classes)},
        }

        for c in range(self.num_classes):
            # 当前类别的候选
            candidates = torch.where(fused_preds == c)[0]
            if len(candidates) == 0:
                continue

            # 计算每个候选的混淆风险
            risk_scores = self.compute_confusion_risk(
                candidates, c, scale_probs, fused_probs
            )

            # 选择风险最低的 K 个
            k = min(quota_per_class, len(candidates))
            if k > 0:
                _, low_risk_idx = risk_scores.topk(k, largest=False)
                selected = candidates[low_risk_idx]
                selected_indices.append(selected)
                selected_labels.append(torch.full((k,), c, device=device))

                # 记录诊断信息
                diagnostics['cbst_risk_scores'][c] = risk_scores.cpu().numpy()
                diagnostics['cbst_selected_counts'][c] = k

        # 合并所有类别
        if len(selected_indices) > 0:
            selected_indices = torch.cat(selected_indices)
            selected_labels = torch.cat(selected_labels)
        else:
            selected_indices = torch.tensor([], dtype=torch.long, device=device)
            selected_labels = torch.tensor([], dtype=torch.long, device=device)

        return selected_indices, selected_labels, diagnostics

    def compute_confusion_risk(self, indices, class_c, scale_probs, fused_probs):
        """
        计算候选样本的混淆风险

        Args:
            indices: [M] 候选样本索引
            class_c: 目标类别
            scale_probs: [N, 3, num_classes] 三尺度概率
            fused_probs: [N, num_classes] 融合概率

        Returns:
            risk_scores: [M] 风险分数（越高越危险）
        """
        M = len(indices)
        risk_scores = torch.zeros(M, device=indices.device)

        for i, idx in enumerate(indices):
            if class_c == 0:  # Positive 类
                risk_scores[i] = self._compute_positive_risk(
                    idx, scale_probs, fused_probs
                )
            else:  # Neutral (1) 或 Negative (2) 类
                risk_scores[i] = self._compute_non_positive_risk(
                    idx, class_c, scale_probs, fused_probs
                )

        return risk_scores

    def _compute_positive_risk(self, idx, scale_probs, fused_probs):
        """
        计算 positive 候选的风险：三尺度一致性

        风险定义：三尺度在 positive 类上的标准差
        """
        pos_probs = scale_probs[idx, :, 0]  # [3] 三尺度的 positive 概率
        disagreement = pos_probs.std()

        # 归一化到 [0, 1]
        risk = disagreement / (self.positive_consistency_threshold + 1e-8)
        risk = torch.clamp(risk, 0.0, 1.0)

        return risk

    def _compute_non_positive_risk(self, idx, class_c, scale_probs, fused_probs):
        """
        计算 non-positive 候选的风险：是否有尺度给 positive 高概率

        风险定义：三尺度中给 positive 的最高概率
        """
        pos_probs = scale_probs[idx, :, 0]  # [3] 三尺度的 positive 概率
        max_pos_prob = pos_probs.max()

        # 如果某个尺度给了 positive > threshold，判定为高风险
        if max_pos_prob > self.positive_risk_threshold:
            risk = max_pos_prob
        else:
            risk = torch.tensor(0.0, device=max_pos_prob.device)

        return risk

    def forward(
        self,
        fused_probs,
        scale_probs_list,
        quota_per_class,
    ):
        """
        前向传播（等价于 select_pseudo_labels）
        """
        return self.select_pseudo_labels(
            fused_probs, scale_probs_list, quota_per_class
        )


def test_confusion_aware_cbst():
    """单元测试"""
    print("=" * 60)
    print("测试 ConfusionAwareCBST")
    print("=" * 60)

    # 模拟数据
    N = 100
    num_classes = 3

    # 创建模块
    cbst = ConfusionAwareCBST(
        num_classes=num_classes,
        positive_consistency_threshold=0.15,
        positive_risk_threshold=0.3,
    )

    # 模拟融合概率（positive 被过度预测）
    fused_logits = torch.randn(N, num_classes)
    fused_logits[:, 0] += 0.5  # boost positive
    fused_probs = F.softmax(fused_logits, dim=1)

    # 模拟三尺度概率
    scale_probs_list = []
    for s in range(3):
        scale_logits = torch.randn(N, num_classes)
        scale_logits[:, 0] += torch.randn(N) * 0.3  # positive 有分歧
        scale_probs_list.append(F.softmax(scale_logits, dim=1))

    # 选择伪标签（每类 10 个）
    quota_per_class = 10
    selected_indices, selected_labels, diagnostics = cbst.select_pseudo_labels(
        fused_probs, scale_probs_list, quota_per_class
    )

    print(f"\n选中样本数: {len(selected_indices)}")
    print(f"各类选中数量:")
    for c in range(num_classes):
        count = diagnostics['cbst_selected_counts'][c]
        print(f"  类别 {c}: {count}")

    # 验证：选中的 positive 样本应该有低分歧
    pos_selected = selected_indices[selected_labels == 0]
    if len(pos_selected) > 0:
        pos_scale_probs = torch.stack(scale_probs_list, dim=1)[pos_selected]
        pos_disagreement = pos_scale_probs[:, :, 0].std(dim=1).mean()
        print(f"\n选中的 positive 样本的平均分歧: {pos_disagreement:.4f}")

    # 对比：未选中的 positive 候选
    fused_preds = fused_probs.argmax(dim=1)
    pos_candidates = torch.where(fused_preds == 0)[0]
    unselected = torch.tensor([idx for idx in pos_candidates if idx not in pos_selected])
    if len(unselected) > 0:
        unsel_scale_probs = torch.stack(scale_probs_list, dim=1)[unselected]
        unsel_disagreement = unsel_scale_probs[:, :, 0].std(dim=1).mean()
        print(f"未选中的 positive 候选的平均分歧: {unsel_disagreement:.4f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_confusion_aware_cbst()
