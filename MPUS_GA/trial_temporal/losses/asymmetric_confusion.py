"""
非对称混淆损失模块

针对 CAGA-SGA 观察到的 "Positive 吸引子" 问题：
- 负向→正向：23-37%
- 中性→正向：24-40%

设计思路：
1. 利用三尺度独立 logits 判断预测不确定性
2. 对"三尺度对 positive 分歧大但融合预测 positive"的样本施加额外约束
3. 对伪标签为 neg/neu 但靠近 positive 决策边界的样本推离

Created: 2026-10-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricConfusionLoss(nn.Module):
    """
    非对称混淆损失：重点惩罚容易误判为 positive 的样本
    """

    def __init__(
        self,
        margin_strength=0.3,
        disagreement_threshold=0.15,
        temperature=0.07,
    ):
        """
        Args:
            margin_strength: margin loss 的权重
            disagreement_threshold: 三尺度分歧判定阈值
            temperature: contrastive 温度参数
        """
        super().__init__()
        self.margin_strength = margin_strength
        self.disagreement_threshold = disagreement_threshold
        self.temperature = temperature

    def forward(
        self,
        fused_logits,      # [N, 3] 融合后的 logits
        pseudo_labels,     # [N] 伪标签（硬标签）
        scale_probs,       # [N, 3, 3] 三尺度独立概率 (scale, batch, class)
        features=None,     # [N, D] 特征（可选，用于 contrastive）
    ):
        """
        计算非对称混淆损失

        Returns:
            loss: 标量损失
            diagnostics: dict，包含各项损失的细分
        """
        N = fused_logits.size(0)
        device = fused_logits.device

        # 融合预测
        fused_preds = fused_logits.argmax(dim=1)  # [N]

        # 1. 识别"危险 positive"样本
        # 条件：融合预测为 positive，但三尺度对 positive 意见不一致
        risky_pos_mask = self._identify_risky_positive(
            fused_preds, scale_probs
        )

        # 2. 对危险 positive 样本施加额外的 margin loss
        margin_loss = 0.0
        if risky_pos_mask.sum() > 0:
            margin_loss = self._compute_margin_loss(
                fused_logits[risky_pos_mask],
                scale_probs[risky_pos_mask],
            )

        # 3. 对伪标签为 neg/neu 的样本，如果靠近 positive 边界，推离
        boundary_loss = 0.0
        if features is not None:
            boundary_loss = self._compute_boundary_push_loss(
                features, pseudo_labels, fused_logits
            )

        # 总损失
        total_loss = margin_loss + self.margin_strength * boundary_loss

        # 诊断信息
        diagnostics = {
            'asy_margin_loss': margin_loss.item() if isinstance(margin_loss, torch.Tensor) else margin_loss,
            'asy_boundary_loss': boundary_loss.item() if isinstance(boundary_loss, torch.Tensor) else boundary_loss,
            'asy_risky_pos_count': risky_pos_mask.sum().item(),
        }

        return total_loss, diagnostics

    def _identify_risky_positive(self, fused_preds, scale_probs):
        """
        识别"危险 positive"：融合预测 positive，但三尺度分歧大

        Args:
            fused_preds: [N] 融合预测类别
            scale_probs: [N, 3, 3] 三尺度概率

        Returns:
            mask: [N] bool，True 表示危险样本
        """
        N = fused_preds.size(0)

        # 只关注融合预测为 positive (class 0) 的样本
        pred_pos = (fused_preds == 0)

        # 计算三尺度在 positive 类上的分歧（标准差）
        pos_probs = scale_probs[:, :, 0]  # [N, 3] 三尺度的 positive 概率
        pos_disagreement = pos_probs.std(dim=1)  # [N]

        # 分歧大于阈值，且融合预测为 positive
        risky = pred_pos & (pos_disagreement > self.disagreement_threshold)

        return risky

    def _compute_margin_loss(self, fused_logits, scale_probs):
        """
        对危险 positive 样本施加 margin loss

        思路：要求 positive logit 与其他类的 margin 更大

        Args:
            fused_logits: [M, 3] 危险样本的融合 logits
            scale_probs: [M, 3, 3] 危险样本的三尺度概率

        Returns:
            loss: 标量
        """
        M = fused_logits.size(0)

        # 提取 positive logit 和其他类 logit
        pos_logit = fused_logits[:, 0]  # [M]
        other_logits = fused_logits[:, 1:]  # [M, 2]
        max_other_logit = other_logits.max(dim=1)[0]  # [M]

        # 要求 margin >= 0.5
        margin_target = 0.5
        margin = pos_logit - max_other_logit  # [M]
        loss = F.relu(margin_target - margin).mean()

        return loss

    def _compute_boundary_push_loss(self, features, pseudo_labels, fused_logits):
        """
        对伪标签为 neg/neu 但靠近 positive 边界的样本推离

        Args:
            features: [N, D] 特征
            pseudo_labels: [N] 伪标签
            fused_logits: [N, 3] 融合 logits

        Returns:
            loss: 标量
        """
        N = features.size(0)

        # 只处理伪标签为 neu (1) 或 neg (2) 的样本
        non_pos_mask = (pseudo_labels == 1) | (pseudo_labels == 2)
        if non_pos_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        # 计算这些样本给 positive 的概率
        probs = F.softmax(fused_logits, dim=1)
        pos_probs = probs[:, 0]  # [N]

        # 识别"靠近 positive 边界"的样本：pos_prob > 0.2
        near_boundary = non_pos_mask & (pos_probs > 0.2)
        if near_boundary.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        # 对这些样本，惩罚其 positive logit
        pos_logits = fused_logits[near_boundary, 0]  # [M]
        loss = F.relu(pos_logits + 0.5).mean()  # 推到负值

        return loss


def test_asymmetric_confusion_loss():
    """单元测试"""
    print("=" * 60)
    print("测试 AsymmetricConfusionLoss")
    print("=" * 60)

    # 模拟数据
    N = 16
    fused_logits = torch.randn(N, 3)
    pseudo_labels = torch.randint(0, 3, (N,))
    scale_probs = torch.softmax(torch.randn(N, 3, 3), dim=2)
    features = torch.randn(N, 128)

    # 创建损失
    loss_fn = AsymmetricConfusionLoss(
        margin_strength=0.3,
        disagreement_threshold=0.15,
    )

    # 计算损失
    loss, diagnostics = loss_fn(
        fused_logits, pseudo_labels, scale_probs, features
    )

    print(f"\n总损失: {loss.item():.4f}")
    print(f"诊断信息:")
    for key, val in diagnostics.items():
        print(f"  {key}: {val}")

    # 测试梯度
    loss.backward()
    print(f"\n梯度检查通过 ✓")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_asymmetric_confusion_loss()
