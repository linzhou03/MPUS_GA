"""Class-conditional multiscale graph-temporal model for multi-source UDA."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from MPUS_GA.layers import GRL, PositionalEncoding, SpatialStream

from .data import NUM_BANDS, NUM_CHANNELS, NUM_CLASSES, scale_key


class EEGChannelAttention(nn.Module):
    """Squeeze-excitation attention over the ordered EEG channels."""

    def __init__(self, num_channels: int, reduction: int = 4) -> None:
        super().__init__()
        if reduction < 1:
            raise ValueError("channel-attention reduction must be positive")
        hidden = max(num_channels // reduction, 8)
        self.num_channels = num_channels
        self.network = nn.Sequential(
            nn.Linear(2 * num_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != self.num_channels:
            raise ValueError(
                f"channel attention expects [windows,{self.num_channels},bands]"
            )
        descriptor = torch.cat((x.mean(dim=2), x.amax(dim=2)), dim=1)
        weights = 0.5 + self.network(descriptor)
        return x * weights.unsqueeze(-1), weights


class WindowSpatialEncoder(nn.Module):
    """Band-gated dynamic graph encoder shared by every temporal scale."""

    def __init__(
        self,
        d_model: int,
        spatial_layers: int,
        spatial_topk: int,
        dropout: float,
        use_channel_attention: bool = True,
        channel_attention_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.channel_attention = (
            EEGChannelAttention(NUM_CHANNELS, channel_attention_reduction)
            if use_channel_attention
            else None
        )
        self.band_gate = nn.Sequential(
            nn.Linear(NUM_BANDS, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, NUM_BANDS),
            nn.Sigmoid(),
        )
        self.input_projection = nn.Linear(NUM_BANDS, d_model)
        self.spatial_stream = SpatialStream(
            d_model, [d_model] * spatial_layers, spatial_topk
        )
        self.residual_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.electrode_attention = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[2:] != (NUM_CHANNELS, NUM_BANDS):
            raise ValueError(
                f"x must have shape [batch,time,{NUM_CHANNELS},{NUM_BANDS}]"
            )
        if mask.shape != x.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [batch,time]")
        if torch.any(~mask.any(dim=1)):
            raise ValueError("Every trial must contain at least one real window")

        valid = x[mask]
        if self.channel_attention is not None:
            valid, _ = self.channel_attention(valid)
        band_weight = self.band_gate(valid.mean(dim=1))
        base = self.input_projection(valid * (1.0 + band_weight.unsqueeze(1)))
        spatial = self.spatial_stream(base)[-1]
        spatial = self.residual_norm(base + self.dropout(spatial))
        electrode_weight = F.softmax(self.electrode_attention(spatial), dim=1)
        window_embedding = (spatial * electrode_weight).sum(dim=1)

        sequence = window_embedding.new_zeros(
            (x.shape[0], x.shape[1], window_embedding.shape[-1])
        )
        sequence[mask] = window_embedding
        return sequence


class TemporalConvBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=5, padding=2, groups=d_model
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        local = sequence.transpose(1, 2)
        local = self.pointwise(F.gelu(self.depthwise(local))).transpose(1, 2)
        output = self.norm(sequence + self.dropout(local))
        return output.masked_fill(~mask.unsqueeze(-1), 0.0)


class ScaleTemporalEncoder(nn.Module):
    """Local convolution, Transformer, and attentive-statistics pooling."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        temporal_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.local = TemporalConvBlock(d_model, dropout)
        self.position = PositionalEncoding(d_model, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=temporal_layers, enable_nested_tensor=False
        )
        self.window_attention = nn.Linear(d_model, 1)
        self.statistics_projection = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sequence = self.local(sequence, mask)
        sequence = self.position(sequence)
        sequence = self.transformer(sequence, src_key_padding_mask=~mask)
        sequence = sequence.masked_fill(~mask.unsqueeze(-1), 0.0)
        attention_logits = self.window_attention(sequence).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, float("-inf"))
        attention = F.softmax(attention_logits, dim=1)
        attentive = (sequence * attention.unsqueeze(-1)).sum(dim=1)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(sequence.dtype)
        mean = sequence.sum(dim=1) / count
        centered = (sequence - mean.unsqueeze(1)).masked_fill(
            ~mask.unsqueeze(-1), 0.0
        )
        std = torch.sqrt((centered.square().sum(dim=1) / count).clamp_min(1e-6))
        return self.statistics_projection(torch.cat((attentive, mean, std), dim=1))


class DomainDiscriminator(nn.Module):
    def __init__(
        self, input_dim: int, d_model: int, num_domains: int, dropout: float
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2 * d_model),
            nn.LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_domains),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.network(feature)


class MultiScaleMultiSourceDANN(nn.Module):
    """Multiscale encoder with class-dependent fusion and selectable alignment."""

    VALID_FUSIONS = {"class_conditional", "attention", "uniform"}
    VALID_PYRAMID_GATE_MODES = {"global", "class_static", "sample_class"}
    VALID_DOMAIN_MODES = {
        "scale_conditional",
        "scale_global",
        "fused_conditional",
    }

    def __init__(
        self,
        scales: Sequence[float],
        num_domains: int,
        d_model: int = 96,
        num_heads: int = 4,
        spatial_layers: int = 2,
        temporal_layers: int = 2,
        fusion_layers: int = 2,
        dim_feedforward: int = 384,
        dropout: float = 0.3,
        spatial_topk: int = 8,
        num_classes: int = NUM_CLASSES,
        use_channel_attention: bool = True,
        channel_attention_reduction: int = 4,
        fusion_mode: str = "class_conditional",
        domain_mode: str = "scale_conditional",
        detach_domain_probability: bool = True,
        relation_strength: float = 1.0,
        pyramid_weight_floor: float = 0.10,
        pyramid_residual_initial: float = 0.05,
        use_feature_pyramid: bool = True,
        pyramid_gate_mode: str = "global",
        use_pyramid_bias_guard: bool = False,
        pyramid_residual_clip: float = 0.50,
        pyramid_guard_strength: float = 4.0,
        pyramid_guard_tolerance: float = 0.05,
    ) -> None:
        super().__init__()
        self.scales = tuple(sorted(map(float, scales)))
        if not self.scales or len(set(self.scales)) != len(self.scales):
            raise ValueError("Scales must be nonempty and unique")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if num_domains < 2:
            raise ValueError("At least two domains are required")
        if fusion_mode not in self.VALID_FUSIONS:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        if domain_mode not in self.VALID_DOMAIN_MODES:
            raise ValueError(f"Unsupported domain mode: {domain_mode}")
        if relation_strength < 0:
            raise ValueError("relation_strength must be nonnegative")
        if not 0 <= pyramid_weight_floor < 1:
            raise ValueError("pyramid_weight_floor must be within [0,1)")
        if not 0 < pyramid_residual_initial < 1:
            raise ValueError("pyramid_residual_initial must be within (0,1)")
        if pyramid_gate_mode not in self.VALID_PYRAMID_GATE_MODES:
            raise ValueError(
                f"Unsupported pyramid gate mode: {pyramid_gate_mode}"
            )
        if pyramid_residual_clip <= 0:
            raise ValueError("pyramid_residual_clip must be positive")
        if min(pyramid_guard_strength, pyramid_guard_tolerance) < 0:
            raise ValueError("pyramid guard parameters must be nonnegative")
        self.scale_keys = tuple(scale_key(scale) for scale in self.scales)
        self.num_classes = int(num_classes)
        self.num_domains = int(num_domains)
        self.d_model = int(d_model)
        self.fusion_mode = fusion_mode
        self.domain_mode = domain_mode
        self.detach_domain_probability = bool(detach_domain_probability)
        self.relation_strength = float(relation_strength)
        self.pyramid_weight_floor = float(pyramid_weight_floor)
        self.use_feature_pyramid = bool(use_feature_pyramid)
        self.pyramid_gate_mode = pyramid_gate_mode
        self.use_pyramid_bias_guard = bool(use_pyramid_bias_guard)
        self.pyramid_residual_clip = float(pyramid_residual_clip)
        self.pyramid_guard_strength = float(pyramid_guard_strength)
        self.pyramid_guard_tolerance = float(pyramid_guard_tolerance)

        self.spatial = WindowSpatialEncoder(
            d_model,
            spatial_layers,
            spatial_topk,
            dropout,
            use_channel_attention,
            channel_attention_reduction,
        )
        self.temporal = nn.ModuleDict(
            {
                key: ScaleTemporalEncoder(
                    d_model,
                    num_heads,
                    temporal_layers,
                    dim_feedforward,
                    dropout,
                )
                for key in self.scale_keys
            }
        )
        self.scale_embedding = nn.Parameter(torch.empty(len(self.scales), d_model))
        nn.init.trunc_normal_(self.scale_embedding, std=0.02)
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.scale_context = nn.TransformerEncoder(
            fusion_layer, num_layers=fusion_layers, enable_nested_tensor=False
        )
        self.scale_output_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        hidden = max(d_model // 2, num_classes)
        self.class_scale_gate = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )
        self.naive_scale_gate = nn.Linear(d_model, 1)
        # A learnable residual on the external scale x class relation matrix.
        # It starts at zero so the source/prototype relation remains the prior.
        self.scale_class_relation_residual = nn.Parameter(
            torch.zeros(len(self.scales), num_classes)
        )
        self.pyramid_projections = nn.ModuleList()
        self.pyramid_output_norm: nn.Module = nn.Identity()
        self.pyramid_sample_gate: nn.Module | None = None
        if self.use_feature_pyramid:
            self.pyramid_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.Linear(d_model, d_model),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_model, d_model),
                    )
                    for _ in self.scales
                ]
            )
            for projection in self.pyramid_projections:
                nn.init.zeros_(projection[-1].weight)
                nn.init.zeros_(projection[-1].bias)
            self.pyramid_output_norm = nn.LayerNorm(d_model)
            residual_logit = math.log(
                pyramid_residual_initial / (1.0 - pyramid_residual_initial)
            )
            self.pyramid_residual_logit = nn.Parameter(
                torch.tensor(residual_logit)
            )
            self.pyramid_class_gate_logit = nn.Parameter(
                torch.full((num_classes,), residual_logit)
            )
            if self.pyramid_gate_mode == "sample_class":
                gate_hidden = max(8, num_classes * 2)
                self.pyramid_sample_gate = nn.Sequential(
                    nn.Linear(6, gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, 1),
                )
                # The sample gate begins as the class-static prior. This makes
                # every new G variant an exact near-B4 fallback at startup.
                nn.init.zeros_(self.pyramid_sample_gate[-1].weight)
                nn.init.zeros_(self.pyramid_sample_gate[-1].bias)
        else:
            self.register_buffer(
                "pyramid_residual_logit", torch.tensor(float("-inf"))
            )
            self.register_buffer(
                "pyramid_class_gate_logit",
                torch.full((num_classes,), float("-inf")),
            )

        self.grl = GRL(alpha=1.0)
        self.scale_domain_heads = nn.ModuleList()
        if domain_mode in {"scale_conditional", "scale_global"}:
            input_dim = (
                d_model * num_classes
                if domain_mode == "scale_conditional"
                else d_model
            )
            self.scale_domain_heads = nn.ModuleList(
                [
                    DomainDiscriminator(input_dim, d_model, num_domains, dropout)
                    for _ in self.scales
                ]
            )
        self.fused_domain_head = None
        if domain_mode == "fused_conditional":
            self.fused_domain_head = DomainDiscriminator(
                d_model * num_classes, d_model, num_domains, dropout
            )

    def _fuse(
        self,
        scale_embeddings: torch.Tensor,
        scale_logits: torch.Tensor,
        scale_class_reliability: torch.Tensor | None,
        pyramid_gate_ramp: float = 1.0,
        pyramid_bias_risk: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, scale_count, _ = scale_logits.shape
        if not 0 <= pyramid_gate_ramp <= 1:
            raise ValueError("pyramid_gate_ramp must be within [0,1]")
        if scale_count == 1:
            class_weight = scale_logits.new_full(
                (batch, scale_count, self.num_classes), 1.0 / scale_count
            )
            class_features = scale_embeddings[:, :1].expand(
                -1, self.num_classes, -1
            )
            logits = scale_logits[:, 0]
            zero_gate = logits.new_zeros((batch, self.num_classes))
            return {
                "logits": logits,
                "embedding": scale_embeddings[:, 0],
                "scale_class_weight": class_weight,
                "pyramid_scale_class_weight": class_weight,
                "pyramid_class_features": class_features,
                "pyramid_logits": logits,
                "pyramid_gate": zero_gate,
                "pyramid_raw_gate": zero_gate,
                "pyramid_guard_factor": torch.ones_like(zero_gate),
                "pyramid_unsupported_excess": zero_gate,
                "pyramid_logit_delta": zero_gate,
            }
        if self.fusion_mode == "uniform":
            class_weight = scale_logits.new_full(
                (batch, scale_count, self.num_classes), 1.0 / scale_count
            )
        elif self.fusion_mode == "attention":
            scale_weight = F.softmax(
                self.naive_scale_gate(scale_embeddings).squeeze(-1), dim=1
            )
            class_weight = scale_weight.unsqueeze(-1).expand(
                -1, -1, self.num_classes
            )
        else:
            gate_logits = (
                self.class_scale_gate(scale_embeddings)
                + self.scale_class_relation_residual.unsqueeze(0)
            )
            if scale_class_reliability is not None:
                reliability = scale_class_reliability.to(gate_logits)
                if reliability.shape != (scale_count, self.num_classes):
                    raise ValueError(
                        "scale_class_reliability must be [scales, classes]"
                    )
                gate_logits = gate_logits + self.relation_strength * torch.log(
                    reliability.clamp_min(1e-6)
                )
            class_weight = F.softmax(gate_logits, dim=1)
        pyramid_class_weight = class_weight
        if self.fusion_mode != "uniform":
            uniform = torch.full_like(class_weight, 1.0 / scale_count)
            pyramid_class_weight = (
                (1.0 - self.pyramid_weight_floor) * class_weight
                + self.pyramid_weight_floor * uniform
            )

        relation_weighted_logits = (class_weight * scale_logits).sum(dim=1)
        if not self.use_feature_pyramid:
            class_features = torch.einsum(
                "bsc,bsd->bcd", class_weight, scale_embeddings
            )
            class_mass = F.softmax(relation_weighted_logits.detach(), dim=-1)
            fused_embedding = torch.einsum(
                "bc,bcd->bd", class_mass, class_features
            )
            zero_gate = relation_weighted_logits.new_zeros(
                (batch, self.num_classes)
            )
            return {
                "logits": relation_weighted_logits,
                "embedding": fused_embedding,
                "scale_class_weight": class_weight,
                "pyramid_scale_class_weight": class_weight,
                "pyramid_class_features": class_features,
                "pyramid_logits": relation_weighted_logits,
                "pyramid_gate": zero_gate,
                "pyramid_raw_gate": zero_gate,
                "pyramid_guard_factor": torch.ones_like(zero_gate),
                "pyramid_unsupported_excess": zero_gate,
                "pyramid_logit_delta": zero_gate,
            }

        projected_levels = torch.stack(
            [
                scale_embeddings[:, index] + projection(
                    scale_embeddings[:, index]
                )
                for index, projection in enumerate(self.pyramid_projections)
            ],
            dim=1,
        )
        class_features = self.pyramid_output_norm(
            torch.einsum(
                "bsc,bsd->bcd", pyramid_class_weight, projected_levels
            )
        )
        pyramid_logits = torch.einsum(
            "bcd,cd->bc", class_features, self.classifier.weight
        )
        if self.classifier.bias is not None:
            pyramid_logits = pyramid_logits + self.classifier.bias

        if self.pyramid_gate_mode == "global":
            raw_gate = torch.sigmoid(self.pyramid_residual_logit).expand(
                batch, self.num_classes
            )
        else:
            raw_gate_logit = self.pyramid_class_gate_logit.unsqueeze(0).expand(
                batch, -1
            )
            if self.pyramid_gate_mode == "sample_class":
                if self.pyramid_sample_gate is None:
                    raise RuntimeError("sample-class pyramid gate is unavailable")
                # Gate evidence comes only from raw independent-scale logits
                # and the downstream relation weights. Detaching it prevents
                # gate gradients from corrupting the independent scale heads.
                detached_probability = F.softmax(scale_logits.detach(), dim=-1)
                mean_probability = detached_probability.mean(dim=1)
                probability_spread = detached_probability.std(
                    dim=1, unbiased=False
                )
                anchor_probability = F.softmax(
                    relation_weighted_logits.detach(), dim=-1
                )
                relation_concentration = class_weight.detach().square().sum(dim=1)
                mean_distribution = mean_probability.unsqueeze(1).clamp_min(1e-8)
                js_divergence = (
                    detached_probability
                    * (
                        torch.log(detached_probability.clamp_min(1e-8))
                        - torch.log(mean_distribution)
                    )
                ).sum(dim=-1).mean(dim=1, keepdim=True)
                logit_delta = torch.tanh(
                    (pyramid_logits - relation_weighted_logits).detach()
                )
                gate_evidence = torch.stack(
                    (
                        mean_probability,
                        probability_spread,
                        anchor_probability,
                        relation_concentration,
                        (1.0 - js_divergence).expand(-1, self.num_classes),
                        logit_delta,
                    ),
                    dim=-1,
                )
                raw_gate_logit = raw_gate_logit + self.pyramid_sample_gate(
                    gate_evidence
                ).squeeze(-1)
            raw_gate = torch.sigmoid(raw_gate_logit)

        guard_factor = torch.ones_like(raw_gate)
        unsupported_excess = torch.zeros_like(raw_gate)
        if self.use_pyramid_bias_guard:
            independent_probability = F.softmax(
                scale_logits.detach(), dim=-1
            ).mean(dim=1)
            pyramid_probability = F.softmax(
                pyramid_logits.detach(), dim=-1
            )
            unsupported_excess = (
                pyramid_probability
                - independent_probability
                - self.pyramid_guard_tolerance
            ).clamp_min(0.0)
            total_risk = unsupported_excess
            if pyramid_bias_risk is not None:
                bias_risk = pyramid_bias_risk.detach().to(raw_gate)
                if bias_risk.shape != (self.num_classes,):
                    raise ValueError("pyramid_bias_risk must be [classes]")
                total_risk = total_risk + bias_risk.clamp_min(0.0).unsqueeze(0)
            guard_factor = torch.exp(
                -self.pyramid_guard_strength * total_risk
            ).clamp(0.0, 1.0)

        effective_gate = raw_gate * float(pyramid_gate_ramp) * guard_factor
        logit_delta = pyramid_logits - relation_weighted_logits
        if self.pyramid_gate_mode != "global":
            logit_delta = logit_delta.clamp(
                -self.pyramid_residual_clip,
                self.pyramid_residual_clip,
            )
        # The relation-weighted logits are always the stable anchor. G gates
        # can accept or reject a bounded pyramid correction per sample/class.
        fused_logits = relation_weighted_logits + effective_gate * logit_delta
        class_mass = F.softmax(fused_logits.detach(), dim=-1)
        fused_embedding = torch.einsum(
            "bc,bcd->bd", class_mass, class_features
        )
        return {
            "logits": fused_logits,
            "embedding": fused_embedding,
            "scale_class_weight": class_weight,
            "pyramid_scale_class_weight": pyramid_class_weight,
            "pyramid_class_features": class_features,
            "pyramid_logits": pyramid_logits,
            "pyramid_gate": effective_gate,
            "pyramid_raw_gate": raw_gate,
            "pyramid_guard_factor": guard_factor,
            "pyramid_unsupported_excess": unsupported_excess,
            "pyramid_logit_delta": logit_delta,
        }

    @staticmethod
    def _conditional_feature(
        embedding: torch.Tensor,
        logits: torch.Tensor,
        detach_probability: bool,
    ) -> torch.Tensor:
        probability = F.softmax(logits, dim=-1)
        if detach_probability:
            probability = probability.detach()
        return torch.einsum("bc,bd->bcd", probability, embedding).flatten(1)

    def forward(
        self,
        x_by_scale: Mapping[str, torch.Tensor],
        mask_by_scale: Mapping[str, torch.Tensor],
        grl_alpha: float = 1.0,
        compute_domain: bool = True,
        scale_class_reliability: torch.Tensor | None = None,
        class_logit_adjustment: torch.Tensor | None = None,
        pyramid_gate_ramp: float = 1.0,
        pyramid_bias_risk: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if tuple(x_by_scale) != self.scale_keys:
            raise ValueError(
                f"Expected scales {self.scale_keys}, got {tuple(x_by_scale)}"
            )
        if tuple(mask_by_scale) != self.scale_keys:
            raise ValueError("Feature and mask scales do not match")

        pooled = []
        for key in self.scale_keys:
            sequence = self.spatial(x_by_scale[key], mask_by_scale[key])
            pooled.append(self.temporal[key](sequence, mask_by_scale[key]))
        tokens = torch.stack(pooled, dim=1) + self.scale_embedding.unsqueeze(0)

        # Hard independence boundary: every per-scale classifier consumes only
        # its own scale token. Cross-scale context and the scale-class relation
        # graph are downstream fusion mechanisms and can never alter these
        # embeddings or logits in the forward pass.
        scale_embeddings = self.scale_output_norm(tokens)
        scale_logits = self.classifier(scale_embeddings)
        fusion_scale_logits = scale_logits
        if class_logit_adjustment is not None:
            adjustment = class_logit_adjustment.to(scale_logits)
            if adjustment.shape != (self.num_classes,):
                raise ValueError("class_logit_adjustment must be [classes]")
            fusion_scale_logits = scale_logits + adjustment.view(1, 1, -1)
        contextual_scale_embeddings = self.output_norm(self.scale_context(tokens))
        fusion = self._fuse(
            contextual_scale_embeddings,
            scale_logits,
            scale_class_reliability,
            pyramid_gate_ramp=pyramid_gate_ramp,
            pyramid_bias_risk=pyramid_bias_risk,
        )
        logits = fusion["logits"]
        embedding = fusion["embedding"]
        scale_class_weight = fusion["scale_class_weight"]
        pyramid_scale_class_weight = fusion["pyramid_scale_class_weight"]
        pyramid_class_features = fusion["pyramid_class_features"]
        if class_logit_adjustment is not None:
            logits = logits + adjustment.view(1, -1)
            embedding = torch.einsum(
                "bc,bcd->bd",
                F.softmax(logits.detach(), dim=-1),
                pyramid_class_features,
            )
        probability = F.softmax(logits, dim=-1)

        self.grl.alpha = float(grl_alpha)
        scale_domain_logits = None
        fused_domain_logits = None
        if compute_domain and self.domain_mode in {
            "scale_conditional",
            "scale_global",
        }:
            domain_outputs = []
            for index, head in enumerate(self.scale_domain_heads):
                feature = scale_embeddings[:, index]
                if self.domain_mode == "scale_conditional":
                    feature = self._conditional_feature(
                        feature,
                        fusion_scale_logits[:, index],
                        self.detach_domain_probability,
                    )
                domain_outputs.append(head(self.grl(feature)))
            scale_domain_logits = torch.stack(domain_outputs, dim=1)
        elif compute_domain and self.fused_domain_head is not None:
            conditional = self._conditional_feature(
                embedding, logits, self.detach_domain_probability
            )
            fused_domain_logits = self.fused_domain_head(self.grl(conditional))

        scale_weight = scale_class_weight.mean(dim=-1)
        pyramid_scale_weight = pyramid_scale_class_weight.mean(dim=-1)
        return {
            "logits": logits,
            "probability": probability,
            "embedding": embedding,
            "scale_embeddings": scale_embeddings,
            "contextual_scale_embeddings": contextual_scale_embeddings,
            "scale_logits": scale_logits,
            "calibrated_scale_logits": fusion_scale_logits,
            "scale_class_weight": scale_class_weight,
            "scale_weight": scale_weight,
            "pyramid_scale_class_weight": pyramid_scale_class_weight,
            "pyramid_scale_weight": pyramid_scale_weight,
            "pyramid_class_features": pyramid_class_features,
            "relation_weighted_logits": (
                scale_class_weight * scale_logits
            ).sum(dim=1),
            "pyramid_logits": fusion["pyramid_logits"],
            "pyramid_residual_weight": fusion["pyramid_gate"].mean(),
            "pyramid_residual_gate": fusion["pyramid_gate"],
            "pyramid_raw_gate": fusion["pyramid_raw_gate"],
            "pyramid_guard_factor": fusion["pyramid_guard_factor"],
            "pyramid_unsupported_excess": fusion[
                "pyramid_unsupported_excess"
            ],
            "pyramid_logit_delta": fusion["pyramid_logit_delta"],
            "scale_class_relation_residual": self.scale_class_relation_residual,
            "scale_domain_logits": scale_domain_logits,
            "fused_domain_logits": fused_domain_logits,
        }
