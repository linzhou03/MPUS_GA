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

    def encode_sequence(
        self, sequence: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        sequence = self.local(sequence, mask)
        sequence = self.position(sequence)
        sequence = self.transformer(sequence, src_key_padding_mask=~mask)
        return sequence.masked_fill(~mask.unsqueeze(-1), 0.0)

    def pool_sequence(
        self, sequence: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
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

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(sequence, mask)
        return self.pool_sequence(encoded, mask)


class TemporalMSAD1D(nn.Module):
    """One-dimensional RDANet-style anti-aliased temporal downsampling."""

    def __init__(self, d_model: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(d_model // reduction, 8)
        self.d_model = int(d_model)
        for size, values in (
            (3, (1.0, 2.0, 1.0)),
            (5, (1.0, 4.0, 6.0, 4.0, 1.0)),
            (7, (1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0)),
        ):
            kernel = torch.tensor(values, dtype=torch.float32)
            kernel = kernel / kernel.sum()
            self.register_buffer(
                f"kernel_{size}", kernel.view(1, 1, size)
            )
        self.filter_gate = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3 * d_model),
        )
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model,
            bias=False,
        )
        self.pointwise = nn.Linear(d_model, d_model, bias=False)
        self.output_norm = nn.LayerNorm(d_model)
        self.channel_gate = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Sigmoid(),
        )
        self.pool_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(sequence.dtype)
        return sequence.sum(dim=1) / count

    def forward(
        self, sequence: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3 or sequence.shape[-1] != self.d_model:
            raise ValueError("MSAD sequence must be [batch,time,d_model]")
        if mask.shape != sequence.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("MSAD mask must be boolean [batch,time]")
        sequence = sequence.masked_fill(~mask.unsqueeze(-1), 0.0)
        descriptor = self._masked_mean(sequence, mask)
        filter_weight = self.filter_gate(descriptor).view(
            sequence.shape[0], 3, self.d_model
        )
        filter_weight = F.softmax(filter_weight, dim=1)
        channel_first = sequence.transpose(1, 2)
        filtered = []
        for size in (3, 5, 7):
            kernel = getattr(self, f"kernel_{size}").to(channel_first)
            weight = kernel.expand(self.d_model, -1, -1)
            filtered.append(
                F.conv1d(
                    channel_first,
                    weight,
                    padding=size // 2,
                    groups=self.d_model,
                ).transpose(1, 2)
            )
        anti_aliased = sum(
            filter_weight[:, index].unsqueeze(1) * value
            for index, value in enumerate(filtered)
        )
        if anti_aliased.shape[1] % 2:
            anti_aliased = F.pad(anti_aliased, (0, 0, 0, 1))
            sequence = F.pad(sequence, (0, 0, 0, 1))
            mask = F.pad(mask, (0, 1), value=False)
        pair_mask = mask.view(mask.shape[0], -1, 2)
        down_mask = pair_mask.any(dim=-1)
        pair_weight = pair_mask.to(sequence.dtype).unsqueeze(-1)
        denominator = pair_weight.sum(dim=2).clamp_min(1.0)
        folded = (
            anti_aliased.view(
                anti_aliased.shape[0], -1, 2, self.d_model
            )
            * pair_weight
        ).sum(dim=2) / denominator
        residual = (
            sequence.view(sequence.shape[0], -1, 2, self.d_model)
            * pair_weight
        ).sum(dim=2) / denominator
        refined = self.depthwise(folded.transpose(1, 2)).transpose(1, 2)
        refined = self.pointwise(refined)
        output = F.gelu(self.output_norm(refined + residual))
        output = output.masked_fill(~down_mask.unsqueeze(-1), 0.0)
        channel_weight = self.channel_gate(
            self._masked_mean(output, down_mask)
        )
        output = output * channel_weight.unsqueeze(1)
        output = output.masked_fill(~down_mask.unsqueeze(-1), 0.0)
        return output, down_mask, filter_weight

    def pool(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mean = self._masked_mean(sequence, mask)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(sequence.dtype)
        centered = (sequence - mean.unsqueeze(1)).masked_fill(
            ~mask.unsqueeze(-1), 0.0
        )
        std = torch.sqrt(
            (centered.square().sum(dim=1) / count).clamp_min(1e-6)
        )
        return self.pool_projection(torch.cat((mean, std), dim=-1))


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
    VALID_MULTIVIEW_FUSIONS = {
        "none",
        "class_query",
        "class_query_low_rank",
    }
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
        multiview_fusion_mode: str = "none",
        use_multiview_uncertainty: bool = False,
        use_sign_aware_pyramid_guard: bool = False,
        multiview_low_rank: int = 16,
        multiview_uncertainty_strength: float = 1.0,
        multiview_conflict_strength: float = 1.0,
        multiview_source_anchor_mix: float = 0.0,
        pyramid_gate_shrinkage: float = 0.0,
        pyramid_gate_ceiling: float = 1.0,
        use_temporal_msad: bool = False,
        use_source_prototype_memory: bool = False,
        use_relative_degradation_fusion: bool = False,
        rda_memory_topk: int = 2,
        rda_memory_temperature: float = 0.25,
        rda_memory_strength: float = 0.25,
        rda_degradation_strength: float = 1.0,
        rda_msad_strength: float = 0.25,
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
        if multiview_fusion_mode not in self.VALID_MULTIVIEW_FUSIONS:
            raise ValueError(
                f"Unsupported multiview fusion mode: {multiview_fusion_mode}"
            )
        if pyramid_residual_clip <= 0:
            raise ValueError("pyramid_residual_clip must be positive")
        if min(pyramid_guard_strength, pyramid_guard_tolerance) < 0:
            raise ValueError("pyramid guard parameters must be nonnegative")
        if multiview_low_rank < 1:
            raise ValueError("multiview_low_rank must be positive")
        if min(
            multiview_uncertainty_strength,
            multiview_conflict_strength,
        ) < 0:
            raise ValueError("multiview reliability strengths must be nonnegative")
        if not 0 <= multiview_source_anchor_mix <= 1:
            raise ValueError("multiview_source_anchor_mix must be within [0,1]")
        if not 0 <= pyramid_gate_shrinkage <= 1:
            raise ValueError("pyramid_gate_shrinkage must be within [0,1]")
        if not 0 < pyramid_gate_ceiling <= 1:
            raise ValueError("pyramid_gate_ceiling must be within (0,1]")
        if rda_memory_topk < 1:
            raise ValueError("rda_memory_topk must be positive")
        if rda_memory_temperature <= 0:
            raise ValueError("rda_memory_temperature must be positive")
        if min(
            rda_memory_strength,
            rda_degradation_strength,
            rda_msad_strength,
        ) < 0:
            raise ValueError("RDA strengths must be nonnegative")
        if multiview_source_anchor_mix > 0 and multiview_fusion_mode == "none":
            raise ValueError("a multiview source anchor requires multiview fusion")
        if multiview_fusion_mode != "none" and not use_feature_pyramid:
            raise ValueError("multiview fusion requires the feature pyramid")
        if use_relative_degradation_fusion and not use_source_prototype_memory:
            raise ValueError(
                "relative degradation fusion requires source prototype memory"
            )
        if (
            use_temporal_msad or use_source_prototype_memory
        ) and len(self.scales) != 3:
            raise ValueError("RDA modules require exactly three temporal scales")
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
        self.multiview_fusion_mode = multiview_fusion_mode
        self.use_multiview_uncertainty = bool(use_multiview_uncertainty)
        self.use_sign_aware_pyramid_guard = bool(
            use_sign_aware_pyramid_guard
        )
        self.multiview_low_rank = int(multiview_low_rank)
        self.multiview_uncertainty_strength = float(
            multiview_uncertainty_strength
        )
        self.multiview_conflict_strength = float(
            multiview_conflict_strength
        )
        self.multiview_source_anchor_mix = float(
            multiview_source_anchor_mix
        )
        self.pyramid_gate_shrinkage = float(pyramid_gate_shrinkage)
        self.pyramid_gate_ceiling = float(pyramid_gate_ceiling)
        self.use_temporal_msad = bool(use_temporal_msad)
        self.use_source_prototype_memory = bool(
            use_source_prototype_memory
        )
        self.use_relative_degradation_fusion = bool(
            use_relative_degradation_fusion
        )
        self.rda_memory_topk = int(rda_memory_topk)
        self.rda_memory_temperature = float(rda_memory_temperature)
        self.rda_memory_strength = float(rda_memory_strength)
        self.rda_degradation_strength = float(rda_degradation_strength)
        self.rda_msad_strength = float(rda_msad_strength)

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
        self.temporal_msad = nn.ModuleList()
        self.temporal_msad_gate_logit: nn.Parameter | None = None
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
        self.multiview_pairs = tuple(
            (left, right)
            for left in range(len(self.scales))
            for right in range(left + 1, len(self.scales))
        )
        self.multiview_class_query: nn.Parameter | None = None
        self.multiview_query_projection: nn.Module = nn.Identity()
        self.multiview_key_projection: nn.Module = nn.Identity()
        self.multiview_value_projection: nn.Module = nn.Identity()
        self.multiview_pair_left = nn.ModuleList()
        self.multiview_pair_right = nn.ModuleList()
        self.multiview_pair_output = nn.ModuleList()
        self.rda_memory_projection: nn.Module = nn.Identity()
        self.rda_memory_gate_logit: nn.Parameter | None = None
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
            if self.multiview_fusion_mode != "none":
                self.multiview_class_query = nn.Parameter(
                    torch.empty(num_classes, d_model)
                )
                nn.init.trunc_normal_(self.multiview_class_query, std=0.02)
                self.multiview_query_projection = nn.Linear(
                    d_model, d_model, bias=False
                )
                self.multiview_key_projection = nn.Linear(
                    d_model, d_model, bias=False
                )
                self.multiview_value_projection = nn.Linear(
                    d_model, d_model, bias=False
                )
                if self.multiview_fusion_mode == "class_query_low_rank":
                    for _ in self.multiview_pairs:
                        self.multiview_pair_left.append(
                            nn.Linear(d_model, multiview_low_rank, bias=False)
                        )
                        self.multiview_pair_right.append(
                            nn.Linear(d_model, multiview_low_rank, bias=False)
                        )
                        self.multiview_pair_output.append(
                            nn.Sequential(
                                nn.LayerNorm(multiview_low_rank),
                                nn.Linear(multiview_low_rank, d_model),
                                nn.GELU(),
                            )
                        )
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

        # Optional R modules are initialized only after every shared H5
        # parameter. This isolates their RNG consumption: for the same seed,
        # H5/R0 and every R ablation start from byte-identical shared weights.
        if self.use_temporal_msad:
            self.temporal_msad = nn.ModuleList(
                [TemporalMSAD1D(d_model), TemporalMSAD1D(d_model)]
            )
            initial_gate = math.log(0.10 / 0.90)
            self.temporal_msad_gate_logit = nn.Parameter(
                torch.full((2,), initial_gate)
            )
        if self.use_source_prototype_memory:
            self.rda_memory_projection = nn.Linear(
                d_model, d_model, bias=False
            )
            nn.init.eye_(self.rda_memory_projection.weight)
            initial_memory_gate = math.log(0.10 / 0.90)
            self.rda_memory_gate_logit = nn.Parameter(
                torch.full((num_classes,), initial_memory_gate)
            )

    def _retrieve_source_prototypes(
        self,
        scale_embeddings: torch.Tensor,
        memory: torch.Tensor | None,
        memory_initialized: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Query every source class independently, without pseudo-label routing."""

        batch, scale_count, _ = scale_embeddings.shape
        empty_feature = scale_embeddings.new_zeros(
            (batch, scale_count, self.num_classes, self.d_model)
        )
        empty_distance = scale_embeddings.new_ones(
            (batch, scale_count, self.num_classes)
        )
        empty_valid = torch.zeros(
            (batch, scale_count, self.num_classes),
            dtype=torch.bool,
            device=scale_embeddings.device,
        )
        if memory is None or memory_initialized is None:
            return {
                "features": empty_feature,
                "distance": empty_distance,
                "valid": empty_valid,
            }
        memory = memory.detach().to(scale_embeddings)
        memory_initialized = memory_initialized.detach().to(
            scale_embeddings.device
        )
        if memory.ndim != 4 or memory.shape[:2] != (
            scale_count,
            self.num_classes,
        ) or memory.shape[-1] != self.d_model:
            raise ValueError("source memory must be [scales,classes,slots,d]")
        if memory_initialized.shape != memory.shape[:3]:
            raise ValueError("source memory mask must be [scales,classes,slots]")

        query = F.normalize(scale_embeddings, dim=-1)
        normalized_memory = F.normalize(memory, dim=-1)
        similarity = torch.einsum(
            "bsd,sckd->bsck", query, normalized_memory
        )
        similarity = similarity.masked_fill(
            ~memory_initialized.unsqueeze(0), float("-inf")
        )
        topk = min(self.rda_memory_topk, memory.shape[2])
        top_similarity, top_index = similarity.topk(topk, dim=-1)
        top_valid = torch.isfinite(top_similarity)
        safe_similarity = top_similarity.masked_fill(~top_valid, -1e4)
        slot_weight = F.softmax(
            safe_similarity / self.rda_memory_temperature, dim=-1
        )
        slot_weight = slot_weight * top_valid.to(slot_weight.dtype)
        slot_weight = slot_weight / slot_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        expanded_memory = memory.unsqueeze(0).expand(batch, -1, -1, -1, -1)
        selected = torch.gather(
            expanded_memory,
            3,
            top_index.unsqueeze(-1).expand(-1, -1, -1, -1, self.d_model),
        )
        retrieved = (selected * slot_weight.unsqueeze(-1)).sum(dim=3)
        weighted_similarity = (
            top_similarity.masked_fill(~top_valid, 0.0) * slot_weight
        ).sum(dim=-1)
        valid = top_valid.any(dim=-1)
        distance = (1.0 - weighted_similarity).clamp(0.0, 2.0)
        retrieved = retrieved.masked_fill(~valid.unsqueeze(-1), 0.0)
        distance = torch.where(valid, distance, torch.ones_like(distance))
        return {
            "features": retrieved,
            "distance": distance,
            "valid": valid,
        }

    def _multiview_class_features(
        self,
        projected_levels: torch.Tensor,
        scale_logits: torch.Tensor,
        scale_class_prior: torch.Tensor,
        source_scale_class_anchor: torch.Tensor | None = None,
        source_memory_features: torch.Tensor | None = None,
        source_memory_distance: torch.Tensor | None = None,
        source_memory_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Fuse scale views after their independent evidence is complete."""

        if self.multiview_class_query is None:
            raise RuntimeError("multiview class queries are unavailable")
        batch, scale_count, _ = projected_levels.shape
        tokens = [projected_levels[:, index] for index in range(scale_count)]
        independent_probability = F.softmax(scale_logits.detach(), dim=-1)
        entropy = -(
            independent_probability
            * torch.log(independent_probability.clamp_min(1e-8))
        ).sum(dim=-1) / math.log(self.num_classes)
        mean_probability = independent_probability.mean(dim=1, keepdim=True)
        conflict = (independent_probability - mean_probability).abs()
        token_uncertainty = [entropy[:, index] for index in range(scale_count)]
        token_conflict = [conflict[:, index] for index in range(scale_count)]
        token_prior = [
            scale_class_prior[:, index] for index in range(scale_count)
        ]
        token_degradation = []
        if source_memory_distance is not None:
            if source_memory_distance.shape != (
                batch,
                scale_count,
                self.num_classes,
            ):
                raise ValueError(
                    "source memory distance must be [batch,scales,classes]"
                )
            token_degradation = [
                source_memory_distance[:, index]
                for index in range(scale_count)
            ]

        if self.multiview_fusion_mode == "class_query_low_rank":
            for pair_index, (left, right) in enumerate(self.multiview_pairs):
                interaction = self.multiview_pair_output[pair_index](
                    self.multiview_pair_left[pair_index](
                        projected_levels[:, left]
                    )
                    * self.multiview_pair_right[pair_index](
                        projected_levels[:, right]
                    )
                )
                tokens.append(interaction)
                token_uncertainty.append(
                    0.5 * (entropy[:, left] + entropy[:, right])
                )
                token_conflict.append(
                    0.5 * (conflict[:, left] + conflict[:, right])
                )
                token_prior.append(
                    0.5
                    * (
                        scale_class_prior[:, left]
                        + scale_class_prior[:, right]
                    )
                )
                if token_degradation:
                    token_degradation.append(
                        0.5
                        * (
                            source_memory_distance[:, left]
                            + source_memory_distance[:, right]
                        )
                    )

        token_tensor = torch.stack(tokens, dim=1)
        uncertainty_tensor = torch.stack(token_uncertainty, dim=1)
        conflict_tensor = torch.stack(token_conflict, dim=1).transpose(1, 2)
        prior_tensor = torch.stack(token_prior, dim=1).transpose(1, 2)
        query = self.multiview_query_projection(self.multiview_class_query)
        key = self.multiview_key_projection(token_tensor)
        value = self.multiview_value_projection(token_tensor)
        score = torch.einsum("cd,btd->bct", query, key) / math.sqrt(
            self.d_model
        )
        score = score + torch.log(prior_tensor.clamp_min(1e-6))
        if self.use_multiview_uncertainty:
            score = score - (
                self.multiview_uncertainty_strength
                * uncertainty_tensor.unsqueeze(1)
                + self.multiview_conflict_strength * conflict_tensor
            )
        degradation_tensor = scale_logits.new_zeros(
            (batch, self.num_classes, token_tensor.shape[1])
        )
        if token_degradation:
            degradation_tensor = torch.stack(
                token_degradation, dim=1
            ).transpose(1, 2)
            if self.use_relative_degradation_fusion:
                score = score - (
                    self.rda_degradation_strength * degradation_tensor
                )
        dynamic_attention = F.softmax(score, dim=-1)

        # H5: shrink the sample-dependent attention toward a source-only
        # scale x class reliability graph.  A uniform anchor is the safe
        # fallback before the first source update; target pseudo-labels never
        # enter this path.  Pair-token reliability is the geometric mean of
        # its two constituent scale reliabilities, so both scales must support
        # a class before their interaction receives a strong anchor weight.
        if source_scale_class_anchor is None:
            source_scale_class_anchor = scale_logits.new_full(
                (scale_count, self.num_classes), 1.0 / scale_count
            )
        else:
            source_scale_class_anchor = source_scale_class_anchor.to(
                scale_logits
            )
            if source_scale_class_anchor.shape != (
                scale_count,
                self.num_classes,
            ):
                raise ValueError(
                    "source_scale_class_anchor must be [scales, classes]"
                )
            source_scale_class_anchor = (
                source_scale_class_anchor.clamp_min(1e-8)
                / source_scale_class_anchor.sum(dim=0, keepdim=True).clamp_min(
                    1e-8
                )
            )
        anchor_tokens = [
            source_scale_class_anchor[index]
            for index in range(scale_count)
        ]
        if self.multiview_fusion_mode == "class_query_low_rank":
            anchor_tokens.extend(
                torch.sqrt(
                    source_scale_class_anchor[left]
                    * source_scale_class_anchor[right]
                )
                for left, right in self.multiview_pairs
            )
        source_token_anchor = torch.stack(anchor_tokens, dim=0).transpose(0, 1)
        source_token_anchor = source_token_anchor / source_token_anchor.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        attention = (
            (1.0 - self.multiview_source_anchor_mix) * dynamic_attention
            + self.multiview_source_anchor_mix
            * source_token_anchor.unsqueeze(0)
        )
        attended = torch.einsum("bct,btd->bcd", attention, value)
        anchor_features = torch.einsum(
            "bsc,bsd->bcd", scale_class_prior, projected_levels
        )
        memory_scale_weight = scale_class_prior.new_zeros(
            (batch, scale_count, self.num_classes)
        )
        memory_class_feature = anchor_features.new_zeros(anchor_features.shape)
        memory_gate = anchor_features.new_zeros((batch, self.num_classes))
        if source_memory_features is not None:
            if source_memory_features.shape != (
                batch,
                scale_count,
                self.num_classes,
                self.d_model,
            ):
                raise ValueError(
                    "source memory features must be [batch,scales,classes,d]"
                )
            if source_memory_valid is None or source_memory_valid.shape != (
                batch,
                scale_count,
                self.num_classes,
            ):
                raise ValueError("source memory valid mask has invalid shape")
            memory_scale_weight = scale_class_prior * source_memory_valid.to(
                scale_class_prior.dtype
            )
            if self.use_relative_degradation_fusion:
                memory_scale_weight = memory_scale_weight * torch.exp(
                    -self.rda_degradation_strength
                    * source_memory_distance
                )
            memory_scale_weight = memory_scale_weight / memory_scale_weight.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            memory_class_feature = torch.einsum(
                "bsc,bscd->bcd",
                memory_scale_weight,
                source_memory_features,
            )
            if self.rda_memory_gate_logit is None:
                raise RuntimeError("source memory gate is unavailable")
            valid_class = source_memory_valid.any(dim=1)
            memory_gate = (
                self.rda_memory_strength
                * torch.sigmoid(self.rda_memory_gate_logit)
                .unsqueeze(0)
                .expand(batch, -1)
                * valid_class.to(anchor_features.dtype)
            )
        class_features = self.pyramid_output_norm(
            anchor_features
            + attended
            + memory_gate.unsqueeze(-1)
            * self.rda_memory_projection(memory_class_feature)
        )
        return {
            "class_features": class_features,
            "attention": attention,
            "dynamic_attention": dynamic_attention,
            "source_anchor": source_token_anchor,
            "uncertainty": uncertainty_tensor,
            "conflict": conflict_tensor,
            "degradation": degradation_tensor,
            "memory_scale_weight": memory_scale_weight,
            "memory_gate": memory_gate,
        }

    def _fuse(
        self,
        scale_embeddings: torch.Tensor,
        scale_logits: torch.Tensor,
        scale_class_reliability: torch.Tensor | None,
        multiview_source_anchor: torch.Tensor | None = None,
        pyramid_gate_ramp: float = 1.0,
        pyramid_bias_risk: torch.Tensor | None = None,
        source_memory_features: torch.Tensor | None = None,
        source_memory_distance: torch.Tensor | None = None,
        source_memory_valid: torch.Tensor | None = None,
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
            token_attention = class_weight.transpose(1, 2)
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
                "multiview_token_attention": token_attention,
                "multiview_dynamic_attention": token_attention,
                "multiview_source_anchor": token_attention[0],
                "multiview_token_uncertainty": logits.new_zeros(
                    (batch, scale_count)
                ),
                "multiview_token_conflict": logits.new_zeros(
                    (batch, self.num_classes, scale_count)
                ),
                "rda_token_degradation": logits.new_zeros(
                    (batch, self.num_classes, scale_count)
                ),
                "rda_memory_scale_weight": class_weight.new_zeros(
                    class_weight.shape
                ),
                "rda_memory_gate": zero_gate,
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
            token_attention = class_weight.transpose(1, 2)
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
                "multiview_token_attention": token_attention,
                "multiview_dynamic_attention": token_attention,
                "multiview_source_anchor": token_attention[0],
                "multiview_token_uncertainty": (
                    relation_weighted_logits.new_zeros((batch, scale_count))
                ),
                "multiview_token_conflict": (
                    relation_weighted_logits.new_zeros(
                        (batch, self.num_classes, scale_count)
                    )
                ),
                "rda_token_degradation": relation_weighted_logits.new_zeros(
                    (batch, self.num_classes, scale_count)
                ),
                "rda_memory_scale_weight": class_weight.new_zeros(
                    class_weight.shape
                ),
                "rda_memory_gate": zero_gate,
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
        multiview_attention = pyramid_class_weight.transpose(1, 2)
        multiview_dynamic_attention = multiview_attention
        multiview_anchor = multiview_attention[0]
        multiview_uncertainty = scale_logits.new_zeros((batch, scale_count))
        multiview_conflict = scale_logits.new_zeros(
            (batch, self.num_classes, scale_count)
        )
        rda_degradation = scale_logits.new_zeros(
            (batch, self.num_classes, scale_count)
        )
        rda_memory_scale_weight = pyramid_class_weight.new_zeros(
            pyramid_class_weight.shape
        )
        rda_memory_gate = scale_logits.new_zeros((batch, self.num_classes))
        if self.multiview_fusion_mode == "none":
            class_features = self.pyramid_output_norm(
                torch.einsum(
                    "bsc,bsd->bcd", pyramid_class_weight, projected_levels
                )
            )
        else:
            multiview = self._multiview_class_features(
                projected_levels,
                scale_logits,
                pyramid_class_weight,
                multiview_source_anchor,
                source_memory_features,
                source_memory_distance,
                source_memory_valid,
            )
            class_features = multiview["class_features"]
            multiview_attention = multiview["attention"]
            multiview_dynamic_attention = multiview["dynamic_attention"]
            multiview_anchor = multiview["source_anchor"]
            multiview_uncertainty = multiview["uncertainty"]
            multiview_conflict = multiview["conflict"]
            rda_degradation = multiview["degradation"]
            rda_memory_scale_weight = multiview["memory_scale_weight"]
            rda_memory_gate = multiview["memory_gate"]
            pyramid_class_weight = multiview_attention[
                :, :, :scale_count
            ].transpose(1, 2)
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
            if self.pyramid_gate_shrinkage > 0:
                class_static_gate = torch.sigmoid(
                    self.pyramid_class_gate_logit
                ).unsqueeze(0)
                raw_gate = (
                    (1.0 - self.pyramid_gate_shrinkage) * raw_gate
                    + self.pyramid_gate_shrinkage * class_static_gate
                )

        # A smooth ceiling retains gradients while preventing the adaptive
        # pyramid from dominating the relation-logit anchor in H5.
        if self.pyramid_gate_ceiling < 1.0:
            raw_gate = self.pyramid_gate_ceiling * torch.tanh(
                raw_gate / self.pyramid_gate_ceiling
            )

        raw_logit_delta = pyramid_logits - relation_weighted_logits
        guard_factor = torch.ones_like(raw_gate)
        unsupported_excess = torch.zeros_like(raw_gate)
        if self.use_pyramid_bias_guard or self.use_sign_aware_pyramid_guard:
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
            risk_guard = torch.exp(
                -self.pyramid_guard_strength * total_risk
            ).clamp(0.0, 1.0)
            if self.use_sign_aware_pyramid_guard:
                # An over-predicted class may still accept a correction that
                # lowers its logit. Only unsupported positive corrections are
                # attenuated; useful negative corrections remain untouched.
                guard_factor = torch.where(
                    raw_logit_delta.detach() > 0,
                    risk_guard,
                    torch.ones_like(risk_guard),
                )
            else:
                guard_factor = risk_guard

        effective_gate = raw_gate * float(pyramid_gate_ramp) * guard_factor
        logit_delta = raw_logit_delta
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
            "multiview_token_attention": multiview_attention,
            "multiview_dynamic_attention": multiview_dynamic_attention,
            "multiview_source_anchor": multiview_anchor,
            "multiview_token_uncertainty": multiview_uncertainty,
            "multiview_token_conflict": multiview_conflict,
            "rda_token_degradation": rda_degradation,
            "rda_memory_scale_weight": rda_memory_scale_weight,
            "rda_memory_gate": rda_memory_gate,
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
        multiview_source_anchor: torch.Tensor | None = None,
        class_logit_adjustment: torch.Tensor | None = None,
        pyramid_gate_ramp: float = 1.0,
        pyramid_bias_risk: torch.Tensor | None = None,
        source_prototype_memory: torch.Tensor | None = None,
        source_prototype_initialized: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if tuple(x_by_scale) != self.scale_keys:
            raise ValueError(
                f"Expected scales {self.scale_keys}, got {tuple(x_by_scale)}"
            )
        if tuple(mask_by_scale) != self.scale_keys:
            raise ValueError("Feature and mask scales do not match")

        pooled = []
        encoded_sequences = []
        for key in self.scale_keys:
            sequence = self.spatial(x_by_scale[key], mask_by_scale[key])
            encoded = self.temporal[key].encode_sequence(
                sequence, mask_by_scale[key]
            )
            encoded_sequences.append(encoded)
            pooled.append(
                self.temporal[key].pool_sequence(
                    encoded, mask_by_scale[key]
                )
            )
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
        contextual = self.scale_context(tokens)
        msad_filter_weights = scale_logits.new_zeros(
            (scale_logits.shape[0], max(len(self.scales) - 1, 1), 3)
        )
        msad_gate = scale_logits.new_zeros(max(len(self.scales) - 1, 1))
        if self.use_temporal_msad:
            if self.temporal_msad_gate_logit is None:
                raise RuntimeError("temporal MSAD gate is unavailable")
            residuals = torch.zeros_like(contextual)
            filter_summaries = []
            effective_gate = self.rda_msad_strength * torch.sigmoid(
                self.temporal_msad_gate_logit
            )
            for index, module in enumerate(self.temporal_msad):
                downsampled, down_mask, filter_weight = module(
                    encoded_sequences[index], mask_by_scale[self.scale_keys[index]]
                )
                residuals[:, index + 1] = (
                    effective_gate[index]
                    * module.pool(downsampled, down_mask)
                )
                filter_summaries.append(filter_weight.mean(dim=-1))
            contextual = contextual + residuals
            msad_filter_weights = torch.stack(filter_summaries, dim=1)
            msad_gate = effective_gate
        contextual_scale_embeddings = self.output_norm(contextual)
        memory_retrieval = self._retrieve_source_prototypes(
            scale_embeddings,
            source_prototype_memory if self.use_source_prototype_memory else None,
            (
                source_prototype_initialized
                if self.use_source_prototype_memory
                else None
            ),
        )
        fusion = self._fuse(
            contextual_scale_embeddings,
            scale_logits,
            scale_class_reliability,
            multiview_source_anchor=multiview_source_anchor,
            pyramid_gate_ramp=pyramid_gate_ramp,
            pyramid_bias_risk=pyramid_bias_risk,
            source_memory_features=(
                memory_retrieval["features"]
                if self.use_source_prototype_memory
                else None
            ),
            source_memory_distance=(
                memory_retrieval["distance"]
                if self.use_source_prototype_memory
                else None
            ),
            source_memory_valid=(
                memory_retrieval["valid"]
                if self.use_source_prototype_memory
                else None
            ),
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
            "multiview_token_attention": fusion[
                "multiview_token_attention"
            ],
            "multiview_dynamic_attention": fusion[
                "multiview_dynamic_attention"
            ],
            "multiview_source_anchor": fusion[
                "multiview_source_anchor"
            ],
            "multiview_token_uncertainty": fusion[
                "multiview_token_uncertainty"
            ],
            "multiview_token_conflict": fusion[
                "multiview_token_conflict"
            ],
            "rda_msad_filter_weight": msad_filter_weights,
            "rda_msad_gate": msad_gate,
            "rda_memory_distance": memory_retrieval["distance"],
            "rda_memory_valid": memory_retrieval["valid"],
            "rda_token_degradation": fusion["rda_token_degradation"],
            "rda_memory_scale_weight": fusion[
                "rda_memory_scale_weight"
            ],
            "rda_memory_gate": fusion["rda_memory_gate"],
            "scale_class_relation_residual": self.scale_class_relation_residual,
            "scale_domain_logits": scale_domain_logits,
            "fused_domain_logits": fused_domain_logits,
        }
