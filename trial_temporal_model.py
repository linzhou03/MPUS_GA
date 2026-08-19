"""Spatial-window encoder plus a true across-window trial encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from layers import DomainClassifier, GRL, PositionalEncoding, SpatialStream  # noqa: E402


class TrialTemporalDANN(nn.Module):
    """Encode electrode graphs per window, then classify the complete trial."""

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        num_heads: int = 4,
        spatial_layers: int = 2,
        temporal_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        spatial_topk: int = 8,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.spatial_stream = SpatialStream(
            d_model, [d_model] * spatial_layers, spatial_topk
        )
        self.electrode_attention = nn.Linear(d_model, 1)
        self.position = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=temporal_layers
        )
        self.window_attention = nn.Linear(d_model, 1)
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.grl = GRL(alpha=1.0)
        self.domain_classifier = DomainClassifier(d_model)

    def forward(
        self,
        x: torch.Tensor,
        window_mask: torch.Tensor,
        grl_alpha: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, windows, electrodes, bands]")
        if window_mask.shape != x.shape[:2] or window_mask.dtype != torch.bool:
            raise ValueError("window_mask must be boolean with shape [batch, windows]")
        if torch.any(~window_mask.any(dim=1)):
            raise ValueError("Every trial must contain at least one real window")

        batch_size, max_windows = x.shape[:2]
        valid_windows = x[window_mask]
        spatial = self.input_projection(valid_windows)
        spatial = self.spatial_stream(spatial)[-1]
        electrode_weight = F.softmax(self.electrode_attention(spatial), dim=1)
        window_embedding = (spatial * electrode_weight).sum(dim=1)

        sequence = window_embedding.new_zeros(
            (batch_size, max_windows, window_embedding.shape[-1])
        )
        sequence[window_mask] = window_embedding
        sequence = self.position(sequence)
        sequence = self.temporal_encoder(
            sequence, src_key_padding_mask=~window_mask
        )

        attention_logits = self.window_attention(sequence).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~window_mask, float("-inf"))
        window_weight = F.softmax(attention_logits, dim=1)
        trial_embedding = (sequence * window_weight.unsqueeze(-1)).sum(dim=1)
        trial_embedding = self.output_dropout(trial_embedding)

        class_logits = self.classifier(trial_embedding)
        probability = F.softmax(class_logits, dim=1)
        self.grl.alpha = float(grl_alpha)
        domain_logits = self.domain_classifier(self.grl(trial_embedding)).view(-1)
        return class_logits, probability, domain_logits, trial_embedding, window_weight
