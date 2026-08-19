import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from layers import (
    SpatialStream,
    PositionalEncoding,
    CoAttnEncoderLayer,
    GRL,
    DomainClassifier,
)
from golden_style import StyleAlignLayer, GoldenStyleBank


class GSA_CAST(nn.Module):
    def __init__(
        self,
        input_feat_dim=450,
        num_nodes=62,
        d_model=64,
        n_head=4,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.3,
        topk=8,
        num_classes=3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_feat_dim, d_model)
        spatial_layer_dims = [d_model] * num_encoder_layers
        self.spatial_stream = SpatialStream(d_model, spatial_layer_dims, topk)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.transformer_encoders = nn.ModuleList(
            [
                CoAttnEncoderLayer(d_model, n_head, dim_feedforward, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.style_layers = nn.ModuleList(
            [StyleAlignLayer(d_model) for _ in range(num_encoder_layers)]
        )
        self.gsb = GoldenStyleBank(num_encoder_layers, d_model)

        self.grl = GRL(alpha=1.0)
        self.domain_classifier = DomainClassifier(d_model)
        self.linend = nn.Linear(d_model, num_classes)
        self.dropout_final = nn.Dropout(dropout)

    def forward(
        self, x, edge_index, batch, return_style_info=False, style_bootstrap_pair=None
    ):
        x, mask = to_dense_batch(x, batch)
        projected = self.input_proj(x)
        spatial_features_list = self.spatial_stream(projected)
        temporal = self.pos_encoder(projected)

        style_infos = []

        for layer_idx, encoder in enumerate(self.transformer_encoders):
            temporal = encoder(temporal, spatial_features_list[layer_idx])

            if style_bootstrap_pair is not None:
                src_f, tgt_f = style_bootstrap_pair
                self.gsb.maybe_bootstrap(layer_idx, src_f[layer_idx], tgt_f[layer_idx])

            mu_g, sig_g = self.gsb.stats(layer_idx)
            y, alpha = self.style_layers[layer_idx](temporal, mu_g, sig_g)
            if return_style_info:
                style_infos.append(
                    {
                        # Keep the loss-bearing tensors attached to autograd. The
                        # original release detached these values, which disabled
                        # the complete composite style objective.
                        "pre": temporal,
                        "post": y,
                        "mu_g": mu_g,
                        "sig_g": sig_g,
                        "alpha": alpha.detach(),
                    }
                )
            temporal = y

        mask_float = mask.unsqueeze(-1).to(temporal.dtype)
        aggregated = (temporal * mask_float).sum(dim=1) / mask_float.sum(
            dim=1
        ).clamp_min(1.0)
        aggregated = self.dropout_final(aggregated)
        reversed_features = self.grl(aggregated)
        domain_output = self.domain_classifier(reversed_features)
        class_output = self.linend(aggregated)
        pred = F.softmax(class_output, dim=1)

        if return_style_info:
            return (
                class_output,
                pred,
                domain_output.view(-1),
                aggregated,
                style_infos,
                spatial_features_list,
            )

        return class_output, pred, domain_output.view(-1), aggregated
