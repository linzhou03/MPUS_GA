import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trial_temporal_model import TrialTemporalDANN  # noqa: E402


def test_trial_model_forward_and_padding_invariance() -> None:
    torch.manual_seed(1)
    model = TrialTemporalDANN(
        d_model=16,
        num_heads=4,
        spatial_layers=1,
        temporal_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        spatial_topk=4,
    ).eval()
    x = torch.randn(2, 4, 62, 5)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    with torch.no_grad():
        logits, probability, domain, embedding, attention = model(x, mask)
        changed = x.clone()
        changed[1, 2:] = 1000.0
        changed_logits = model(changed, mask)[0]
    assert logits.shape == (2, 3)
    assert probability.shape == (2, 3)
    assert domain.shape == (2,)
    assert embedding.shape == (2, 16)
    assert attention.shape == (2, 4)
    assert torch.allclose(probability.sum(1), torch.ones(2), atol=1e-6)
    assert torch.allclose(attention[1, 2:], torch.zeros(2))
    assert torch.allclose(logits[1], changed_logits[1], atol=1e-6)
