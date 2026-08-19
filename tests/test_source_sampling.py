import torch

from train_transfer import source_sampling_weights


def _weighted_class_mass(labels: torch.Tensor, alpha: float) -> torch.Tensor:
    weights = source_sampling_weights(labels, alpha)
    return torch.stack([weights[labels == label].sum() for label in range(3)])


def test_alpha_zero_preserves_natural_class_mass() -> None:
    labels = torch.tensor([0, 0, 1, 2, 2, 2, 2])
    mass = _weighted_class_mass(labels, 0.0)
    assert torch.equal(mass, torch.tensor([2.0, 1.0, 4.0], dtype=torch.double))


def test_alpha_one_balances_class_mass() -> None:
    labels = torch.tensor([0, 0, 1, 2, 2, 2, 2])
    mass = _weighted_class_mass(labels, 1.0)
    assert torch.allclose(mass, torch.ones(3, dtype=torch.double))
