import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transfer_losses import (  # noqa: E402
    balanced_reliable_gate,
    entropy_mi_reliability,
    graph_ramp_weight,
    soft_graph_edge_loss,
)


class TransferLossTests(unittest.TestCase):
    def test_reliability_prefers_certain_stable_prediction(self):
        certain = torch.tensor([0.98, 0.01, 0.01])
        uniform = torch.tensor([1 / 3, 1 / 3, 1 / 3])
        probabilities = torch.stack((certain, uniform)).repeat(3, 1, 1)
        _, entropy, mi, reliability = entropy_mi_reliability(probabilities)
        self.assertGreater(float(reliability[0]), float(reliability[1]))
        self.assertLess(float(mi.max()), 1e-6)
        self.assertLess(float(entropy[0]), float(entropy[1]))

    def test_balanced_gate_keeps_equal_class_counts(self):
        probability = torch.tensor(
            [
                [0.9, 0.05, 0.05],
                [0.8, 0.1, 0.1],
                [0.05, 0.9, 0.05],
                [0.1, 0.8, 0.1],
                [0.05, 0.05, 0.9],
            ]
        )
        selected, raw = balanced_reliable_gate(
            probability, torch.ones(5), 0.7, 0.2
        )
        labels = probability.argmax(1)[selected]
        self.assertEqual(raw.tolist(), [2, 2, 1])
        self.assertEqual(torch.bincount(labels, minlength=3).tolist(), [1, 1, 1])

    def test_soft_edge_loss_backpropagates_and_ramp_is_bounded(self):
        source_labels = torch.tensor([0, 1, 2])
        target_probability = torch.tensor(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        )
        reliability = torch.tensor([0.9, 0.8, 0.7])
        affinity = torch.randn(6, 6, requires_grad=True)
        loss, blocks = soft_graph_edge_loss(
            affinity,
            source_labels,
            target_probability,
            reliability,
            torch.ones(3, dtype=torch.bool),
            1.0,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(affinity.grad)
        self.assertTrue(all(torch.isfinite(value) for value in blocks.values()))
        self.assertEqual(graph_ramp_weight(200, 200, 200), 0.0)
        self.assertAlmostEqual(graph_ramp_weight(300, 200, 200), 0.5)
        self.assertEqual(graph_ramp_weight(400, 200, 200), 1.0)


if __name__ == "__main__":
    unittest.main()
