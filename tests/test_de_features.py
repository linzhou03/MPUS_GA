from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from de_features import extract_multiscale_de, validate_window_seconds


class DifferentialEntropyTests(unittest.TestCase):
    def test_multiscale_shapes_and_window_starts(self) -> None:
        sfreq = 200.0
        seconds = 10
        time = np.arange(int(sfreq * seconds)) / sfreq
        signal = np.sin(2 * np.pi * 10.0 * time)
        trial = np.tile(signal, (62, 1)).astype(np.float32)

        outputs = extract_multiscale_de(trial, sfreq, (1, 2, 4))

        self.assertEqual(outputs[1.0][0].shape, (10, 62, 5))
        self.assertEqual(outputs[2.0][0].shape, (5, 62, 5))
        self.assertEqual(outputs[4.0][0].shape, (2, 62, 5))
        np.testing.assert_array_equal(outputs[1.0][1], np.arange(10))
        np.testing.assert_array_equal(outputs[2.0][1], np.arange(0, 10, 2))
        np.testing.assert_array_equal(outputs[4.0][1], np.arange(0, 8, 4))

    def test_alpha_band_dominates_for_ten_hz_signal(self) -> None:
        sfreq = 200.0
        time = np.arange(int(sfreq * 12)) / sfreq
        signal = np.sin(2 * np.pi * 10.0 * time)
        trial = np.tile(signal, (62, 1)).astype(np.float32)

        features, _ = extract_multiscale_de(trial, sfreq, (2,))[2.0]
        band_means = features.mean(axis=(0, 1))
        self.assertEqual(int(band_means.argmax()), 2)

    def test_reject_duplicate_scales(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_window_seconds((1, 1, 2), 200.0)


if __name__ == "__main__":
    unittest.main()
