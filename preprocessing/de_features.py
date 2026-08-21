"""Multiscale differential-entropy extraction from trial-level raw EEG."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from scipy.signal import butter, sosfiltfilt


BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 14.0),
    ("beta", 14.0, 31.0),
    ("gamma", 31.0, 50.0),
)


def validate_window_seconds(
    window_seconds: Iterable[float], sfreq: float
) -> tuple[float, ...]:
    values = tuple(float(value) for value in window_seconds)
    if not values or any(value <= 0 for value in values):
        raise ValueError("window_seconds must contain positive values")
    if len(set(values)) != len(values):
        raise ValueError(f"window_seconds contains duplicates: {values}")
    for value in values:
        sample_count = value * sfreq
        if not math.isclose(sample_count, round(sample_count), abs_tol=1e-8):
            raise ValueError(
                f"Window {value}s does not contain an integer number of samples "
                f"at {sfreq} Hz"
            )
    return tuple(sorted(values))


def filter_trial_into_bands(
    trial: np.ndarray,
    sfreq: float,
    bands: tuple[tuple[str, float, float], ...] = BANDS,
    order: int = 4,
) -> np.ndarray:
    """Return zero-phase band-filtered EEG shaped ``[bands, channels, time]``."""

    values = np.asarray(trial)
    if values.ndim != 2:
        raise ValueError(f"Expected [channels, time] trial, got {values.shape}")
    if values.shape[0] != 62:
        raise ValueError(f"Expected 62 EEG channels, got {values.shape[0]}")
    if values.shape[1] < int(round(4 * sfreq)):
        raise ValueError(
            f"Trial has only {values.shape[1]} samples at {sfreq} Hz; "
            "at least four seconds are required"
        )
    if not np.isfinite(values).all():
        raise ValueError("Raw trial contains NaN or infinite values")

    nyquist = sfreq / 2.0
    filtered = np.empty((len(bands), *values.shape), dtype=np.float32)
    work = values.astype(np.float64, copy=False)
    for band_index, (_, low_hz, high_hz) in enumerate(bands):
        if not 0 < low_hz < high_hz < nyquist:
            raise ValueError(
                f"Invalid band ({low_hz}, {high_hz}) for {sfreq} Hz sampling"
            )
        sos = butter(
            order,
            [low_hz, high_hz],
            btype="bandpass",
            fs=sfreq,
            output="sos",
        )
        filtered[band_index] = sosfiltfilt(sos, work, axis=-1).astype(
            np.float32, copy=False
        )
    return filtered


def differential_entropy_windows(
    band_filtered_trial: np.ndarray,
    sfreq: float,
    window_seconds: float,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute non-overlapping DE windows.

    Returns ``(features, start_seconds)`` where features are shaped
    ``[windows, channels, bands]``.
    """

    values = np.asarray(band_filtered_trial)
    if values.ndim != 3:
        raise ValueError(
            f"Expected [bands, channels, time] filtered trial, got {values.shape}"
        )
    if values.shape[:2] != (len(BANDS), 62):
        raise ValueError(
            f"Expected [{len(BANDS)}, 62, time] filtered trial, got {values.shape}"
        )

    window_samples = int(round(float(window_seconds) * sfreq))
    if window_samples <= 0:
        raise ValueError("window_seconds must be positive")
    window_count = values.shape[-1] // window_samples
    if window_count == 0:
        return (
            np.empty((0, 62, len(BANDS)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    trimmed = values[..., : window_count * window_samples]
    blocks = trimmed.reshape(
        len(BANDS), 62, window_count, window_samples
    )
    variance = blocks.var(axis=-1, dtype=np.float64)
    de = 0.5 * np.log(2.0 * np.pi * np.e * np.maximum(variance, epsilon))
    features = de.transpose(2, 1, 0).astype(np.float32, copy=False)
    starts = (
        np.arange(window_count, dtype=np.float32) * float(window_seconds)
    )
    return features, starts


def extract_multiscale_de(
    trial: np.ndarray,
    sfreq: float,
    window_seconds: Iterable[float],
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Filter a trial once and calculate each requested DE window scale."""

    scales = validate_window_seconds(window_seconds, sfreq)
    filtered = filter_trial_into_bands(trial, sfreq)
    return {
        scale: differential_entropy_windows(filtered, sfreq, scale)
        for scale in scales
    }
