"""Compute IED-band envelopes from LFP arrays."""

from functools import lru_cache

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import cheby2, hilbert, sosfiltfilt


@lru_cache(maxsize=16)
def _ied_filter(sample_rate: float, low: float, high: float) -> np.ndarray:
    nyquist = sample_rate / 2
    return cheby2(
        4,
        20,
        [low / nyquist, high / nyquist],
        btype="bandpass",
        output="sos",
    )


def amplitude_envelope(
    lfp: np.ndarray,
    sample_rate: float,
    low: float = 20,
    high: float = 80,
) -> np.ndarray:
    """Return the Hilbert amplitude of IED-band-filtered LFP data."""
    lfp = np.asarray(lfp)
    filtered = sosfiltfilt(_ied_filter(sample_rate, low, high), lfp, axis=-1)
    filtered -= filtered.mean(axis=-1, keepdims=True)
    return np.abs(hilbert(filtered, axis=-1))


def zscore_from_baseline(envelope: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    """Z-score an envelope using samples selected by ``baseline_mask``."""
    baseline = envelope[..., baseline_mask]
    mean = baseline.mean(axis=-1, keepdims=True)
    std = baseline.std(axis=-1, keepdims=True)
    return (envelope - mean) / (std + 1e-8)


def ied_envelope(
    lfp: np.ndarray,
    baseline_mask: np.ndarray,
    sample_rate: float = 2_000,
) -> np.ndarray:
    """Compute a baseline-normalized IED envelope for one or more channels."""
    return zscore_from_baseline(amplitude_envelope(lfp, sample_rate), baseline_mask)


def consensus_ied_envelope(
    channel_envelopes: np.ndarray,
    sample_rate: float = 2_000,
    rms_window_ms: float = 20,
) -> np.ndarray:
    """Combine channel envelopes using the historical moving-RMS calculation."""
    window = max(1, round(sample_rate * rms_window_ms / 1_000))
    mean_square = uniform_filter1d(
        np.asarray(channel_envelopes) ** 2,
        size=window,
        axis=-1,
        mode="nearest",
    )
    return np.sqrt(mean_square).mean(axis=0)
