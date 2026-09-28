"""Compute ripple-band envelopes from LFP arrays."""

from functools import lru_cache

import numpy as np
from scipy.signal import cheby2, hilbert, sosfiltfilt


@lru_cache(maxsize=16)
def _ripple_filter(sample_rate: float, low: float, high: float) -> np.ndarray:
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
    low: float = 100,
    high: float = 250,
) -> np.ndarray:
    """Return the Hilbert amplitude of ripple-band-filtered LFP data."""
    lfp = np.asarray(lfp)
    filtered = sosfiltfilt(_ripple_filter(sample_rate, low, high), lfp, axis=-1)
    filtered -= filtered.mean(axis=-1, keepdims=True)
    return np.abs(hilbert(filtered, axis=-1))


def zscore_from_baseline(envelope: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    """Robustly z-score an envelope using samples selected by ``baseline_mask``."""
    baseline = envelope[..., baseline_mask]
    median = np.median(baseline, axis=-1, keepdims=True)
    mad = np.median(np.abs(baseline - median), axis=-1, keepdims=True)
    return (envelope - median) / (1.4826 * mad + 1e-8)


def ripple_envelope(
    lfp: np.ndarray,
    baseline_mask: np.ndarray,
    sample_rate: float = 2_000,
) -> np.ndarray:
    """Compute a baseline-normalized ripple envelope for one or more channels."""
    return zscore_from_baseline(amplitude_envelope(lfp, sample_rate), baseline_mask)


def consensus_ripple_envelope(
    channel_envelopes: np.ndarray,
    sample_rate: float = 2_000,
    top_k: int = 3,
    smooth_seconds: float = 0.006,
) -> np.ndarray:
    """Combine the strongest channels at each sample into one ripple envelope."""
    channel_envelopes = np.asarray(channel_envelopes)
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    top_k = min(top_k, channel_envelopes.shape[0])
    strongest = np.partition(channel_envelopes, -top_k, axis=0)[-top_k:]
    consensus = np.median(strongest, axis=0)

    half_width = round(smooth_seconds * sample_rate)
    if half_width > 1:
        positions = np.arange(-half_width, half_width + 1)
        kernel = np.exp(-0.5 * (positions / half_width) ** 2)
        kernel /= kernel.sum()
        consensus = np.convolve(consensus, kernel, mode="same")
    return consensus
