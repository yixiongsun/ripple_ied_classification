"""Create per-channel background spectrograms for event normalization.

This is the reusable Python replacement for ``baseline_spectrograms.ipynb``.
It accepts ordinary NumPy arrays and does not know about subjects or directory
layouts.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
from scipy.signal import resample
from tqdm import tqdm


CWT_WAVELET = "cmor1.5-1.0"
CWT_FREQUENCIES = np.geomspace(20.0, 300.0, num=100)


class InsufficientBaselineSamplesError(ValueError):
    """Raised when a session cannot provide one complete baseline CWT window."""


@lru_cache(maxsize=8)
def cwt_scales(sample_rate: float) -> np.ndarray:
    """Return the notebook's 20–300 Hz CWT scales for a sample rate."""
    scales = pywt.frequency2scale(CWT_WAVELET, CWT_FREQUENCIES / sample_rate)
    scales.setflags(write=False)
    return scales


def compute_baseline_cwt(
    lfp_chunk: np.ndarray,
    sample_rate: float = 2_000,
    *,
    time_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one background LFP chunk into a centered log-power CWT."""
    cwt_matrix, frequencies = pywt.cwt(
        np.asarray(lfp_chunk),
        cwt_scales(float(sample_rate)),
        CWT_WAVELET,
        sampling_period=1.0 / sample_rate,
    )
    log_power = np.log(np.abs(cwt_matrix) ** 2 + 1e-8)
    log_power = resample(log_power, time_bins, axis=1)
    log_power -= np.median(log_power)
    return log_power, frequencies


def background_mask(
    sleep_stages: np.ndarray,
    event_windows: Iterable[Sequence[int]],
    *,
    baseline_stage: int = 2,
) -> np.ndarray:
    """Select baseline-stage samples while excluding ripple and IED windows."""
    mask = np.asarray(sleep_stages).reshape(-1) == baseline_stage
    for start, end in event_windows:
        start = max(0, int(start))
        end = min(len(mask), int(end))
        if start < end:
            mask[start:end] = False
    return mask


def overlapping_chunks(
    signal: np.ndarray,
    chunk_size: int,
    step_size: int,
):
    """Yield complete overlapping chunks along a one-dimensional signal."""
    for start in range(0, len(signal) - chunk_size + 1, step_size):
        yield signal[start : start + chunk_size]


def compute_baseline_spectrogram(
    lfp: np.ndarray,
    mask: np.ndarray,
    sample_rate: float = 2_000,
    *,
    window_seconds: float = 4.0,
    step_seconds: float = 2.0,
    time_bins: int = 100,
    smoothing_sigma: tuple[float, float] = (1.0, 2.0),
    show_progress: bool = True,
) -> np.ndarray:
    """Compute the notebook's median, smoothed background spectrogram."""
    lfp = np.asarray(lfp).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if len(lfp) != len(mask):
        raise ValueError("LFP and baseline mask must have the same length")

    background = lfp[mask]
    chunk_size = int(round(window_seconds * sample_rate))
    step_size = int(round(step_seconds * sample_rate))
    if chunk_size < 1 or step_size < 1:
        raise ValueError("Window and step durations must be positive")

    chunk_count = 1 + (len(background) - chunk_size) // step_size
    if chunk_count < 1:
        raise InsufficientBaselineSamplesError(
            "Not enough baseline samples for one complete CWT window"
        )

    spectra = np.empty((chunk_count, len(CWT_FREQUENCIES), time_bins))
    chunks = overlapping_chunks(background, chunk_size, step_size)
    chunks = tqdm(
        chunks,
        total=chunk_count,
        desc="baseline CWT",
        disable=not show_progress,
    )
    for index, chunk in enumerate(chunks):
        spectra[index], _ = compute_baseline_cwt(
            chunk, sample_rate, time_bins=time_bins
        )

    baseline = np.median(spectra, axis=0)
    return gaussian_filter(baseline, sigma=smoothing_sigma)


# Historical notebook name.
compute_baseline_cwt_100 = compute_baseline_cwt
