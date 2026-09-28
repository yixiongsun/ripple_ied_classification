"""Compute the waveform-normalization statistics used during event export."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt


SAMPLE_RATE = 2_000


@lru_cache(maxsize=1)
def _bandpass_filter() -> np.ndarray:
    """Return the notebook's fourth-order 1-300 Hz Butterworth filter."""
    nyquist = SAMPLE_RATE / 2
    sos = butter(
        4,
        [1 / nyquist, 300 / nyquist],
        btype="band",
        output="sos",
    )
    sos.setflags(write=False)
    return sos


def compute_lfp_statistics(
    lfp: np.ndarray,
    sleep_stages: np.ndarray,
) -> dict[str, float]:
    """Return the notebook's stage-2 statistics for one 2 kHz LFP channel.

    The full recording is filtered to 1-300 Hz. Its values are clipped to the
    1st and 99th percentiles measured during stage 2, and the normalization
    statistics are then calculated from the clipped stage-2 samples.
    """
    lfp = np.asarray(lfp).reshape(-1)
    sleep_stages = np.asarray(sleep_stages).reshape(-1)
    if len(lfp) != len(sleep_stages):
        raise ValueError("LFP and sleep stages must have the same length")

    stage_2 = sleep_stages == 2
    if not np.any(stage_2):
        raise ValueError("Cannot compute LFP statistics without stage-2 samples")

    # SciPy 1.18 requests a writable SOS buffer in its Cython filtering path.
    # Keep the cached coefficients immutable and copy only this small array;
    # the potentially very large read-only LFP memory map does not need copying.
    sos = _bandpass_filter().copy()
    filtered = sosfiltfilt(
        sos,
        lfp,
        padtype="odd",
        padlen=6 * sos.shape[0],
    )
    lower, upper = np.percentile(filtered[stage_2], [1, 99])
    baseline = np.clip(filtered, lower, upper)[stage_2]

    q25, q75 = np.percentile(baseline, [25, 75])
    return {
        "median": float(np.median(baseline)),
        "iqr": float(q75 - q25 + 1e-6),
        "mean": float(np.mean(baseline)),
        "std": float(np.std(baseline)),
    }
