"""Shared ripple/IED candidate extraction implementation."""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import butter, resample, sosfiltfilt, welch
from tqdm import tqdm


DEFAULT_SAMPLE_RATE = 2_000
N_SELECTED_CHANNELS = 3
WAVE_HALF_WINDOW = 100
PLOT_HALF_WINDOW = 300
CWT_HALF_WINDOW = 750
CWT_CROP = slice(650, 850, 2)
CWT_FREQUENCIES = np.geomspace(20.0, 300.0, num=100)
CWT_WAVELET = "cmor1.5-1.0"
_PLOT_LOCK = threading.Lock()


def _require_2khz(sample_rate: float) -> None:
    """Reject rates incompatible with the fixed event windows/model inputs."""
    if float(sample_rate) != DEFAULT_SAMPLE_RATE:
        raise ValueError(
            "Event extraction is fixed at 2,000 Hz because its sample windows "
            "and trained-model inputs are defined at that rate"
        )


def peak_ind(amplitude: np.ndarray, window: Sequence[int]) -> int:
    """Return the absolute index of the maximum inside ``[start, end)``."""
    start, end = map(int, window)
    if start < 0 or end <= start or end > len(amplitude):
        raise ValueError(f"Invalid event window ({start}, {end})")
    return int(np.argmax(amplitude[start:end]) + start)


@lru_cache(maxsize=32)
def _band_weights(band_size: int) -> np.ndarray:
    positions = np.linspace(-1.0, 1.0, band_size)
    weights = np.exp(-0.5 * (positions / 0.5) ** 2)[:, None]
    weights.setflags(write=False)
    return weights


def bandwise_norm_overlap(
    spec: np.ndarray,
    band_size: int = 20,
    step: int = 10,
    eps: float = 1e-8,
) -> np.ndarray:
    """Apply overlapping, robust band-wise normalization to an ``(F, T)`` array."""
    spec = np.asarray(spec)
    if spec.ndim != 2:
        raise ValueError(f"Expected a 2-D spectrogram, got shape {spec.shape}")
    n_freq, _ = spec.shape
    if not 0 < band_size <= n_freq:
        raise ValueError("band_size must be between 1 and the frequency dimension")
    if step <= 0:
        raise ValueError("step must be positive")

    out = np.zeros_like(spec, dtype=np.float64)
    weight_sum = np.zeros_like(spec, dtype=np.float64)
    weights = _band_weights(band_size)
    starts = list(range(0, n_freq - band_size + 1, step))
    final_start = n_freq - band_size
    if starts[-1] != final_start:
        starts.append(final_start)

    for start in starts:
        end = start + band_size
        band = spec[start:end]
        median = np.median(band)
        mad = np.median(np.abs(band - median))
        band_norm = (band - median) / (1.4826 * mad + eps)
        out[start:end] += band_norm * weights
        weight_sum[start:end] += weights
    np.divide(out, weight_sum, out=out, where=weight_sum > eps)
    return out


@lru_cache(maxsize=16)
def _bandpass_sos(sample_rate: float, order: int) -> np.ndarray:
    if sample_rate <= 600:
        raise ValueError("sample_rate must exceed 600 Hz for a 300 Hz low-pass edge")
    nyquist = 0.5 * sample_rate
    sos = butter(order, [1.0 / nyquist, 300.0 / nyquist], btype="band", output="sos")
    sos.setflags(write=False)
    return sos


def bandpass_1_300_padded(lfp: np.ndarray, fs: float, order: int = 4) -> np.ndarray:
    """Band-pass LFP data at 1–300 Hz using a cached SOS filter design."""
    lfp = np.asarray(lfp)
    # SciPy 1.18 requests a writable SOS buffer in its Cython filtering path.
    # Copy only the small coefficient array while leaving large memory-mapped
    # LFP inputs untouched.
    sos = _bandpass_sos(float(fs), int(order)).copy()
    return sosfiltfilt(sos, lfp, axis=-1, padtype="odd", padlen=6 * sos.shape[0])


def normalize_waveform_channel(
    segment: np.ndarray,
    fs: float,
    median: np.ndarray,
    iqr: np.ndarray,
    resample_len: int = 128,
    axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter, robust-scale, transform, and resample selected LFP channels."""
    segment = bandpass_1_300_padded(segment, fs)
    amplitude = np.max(np.abs(segment), axis=axis)
    median = np.asarray(median)
    iqr = np.asarray(iqr)
    if np.any(~np.isfinite(iqr)) or np.any(iqr <= 0):
        raise ValueError("Every selected channel must have a finite, positive IQR")
    transformed = np.arcsinh(((segment - median[:, None]) / iqr[:, None]) / 2.0)
    return resample(transformed, resample_len, axis=axis), amplitude


def normalize_spectrogram(spec: np.ndarray, baseline_spec: np.ndarray) -> np.ndarray:
    """Log/baseline normalize a ``(channels, frequency, time)`` spectrogram."""
    spec = np.asarray(spec)
    baseline_spec = np.asarray(baseline_spec)
    if spec.ndim != 3:
        raise ValueError(f"Expected a 3-D spectrogram, got shape {spec.shape}")
    if baseline_spec.ndim not in (2, 3):
        raise ValueError("baseline_spec must have 2 or 3 dimensions")

    normalized = []
    for channel in range(spec.shape[0]):
        current = np.log(spec[channel] + 1e-8)
        current -= baseline_spec[channel] if baseline_spec.ndim == 3 else baseline_spec
        current = np.clip(bandwise_norm_overlap(current), -5.0, 5.0)
        normalized.append(resample(current, 128, axis=1))
    return np.stack(normalized, axis=0)


def _empty_events() -> np.ndarray:
    return np.empty((0, 2), dtype=np.int64)


def merge_close_events(events: np.ndarray, fs: float, merge_gap: float = 0.015) -> np.ndarray:
    """Merge sorted event windows separated by at most ``merge_gap`` seconds."""
    events = np.asarray(events, dtype=np.int64)
    if events.size == 0:
        return _empty_events()
    events = events.reshape(-1, 2)
    merge_gap_samples = int(round(merge_gap * fs))
    merged: list[tuple[int, int]] = []
    current_start, current_end = map(int, events[0])
    for start, end in events[1:]:
        start, end = int(start), int(end)
        if start - current_end <= merge_gap_samples:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return np.asarray(merged, dtype=np.int64)


def reject_events(
    events: np.ndarray,
    aggregate: np.ndarray,
    fs: float,
    min_duration: float = 0.02,
    max_duration: float = 0.25,
    peak_threshold: float | None = None,
) -> np.ndarray:
    """Filter event windows by duration and peak envelope amplitude."""
    events = np.asarray(events, dtype=np.int64)
    if events.size == 0:
        return _empty_events()
    events = events.reshape(-1, 2)
    durations = (events[:, 1] - events[:, 0]) / fs
    keep = (durations >= min_duration) & (durations <= max_duration)
    if peak_threshold is not None:
        peaks = np.fromiter(
            (np.max(aggregate[start:end]) for start, end in events),
            dtype=np.float64,
            count=len(events),
        )
        keep &= peaks >= peak_threshold
    return events[keep]


def find_events(
    aggregate: np.ndarray,
    fs: float,
    z_entry: float = 3.0,
    z_exit: float = 3.0,
    peak_threshold: float = 7.0,
    min_duration: float = 0.02,
    max_duration: float = 0.25,
    merge_gap: float = 0.015,
) -> np.ndarray:
    """Detect threshold events using supra-exit runs and entry hysteresis."""
    aggregate = np.asarray(aggregate)
    if aggregate.ndim != 1:
        raise ValueError(f"Expected a 1-D consensus envelope, got {aggregate.shape}")
    if not np.all(np.isfinite(aggregate)):
        raise ValueError("Consensus envelope contains NaN or infinite values")
    if z_entry < z_exit:
        raise ValueError("z_entry must be greater than or equal to z_exit")
    if aggregate.size == 0:
        return _empty_events()

    # Every event is contained in a contiguous run at or above z_exit. Find
    # those run boundaries with one boolean array instead of retaining entry
    # and exit indices for potentially millions of samples.
    above_exit = aggregate >= z_exit
    transitions = np.flatnonzero(above_exit[1:] != above_exit[:-1]) + 1
    run_starts = transitions[above_exit[transitions]]
    run_ends = transitions[~above_exit[transitions]]
    if above_exit[0]:
        run_starts = np.concatenate(([0], run_starts))
    if above_exit[-1]:
        run_ends = np.concatenate((run_ends, [len(aggregate)]))

    raw_events: list[tuple[int, int]] = []
    for run_start, run_end in zip(run_starts, run_ends):
        entry_offsets = np.flatnonzero(aggregate[run_start:run_end] >= z_entry)
        if entry_offsets.size:
            raw_events.append((int(run_start + entry_offsets[0]), int(run_end)))
    if not raw_events:
        return _empty_events()

    events = np.asarray(raw_events, dtype=np.int64)
    if merge_gap > 0:
        events = merge_close_events(events, fs=fs, merge_gap=merge_gap)
    return reject_events(
        events,
        aggregate=aggregate,
        fs=fs,
        min_duration=min_duration,
        max_duration=max_duration,
        peak_threshold=peak_threshold,
    )


def select_top_k_events(
    events: np.ndarray,
    aggregate: np.ndarray,
    top_k: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable original indices and the strongest events in chronological order."""
    indices = np.arange(len(events), dtype=np.int64)
    if top_k is None or top_k >= len(events):
        return indices, events
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    peaks = np.fromiter(
        (np.max(aggregate[start:end]) for start, end in events),
        dtype=np.float64,
        count=len(events),
    )
    selected = np.argpartition(peaks, -top_k)[-top_k:]
    selected.sort()
    return indices[selected], events[selected]


def _validate_event_inputs(
    event: Sequence[int],
    detection_envelope: np.ndarray,
    scoring_envelopes: np.ndarray,
    context_envelopes: np.ndarray,
    lfps: Sequence[np.ndarray],
    medians: np.ndarray,
    iqrs: np.ndarray,
    baseline_spectra: np.ndarray,
) -> None:
    channel_count = len(lfps)
    if channel_count < N_SELECTED_CHANNELS:
        raise ValueError(f"At least {N_SELECTED_CHANNELS} LFP channels are required")
    if scoring_envelopes.shape[0] != channel_count or context_envelopes.shape[0] != channel_count:
        raise ValueError("Envelope channel counts do not match the LFP channel count")
    if len(medians) != channel_count or len(iqrs) != channel_count:
        raise ValueError("LFP statistics do not match the channel count")
    if baseline_spectra.shape[0] != channel_count:
        raise ValueError("Baseline spectrograms do not match the channel count")
    start, end = map(int, event)
    if start < 0 or end <= start or end > len(detection_envelope):
        raise ValueError(f"Invalid event window ({start}, {end})")


def _render_preview(
    path: Path,
    lfp_windows: Sequence[np.ndarray],
    event: Sequence[int],
    peak: int,
    cwt_power: np.ndarray,
    cwt_frequencies: np.ndarray,
    sample_rate: float,
) -> None:
    time = np.arange(cwt_power.shape[-1]) * 2.0 / sample_rate
    temporary = path.with_name(path.name + ".tmp")
    with _PLOT_LOCK:
        fig, axes = plt.subplots(
            N_SELECTED_CHANNELS,
            2,
            figsize=(4, 3),
            gridspec_kw={"width_ratios": [3, 1]},
            layout="constrained",
        )
        try:
            for row in range(N_SELECTED_CHANNELS):
                axes[row, 0].plot(lfp_windows[row], linewidth=1, color="black")
                axes[row, 0].axvspan(
                    PLOT_HALF_WINDOW - peak + event[0],
                    event[1] - peak + PLOT_HALF_WINDOW,
                    color="red",
                    alpha=0.3,
                    linewidth=0,
                )
                axes[row, 0].axis("off")
                axes[row, 1].pcolormesh(time, cwt_frequencies, cwt_power[row], shading="auto")
                axes[row, 1].set_xticks([])
                axes[row, 1].set_yticks([])
            fig.savefig(temporary, format="jpg", dpi=100, bbox_inches=None)
        finally:
            plt.close(fig)
    os.replace(temporary, path)


def process_event(
    *,
    index: int,
    event: np.ndarray,
    event_name: str,
    detection_envelope: np.ndarray,
    scoring_envelopes: np.ndarray,
    context_envelopes: np.ndarray,
    lfps: Sequence[np.ndarray],
    medians: np.ndarray,
    iqrs: np.ndarray,
    baseline_spectra: np.ndarray,
    output_directory: str | os.PathLike[str],
    cwt_scales: np.ndarray,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    overwrite: bool = False,
) -> str:
    """Create the JPG/JSON pair for one ripple or IED event."""
    _require_2khz(sample_rate)
    output_directory = Path(output_directory)
    json_path = output_directory / f"{event_name}{index:04d}.json"
    jpg_path = output_directory / f"{event_name}{index:04d}.jpg"
    if not overwrite and json_path.exists() and jpg_path.exists():
        return "skipped"

    _validate_event_inputs(
        event, detection_envelope, scoring_envelopes, context_envelopes,
        lfps, medians, iqrs, baseline_spectra,
    )
    peak = peak_ind(detection_envelope, event)
    signal_length = min(map(len, lfps))
    if peak - CWT_HALF_WINDOW < 0 or peak + CWT_HALF_WINDOW > signal_length:
        return "edge"

    start, end = map(int, event)
    peak_z = np.max(scoring_envelopes[:, start:end], axis=1)
    participation = np.mean(scoring_envelopes[:, start:end] > 3.0, axis=1)
    selected_channels = np.argsort(peak_z * participation)[-N_SELECTED_CHANNELS:][::-1]
    raw_waves = np.stack([
        lfps[channel][peak - WAVE_HALF_WINDOW : peak + WAVE_HALF_WINDOW]
        for channel in selected_channels
    ])
    plot_waves = [
        lfps[channel][peak - PLOT_HALF_WINDOW : peak + PLOT_HALF_WINDOW]
        for channel in selected_channels
    ]
    context = np.stack([
        context_envelopes[channel, peak - WAVE_HALF_WINDOW : peak + WAVE_HALF_WINDOW]
        for channel in selected_channels
    ])

    cwt_magnitudes = []
    cwt_frequencies = None
    for channel in selected_channels:
        cwt_matrix, cwt_frequencies = pywt.cwt(
            lfps[channel][peak - CWT_HALF_WINDOW : peak + CWT_HALF_WINDOW],
            cwt_scales,
            CWT_WAVELET,
            sampling_period=1.0 / sample_rate,
        )
        cwt_magnitudes.append(np.abs(cwt_matrix[:, CWT_CROP]))
    cwt_magnitudes = np.stack(cwt_magnitudes)

    frequencies, power = welch(raw_waves, sample_rate, nperseg=raw_waves.shape[-1], axis=-1)
    ripple_power = power[:, (frequencies >= 100) & (frequencies <= 250)].sum(axis=1)
    low_power = power[:, (frequencies >= 1) & (frequencies <= 80)].sum(axis=1)
    ratio = ripple_power / (low_power + 1e-8)
    probability = power / (power.sum(axis=1, keepdims=True) + 1e-8)
    entropy = -np.sum(probability * np.log(probability + 1e-8), axis=1)

    normalized_wave, amplitude = normalize_waveform_channel(
        raw_waves, sample_rate, medians[selected_channels], iqrs[selected_channels], resample_len=128,
    )
    normalized_spec = normalize_spectrogram(cwt_magnitudes**2, baseline_spectra[selected_channels])
    context = resample(context, 128, axis=1)
    _render_preview(
        jpg_path, plot_waves, event, peak, cwt_magnitudes,
        np.asarray(cwt_frequencies), sample_rate,
    )

    record = {
        "sample_rate": sample_rate,
        "window": [start, end],
        "wave": normalized_wave.tolist(),
        "spec": normalized_spec.tolist(),
        "freqs": np.asarray(cwt_frequencies).tolist(),
        "ch": selected_channels.tolist(),
        "ratio": ratio.tolist(),
        "entropy": entropy.tolist(),
        "ripple_env": context.tolist(),
        "amp": amplitude.tolist(),
    }
    temporary_json = json_path.with_name(json_path.name + ".tmp")
    with temporary_json.open("w", encoding="utf-8") as output:
        json.dump(record, output, separators=(",", ":"), allow_nan=False)
    os.replace(temporary_json, json_path)
    return "saved"


def process_events(
    events: np.ndarray,
    *,
    event_name: str,
    detection_envelope: np.ndarray,
    scoring_envelopes: np.ndarray,
    context_envelopes: np.ndarray,
    lfps: Sequence[np.ndarray],
    medians: np.ndarray,
    iqrs: np.ndarray,
    baseline_spectra: np.ndarray,
    output_directory: str | os.PathLike[str],
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    workers: int = 4,
    top_k: int | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Counter[str]:
    """Write JPG/JSON pairs for a collection of already detected events."""
    _require_2khz(sample_rate)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    event_indices, selected_events = select_top_k_events(events, detection_envelope, top_k)
    indexed_events = list(zip(event_indices.tolist(), selected_events))
    scales = pywt.frequency2scale(CWT_WAVELET, CWT_FREQUENCIES / sample_rate)

    def process(item: tuple[int, np.ndarray]) -> str:
        index, event = item
        return process_event(
            index=index,
            event=event,
            event_name=event_name,
            detection_envelope=detection_envelope,
            scoring_envelopes=scoring_envelopes,
            context_envelopes=context_envelopes,
            lfps=lfps,
            medians=medians,
            iqrs=iqrs,
            baseline_spectra=baseline_spectra,
            output_directory=output_directory,
            cwt_scales=scales,
            sample_rate=sample_rate,
            overwrite=overwrite,
        )

    if workers == 1:
        statuses = [
            process(item)
            for item in tqdm(
                indexed_events,
                desc=event_name,
                disable=not show_progress,
            )
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            statuses = list(
                tqdm(
                    executor.map(process, indexed_events),
                    total=len(indexed_events),
                    desc=event_name,
                    disable=not show_progress,
                )
            )
    return Counter(statuses)
