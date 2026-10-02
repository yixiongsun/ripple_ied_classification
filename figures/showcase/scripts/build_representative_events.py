"""Build the representative ripple, IED, and ambiguous/noise event figure."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.ndimage import gaussian_filter

import subjects
from dataset_pipeline._event_extraction import CWT_FREQUENCIES, CWT_WAVELET, peak_ind
from showcase_style import COLORS, configure_matplotlib, export_figure


SHOWCASE = ROOT / "figures" / "showcase"
CLASS_COLORS = {"ripple": COLORS["ripple"], "ied": COLORS["ied"], "noise": COLORS["noise"]}
CLASS_TITLES = {"ripple": "Ripple", "ied": "IED", "noise": "Ambiguous / noise"}
SAMPLE_RATE = 2_000
DISPLAY_HALF_SAMPLES = 300


def display_arrays(record: dict) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Reconstruct the 300 ms preview window used by the source event JPG."""
    subject = subjects.load(record["source_subject"])
    session_directory = Path(subject["olm"]["directory"]) / record["session"]
    file_name = record["source_file_name"]
    candidate_type = "ripple" if file_name.startswith("ripple") else "ied"
    event_path = session_directory / f"{candidate_type}s" / file_name
    event = json.loads(event_path.read_text(encoding="utf-8"))
    window = event["window"]
    envelope = np.load(session_directory / f"consensus_{candidate_type}_env.npy", mmap_mode="r")
    peak = peak_ind(envelope, window)
    channel_names = [subject["olm"]["lfp_channels"][int(index)] for index in event["ch"]]
    scales = pywt.frequency2scale(CWT_WAVELET, CWT_FREQUENCIES / SAMPLE_RATE)

    waves = []
    normalized_log_powers = []
    for channel in channel_names:
        lfp = np.load(session_directory / f"{channel}_cleaned_hpc.npy", mmap_mode="r")[1]
        segment = np.asarray(lfp[peak - DISPLAY_HALF_SAMPLES : peak + DISPLAY_HALF_SAMPLES])
        stats = np.load(session_directory / f"{channel}_lfp_stats.npy", allow_pickle=True).item()
        waves.append(np.arcsinh(((segment - stats["median"]) / stats["iqr"]) / 2.0))
        cwt_matrix, _ = pywt.cwt(
            segment,
            scales,
            CWT_WAVELET,
            sampling_period=1.0 / SAMPLE_RATE,
        )
        log_power = np.log(np.abs(cwt_matrix) ** 2 + 1e-8)
        baseline = np.load(session_directory / f"{channel}_baseline_spec.npy")
        baseline_center = np.median(baseline, axis=1, keepdims=True)
        baseline_scale = 1.4826 * np.median(np.abs(baseline - baseline_center), axis=1, keepdims=True)
        normalized_log_powers.append((log_power - baseline_center) / np.maximum(baseline_scale, 0.15))

    spectrum = np.mean(np.stack(normalized_log_powers), axis=0)
    spectrum = gaussian_filter(spectrum, sigma=(0.65, 1.0))
    event_span = (
        (float(window[0]) - peak) * 1_000 / SAMPLE_RATE,
        (float(window[1]) - peak) * 1_000 / SAMPLE_RATE,
    )
    return np.stack(waves), spectrum, event_span


def main() -> None:
    configure_matplotlib()
    selection = json.loads((SHOWCASE / "data" / "representative_events.json").read_text(encoding="utf-8"))
    records = selection["events"]
    chosen = [display_arrays(record) for record in records]

    spectrum_values = np.concatenate([spectrum.ravel() for _, spectrum, _ in chosen])
    spec_low, spec_high = np.quantile(spectrum_values, (0.03, 0.97))

    fig = plt.figure(figsize=(7.6, 6.5), dpi=100, facecolor=COLORS["page"])
    grid = fig.add_gridspec(
        4,
        3,
        left=0.075,
        right=0.975,
        bottom=0.11,
        top=0.96,
        height_ratios=(0.15, 1.0, 1.1, 0.22),
        hspace=0.24,
        wspace=0.14,
    )

    time = np.linspace(-150, 150, DISPLAY_HALF_SAMPLES * 2, endpoint=False)
    channel_offsets = np.asarray((2.35, 0.0, -2.35))
    image = None
    for column, (record, (waves, spectrum, event_span)) in enumerate(zip(records, chosen)):
        class_name = record["class"]
        title_ax = fig.add_subplot(grid[0, column])
        title_ax.axis("off")
        title_ax.text(0, 0.72, CLASS_TITLES[class_name], color=CLASS_COLORS[class_name], fontsize=13, weight="bold")
        title_ax.text(0, 0.10, record["display_subject"], color=COLORS["muted"], fontsize=8.5)

        wave_ax = fig.add_subplot(grid[1, column])
        wave_ax.set_facecolor(COLORS["surface"])
        panel_wave_limit = float(np.quantile(np.abs(waves), 0.995))
        for channel, offset in enumerate(channel_offsets):
            wave_ax.plot(time, waves[channel] / panel_wave_limit + offset, color=COLORS["secondary"], lw=0.8)
        wave_ax.axvspan(*event_span, facecolor=CLASS_COLORS[class_name], alpha=0.13, linewidth=0)
        wave_ax.axvline(0, color=COLORS["rule"], lw=0.8)
        wave_ax.set_xlim(-150, 150)
        wave_ax.set_ylim(-3.6, 3.6)
        wave_ax.set_xticks([])
        wave_ax.set_yticks([])
        for spine in wave_ax.spines.values():
            spine.set_color(COLORS["rule"])
            spine.set_linewidth(0.8)

        spec_ax = fig.add_subplot(grid[2, column])
        image = spec_ax.imshow(
            spectrum,
            aspect="auto",
            origin="lower",
            extent=(-150, 150, 20, 300),
            cmap="viridis",
            vmin=spec_low,
            vmax=spec_high,
            interpolation="nearest",
        )
        spec_ax.axvspan(*event_span, facecolor="none", edgecolor=COLORS["ink"], alpha=0.50, lw=0.7)
        spec_ax.axvline(0, color=COLORS["ink"], alpha=0.40, lw=0.7)
        spec_ax.set_xlim(-150, 150)
        spec_ax.set_ylim(20, 300)
        spec_ax.set_xticks((-150, 0, 150))
        spec_ax.set_yticks((20, 160, 300) if column == 0 else ())
        spec_ax.tick_params(colors=COLORS["muted"], labelsize=7, length=2)
        spec_ax.set_xlabel("Time (ms)", color=COLORS["muted"], fontsize=7.5, labelpad=3)
        if column == 0:
            spec_ax.set_ylabel("Frequency (Hz)", color=COLORS["muted"], fontsize=7.5, labelpad=4)
        for spine in spec_ax.spines.values():
            spine.set_color(COLORS["rule"])
            spine.set_linewidth(0.8)

        note_ax = fig.add_subplot(grid[3, column])
        note_ax.axis("off")
        note = {
            "ripple": "Oscillatory burst across channels",
            "ied": "Sharper transient with broad energy",
            "noise": "Morphology lies between class centers",
        }[class_name]
        note_ax.text(0, 0.72, note, color=COLORS["secondary"], fontsize=8.3, va="top", wrap=True)

    fig.text(0.075, 0.045, "Waveforms normalized within event • Spectrograms share one normalized-power scale", color=COLORS["muted"], fontsize=8)
    if image is not None:
        colorbar_ax = fig.add_axes((0.735, 0.050, 0.24, 0.012))
        colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
        colorbar.set_ticks((spec_low, spec_high))
        colorbar.set_ticklabels(("lower", "higher"))
        colorbar.ax.tick_params(colors=COLORS["muted"], labelsize=7, length=0, pad=2)
        colorbar.outline.set_edgecolor(COLORS["rule"])
    # The website displays this at about 760 px wide; export at 2x density so
    # axes and the time-frequency texture stay crisp on high-DPI screens.
    export_figure(fig, SHOWCASE / "02_representative_events", web_width=1520, web_height=1300)
    plt.close(fig)
    print("Wrote representative-event SVG, PNG, and WebP exports")


if __name__ == "__main__":
    main()
