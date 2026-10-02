"""Build desktop and mobile versions of the end-to-end pipeline figure."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from showcase_style import COLORS, configure_matplotlib, export_figure


OUTPUT_DIR = Path(__file__).resolve().parents[1]


def rounded_box(ax, x, y, width, height, *, face="surface", edge="rule", radius=0.012, lw=1.2, zorder=1):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=COLORS.get(face, face),
        edgecolor=COLORS.get(edge, edge),
        linewidth=lw,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, *, size=11, weight="normal", color="ink", ha="left", va="center", zorder=5):
    return ax.text(
        x,
        y,
        value,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        color=COLORS.get(color, color),
        ha=ha,
        va=va,
        fontfamily="sans-serif",
        zorder=zorder,
    )


def multiline(ax, x, y, lines, *, size=10, color="secondary", gap=0.031, bullet=True):
    for index, line in enumerate(lines):
        prefix = "•  " if bullet else ""
        text(ax, x, y - index * gap, prefix + line, size=size, color=color, va="top")


def arrow(ax, start, end, *, color="secondary", lw=1.5):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color=COLORS[color], lw=lw, shrinkA=1, shrinkB=1),
        zorder=3,
    )


def stage_header(ax, x, y, number, title, role, *, width=None):
    ax.add_patch(Circle((x, y), 0.017, transform=ax.transAxes, facecolor=COLORS[role], edgecolor="none", zorder=4))
    text(ax, x, y, str(number), size=9, weight="bold", color="surface", ha="center")
    text(ax, x + 0.027, y, title, size=13, weight="bold")
    if width:
        ax.plot([x - 0.017, x - 0.017 + width], [y - 0.035, y - 0.035], color=COLORS[role], lw=2, transform=ax.transAxes, zorder=2)


def waveform_icon(ax, x, y, width, height, *, color="ripple", channels=3):
    xs = np.linspace(0, 1, 120)
    for channel in range(channels):
        baseline = y + height * (0.18 + channel * 0.30)
        center = 0.55 + (channel - 1) * 0.025
        envelope = np.exp(-((xs - center) / 0.18) ** 2)
        signal = np.sin((26 + channel * 3) * np.pi * xs) * envelope
        ax.plot(x + xs * width, baseline + signal * height * 0.10, color=COLORS[color], lw=1.2, transform=ax.transAxes, zorder=4)


def envelope_icon(ax, x, y, width, height):
    xs = np.linspace(0, 1, 100)
    curve = 0.15 + 0.72 * np.exp(-((xs - 0.58) / 0.14) ** 2)
    ax.plot(x + xs * width, y + curve * height, color=COLORS["ripple"], lw=1.6, transform=ax.transAxes, zorder=4)
    ax.plot([x, x + width], [y + height * 0.43] * 2, color=COLORS["secondary"], lw=1, dashes=(3, 2), transform=ax.transAxes, zorder=3)
    ax.add_patch(Rectangle((x + width * 0.46, y), width * 0.23, height, transform=ax.transAxes, facecolor=COLORS["ripple"] + "18", edgecolor="none", zorder=2))


def event_icon(ax, x, y, width, height):
    waveform_icon(ax, x, y + height * 0.52, width, height * 0.42, color="baseline", channels=2)
    cols, rows = 18, 8
    for row in range(rows):
        for col in range(cols):
            strength = np.exp(-((col / (cols - 1) - 0.56) / 0.18) ** 2) * (0.3 + row / rows)
            base = np.array([232, 232, 227])
            target = np.array([80, 82, 79])
            rgb = (base * (1 - strength) + target * strength).astype(int)
            color = "#" + "".join(f"{component:02X}" for component in rgb)
            ax.add_patch(
                Rectangle(
                    (x + col * width / cols, y + row * height * 0.42 / rows),
                    width / cols,
                    height * 0.42 / rows,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="none",
                    zorder=2,
                )
            )


def class_pill(ax, x, y, width, label, role, marker, *, size=8.7):
    rounded_box(ax, x, y, width, 0.036, face=COLORS[role] + "18", edge=role, radius=0.018, lw=1)
    if marker == "circle":
        ax.scatter([x + 0.016], [y + 0.018], s=28, marker="o", color=COLORS[role], transform=ax.transAxes, zorder=5)
    elif marker == "diamond":
        ax.scatter([x + 0.016], [y + 0.018], s=30, marker="D", color=COLORS[role], transform=ax.transAxes, zorder=5)
    else:
        ax.scatter([x + 0.016], [y + 0.018], s=28, marker="s", color=COLORS[role], transform=ax.transAxes, zorder=5)
    text(ax, x + 0.032, y + 0.018, label, size=size, weight="bold")


def setup_figure(figsize):
    configure_matplotlib()
    fig = plt.figure(figsize=figsize, dpi=100, facecolor=COLORS["page"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def add_title(ax, *, mobile=False):
    if mobile:
        text(ax, 0.065, 0.966, "END-TO-END PIPELINE", size=9, weight="bold", color="compact")
        text(ax, 0.065, 0.932, "From recordings to", size=22, weight="bold")
        text(ax, 0.065, 0.902, "calibrated decisions", size=22, weight="bold")
        text(ax, 0.065, 0.878, "Subject-aware training separates model fitting,", size=10.5, color="secondary")
        text(ax, 0.065, 0.858, "threshold calibration, and held-out validation.", size=10.5, color="secondary")
    else:
        text(ax, 0.045, 0.938, "END-TO-END PIPELINE", size=10, weight="bold", color="compact")
        text(ax, 0.045, 0.892, "From multichannel recordings to calibrated decisions", size=25, weight="bold")
        text(ax, 0.045, 0.850, "Candidate events become waveform-based ripple or IED predictions, with low-confidence cases rejected as noise.", size=11.5, color="secondary")


def draw_training_split(ax, x, y, width, height, *, mobile=False):
    pad = 0.014 if not mobile else 0.020
    top = y + height
    text(ax, x + pad, top - pad * 1.2, "Outer subject-grouped fold", size=9.5 if not mobile else 10.5, weight="bold", color="compact", va="top")
    if mobile:
        box_h = (height - 0.075) / 3
        model_y = y + height - 0.052 - box_h
        calibration_y = model_y - 0.012 - box_h
        validation_y = calibration_y - 0.012 - box_h
        rounded_box(ax, x + pad, model_y, width - 2 * pad, box_h, face=COLORS["compact"] + "12", edge="compact", radius=0.009, lw=1)
        text(ax, x + 2 * pad, model_y + box_h * 0.70, "Model-fitting subjects", size=10.2, weight="bold")
        text(ax, x + 2 * pad, model_y + box_h * 0.20, "Learn model parameters", size=9.2, color="secondary")
        rounded_box(ax, x + pad, calibration_y, width - 2 * pad, box_h, face=COLORS["baseline"] + "12", edge="baseline", radius=0.009, lw=1)
        text(ax, x + 2 * pad, calibration_y + box_h * 0.70, "Calibration subjects", size=10.2, weight="bold")
        text(ax, x + 2 * pad, calibration_y + box_h * 0.20, "20% of training subjects • set threshold", size=9.0, color="secondary")
        rounded_box(ax, x + pad, validation_y, width - 2 * pad, box_h, face=COLORS["baseline"] + "18", edge="baseline", radius=0.009, lw=1)
        text(ax, x + 2 * pad, validation_y + box_h * 0.70, "Held-out validation subjects", size=10.2, weight="bold")
        text(ax, x + 2 * pad, validation_y + box_h * 0.20, "Evaluate only", size=9.2, color="secondary")
    else:
        inner_y = y + 0.040
        inner_h = height - 0.098
        left_w = width * 0.58
        right_x = x + pad + left_w + 0.012
        right_w = width - (right_x - x) - pad
        rounded_box(ax, x + pad, inner_y, left_w, inner_h, face=COLORS["compact"] + "0D", edge="compact", radius=0.009, lw=1)
        text(ax, x + pad * 1.7, inner_y + inner_h - 0.027, "Outer training subjects", size=9.3, weight="bold", va="top")
        model_x = x + pad * 1.7
        sub_w = left_w - pad * 1.4
        sub_h = (inner_h - 0.092) / 2
        model_y = inner_y + 0.036 + sub_h + 0.010
        cal_y = inner_y + 0.032
        rounded_box(ax, model_x, model_y, sub_w, sub_h, face="surface", edge="compact", radius=0.008, lw=1)
        text(ax, model_x + 0.010, model_y + sub_h * 0.64, "Model-fitting subjects", size=8.5, weight="bold")
        text(ax, model_x + 0.010, model_y + sub_h * 0.27, "learn model parameters", size=7.7, color="secondary")
        rounded_box(ax, model_x, cal_y, sub_w, sub_h, face=COLORS["ied"] + "12", edge="ied", radius=0.008, lw=1)
        text(ax, model_x + 0.010, cal_y + sub_h * 0.64, "Calibration subjects", size=8.5, weight="bold")
        text(ax, model_x + 0.010, cal_y + sub_h * 0.27, "20% of training subjects • set threshold", size=7.2, color="secondary")
        rounded_box(ax, right_x, inner_y, right_w, inner_h, face=COLORS["baseline"] + "18", edge="baseline", radius=0.009, lw=1)
        right_center = right_x + right_w / 2
        text(ax, right_center, inner_y + inner_h * 0.68, "Held-out", size=8.6, weight="bold", ha="center")
        text(ax, right_center, inner_y + inner_h * 0.45, "validation subjects", size=8.4, weight="bold", ha="center")
        text(ax, right_center, inner_y + inner_h * 0.19, "evaluate only", size=7.8, color="secondary", ha="center")


def draw_decision_logic(ax, x, y, width, height, *, mobile=False):
    pad = 0.014 if not mobile else 0.022
    center = x + width / 2
    if mobile:
        text(ax, center, y + height - 0.016, "Binary ripple ↔ IED score", size=10.2, weight="bold", ha="center")
        rounded_box(ax, x + pad, y + 0.052, width - 2 * pad, 0.030, face=COLORS["compact"] + "10", edge="compact", radius=0.008, lw=1)
        text(ax, center, y + 0.067, "Below threshold → noise; otherwise use score sign", size=9.0, ha="center")
        split_y = y + 0.004
        pill_w = (width - 2 * pad - 0.020) / 3
        class_pill(ax, x + pad, split_y, pill_w, "Noise", "noise", "square", size=9.5)
        class_pill(ax, x + pad + pill_w + 0.010, split_y, pill_w, "Ripple", "ripple", "circle", size=9.5)
        class_pill(ax, x + pad + 2 * (pill_w + 0.010), split_y, pill_w, "IED", "ied", "diamond", size=9.5)
        return

    text(ax, center, y + height - 0.052, "Binary ripple ↔ IED score", size=9.2, weight="bold", ha="center")
    rounded_box(ax, x + pad, y + height - 0.118, width - 2 * pad, 0.046, face=COLORS["compact"] + "10", edge="compact", radius=0.008, lw=1)
    text(ax, center, y + height - 0.090, "Is |score| below", size=8.0, ha="center")
    text(ax, center, y + height - 0.106, "the calibrated threshold?", size=8.0, ha="center")
    pill_x = x + pad
    pill_w = width - 2 * pad
    positions = [
        (y + 0.116, "Noise", "noise", "square", "yes → reject"),
        (y + 0.068, "Ripple", "ripple", "circle", "no • negative"),
        (y + 0.020, "IED", "ied", "diamond", "no • positive"),
    ]
    for pill_y, label_value, role, marker, qualifier in positions:
        class_pill(ax, pill_x, pill_y, pill_w, label_value, role, marker)
        text(ax, pill_x + pill_w - 0.008, pill_y + 0.018, qualifier, size=6.8, color=role, ha="right")


def build_desktop():
    fig, ax = setup_figure((7.6, 15.5))
    text(ax, 0.055, 0.974, "END-TO-END PIPELINE", size=8.5, weight="bold", color="muted")
    text(ax, 0.055, 0.946, "From recordings to calibrated decisions", size=22, weight="normal")
    text(ax, 0.055, 0.918, "Five stages, one subject-aware path from signal detection to", size=9.2, color="secondary")
    text(ax, 0.055, 0.902, "ripple, IED, or low-confidence noise rejection.", size=9.2, color="secondary")

    card_x, card_w = 0.055, 0.890
    stages = [
        (0.748, 0.115, 1, "Recordings"),
        (0.596, 0.115, 2, "Detect candidates"),
        (0.423, 0.135, 3, "Represent + label"),
        (0.205, 0.178, 4, "Train + calibrate"),
        (0.054, 0.112, 5, "Predict"),
    ]
    for y, height, number, title_value in stages:
        rounded_box(ax, card_x, y, card_w, height, face="surface", edge="rule", radius=0.008, lw=1)
        ax.plot([card_x, card_x + card_w], [y + height, y + height], color=COLORS["baseline"], lw=1.2, transform=ax.transAxes, zorder=3)
        ax.add_patch(Circle((card_x + 0.035, y + height - 0.031), 0.015, transform=ax.transAxes, facecolor=COLORS["surface"], edgecolor=COLORS["baseline"], linewidth=1.2, zorder=4))
        text(ax, card_x + 0.035, y + height - 0.031, str(number), size=8, weight="bold", ha="center")
        text(ax, card_x + 0.063, y + height - 0.031, title_value, size=12, weight="bold")

    # 1 — Recordings
    y, height, *_ = stages[0]
    waveform_icon(ax, card_x + 0.090, y + 0.024, 0.280, 0.040, color="secondary", channels=3)
    text(ax, card_x + 0.440, y + 0.059, "Cleaned hippocampal LFP", size=9.0)
    text(ax, card_x + 0.440, y + 0.032, "Multiple channels • sleep-stage context", size=8.4, color="secondary")

    # 2 — Detection, with class colors used only for the two detectors.
    y, height, *_ = stages[1]
    xs = np.linspace(0, 1, 100)
    ripple_curve = 0.10 + 0.70 * np.exp(-((xs - 0.42) / 0.13) ** 2)
    ied_curve = 0.10 + 0.62 * np.exp(-((xs - 0.64) / 0.16) ** 2)
    icon_x, icon_y, icon_w, icon_h = card_x + 0.090, y + 0.025, 0.280, 0.042
    ax.plot(icon_x + xs * icon_w, icon_y + ripple_curve * icon_h, color=COLORS["ripple"], lw=1.6, transform=ax.transAxes)
    ax.plot(icon_x + xs * icon_w, icon_y + ied_curve * icon_h, color=COLORS["ied"], lw=1.6, transform=ax.transAxes)
    text(ax, card_x + 0.440, y + 0.061, "Ripple and IED envelopes", size=9.0)
    text(ax, card_x + 0.440, y + 0.034, "Channel consensus • thresholded event windows", size=8.4, color="secondary")

    # 3 — Event representation and labeling.
    y, height, *_ = stages[2]
    event_icon(ax, card_x + 0.090, y + 0.026, 0.280, 0.058)
    text(ax, card_x + 0.440, y + 0.077, "100 ms waveform → 128 values", size=8.9)
    text(ax, card_x + 0.440, y + 0.050, "Background-corrected spectrum • manual label", size=8.3, color="secondary")
    text(ax, card_x + 0.440, y + 0.024, "Spectrum explains development; compact model omits it", size=8.0, color="muted")

    # 4 — Nested subject split, drawn as neutral rows.
    y, height, *_ = stages[3]
    text(ax, card_x + 0.090, y + 0.117, "Outer subject-grouped fold", size=9.0, weight="bold")
    row_left, row_right = card_x + 0.090, card_x + card_w - 0.060
    split_rows = [
        (y + 0.087, "Model fitting", "Learn model parameters"),
        (y + 0.057, "Calibration", "20% of training subjects • set threshold"),
        (y + 0.027, "Held-out validation", "Evaluate only"),
    ]
    for row_y, title_value, detail in split_rows:
        ax.plot([row_left, row_right], [row_y + 0.012, row_y + 0.012], color=COLORS["rule"], lw=1, transform=ax.transAxes)
        text(ax, row_left, row_y, title_value, size=8.3, weight="bold")
        text(ax, row_right, row_y, detail, size=8.0, color="secondary", ha="right")

    # 5 — Decision logic. Only the class outputs carry accent color.
    y, height, *_ = stages[4]
    text(ax, card_x + 0.090, y + 0.055, "Binary ripple ↔ IED score", size=8.8, weight="bold")
    text(ax, card_x + 0.090, y + 0.028, "Low |score| → noise; otherwise use score sign", size=8.2, color="secondary")
    pill_x = card_x + 0.535
    pill_y = y + 0.024
    pill_gap = 0.012
    pill_w = 0.095
    class_pill(ax, pill_x, pill_y, pill_w, "Noise", "noise", "square", size=7.6)
    class_pill(ax, pill_x + pill_w + pill_gap, pill_y, pill_w, "Ripple", "ripple", "circle", size=7.6)
    class_pill(ax, pill_x + 2 * (pill_w + pill_gap), pill_y, pill_w, "IED", "ied", "diamond", size=7.6)

    # Straight connectors in dedicated gutters.
    for index in range(len(stages) - 1):
        current_y = stages[index][0]
        next_top = stages[index + 1][0] + stages[index + 1][1]
        arrow(ax, (0.5, current_y - 0.006), (0.5, next_top + 0.006), color="baseline", lw=1.1)

    rounded_box(ax, 0.055, 0.010, 0.890, 0.030, face=COLORS["panel"], edge="rule", radius=0.006, lw=1)
    text(ax, 0.075, 0.025, "No train–validation subject overlap • validation subjects never set the threshold", size=8.0, color="secondary")
    return fig


def mobile_stage_card(ax, x, y, width, height, number, title, role):
    rounded_box(ax, x, y, width, height, face="surface", edge="rule", radius=0.016, lw=1.2)
    ax.plot([x, x + width], [y + height, y + height], color=COLORS[role], lw=3, transform=ax.transAxes, zorder=3)
    stage_header(ax, x + 0.035, y + height - 0.040, number, title, role)


def build_mobile():
    fig, ax = setup_figure((3.39, 12.5))
    text(ax, 0.060, 0.975, "END-TO-END PIPELINE", size=8.5, weight="bold", color="compact")
    text(ax, 0.060, 0.947, "From recordings to", size=18.5, weight="normal")
    text(ax, 0.060, 0.922, "calibrated decisions", size=18.5, weight="normal")
    text(ax, 0.060, 0.891, "Model fitting, calibration, and validation", size=8.7, color="secondary")
    text(ax, 0.060, 0.875, "remain subject-separated.", size=8.7, color="secondary")

    x, width = 0.060, 0.880
    stages = [
        (0.742, 0.100, 1, "Recordings", "compact"),
        (0.612, 0.100, 2, "Detect candidates", "compact"),
        (0.458, 0.124, 3, "Represent + label", "compact"),
        (0.232, 0.196, 4, "Train + calibrate", "compact"),
        (0.065, 0.137, 5, "Predict", "compact"),
    ]
    for y, height, number, title_value, role in stages:
        mobile_stage_card(ax, x, y, width, height, number, title_value, role)

    waveform_icon(ax, x + 0.040, 0.755, 0.255, 0.038, color="baseline", channels=2)
    text(ax, x + 0.335, 0.785, "Cleaned hippocampal LFP", size=8.8)
    text(ax, x + 0.335, 0.762, "Channels + sleep stages", size=8.2, color="secondary")

    envelope_icon(ax, x + 0.040, 0.625, 0.255, 0.038)
    text(ax, x + 0.335, 0.655, "Ripple + IED envelopes", size=8.8)
    text(ax, x + 0.335, 0.632, "Consensus + event windows", size=8.2, color="secondary")

    event_icon(ax, x + 0.040, 0.479, 0.255, 0.052)
    text(ax, x + 0.335, 0.530, "100 ms waveform → 128", size=8.5)
    text(ax, x + 0.335, 0.507, "Spectrum + manual label", size=8.1, color="secondary")
    text(ax, x + 0.335, 0.482, "Compact model omits spectrum", size=7.8, weight="bold", color="secondary")

    text(ax, x + 0.050, 0.363, "Outer subject-grouped fold", size=8.6, weight="bold", color="compact")
    row_x, row_w = x + 0.050, width - 0.100
    rows = [
        (0.326, "Model fitting", "Learn parameters", "compact"),
        (0.283, "Calibration", "20% • set threshold", "baseline"),
        (0.240, "Held-out validation", "Evaluate only", "baseline"),
    ]
    for row_y, title_value, detail, role in rows:
        ax.plot([row_x, row_x + row_w], [row_y + 0.028, row_y + 0.028], color=COLORS[role], lw=1.8, transform=ax.transAxes)
        text(ax, row_x, row_y + 0.012, title_value, size=8.2, weight="bold")
        text(ax, row_x + row_w, row_y + 0.012, detail, size=7.8, color="secondary", ha="right")

    text(ax, 0.5, 0.143, "Binary ripple ↔ IED score", size=8.6, weight="bold", ha="center")
    text(ax, 0.5, 0.125, "Low |score| → noise; otherwise use sign", size=7.5, color="secondary", ha="center")
    pill_x, pill_y = x + 0.050, 0.070
    pill_gap = 0.015
    pill_w = (width - 0.100 - 2 * pill_gap) / 3
    class_pill(ax, pill_x, pill_y, pill_w, "Noise", "noise", "square", size=7.4)
    class_pill(ax, pill_x + pill_w + pill_gap, pill_y, pill_w, "Ripple", "ripple", "circle", size=7.4)
    class_pill(ax, pill_x + 2 * (pill_w + pill_gap), pill_y, pill_w, "IED", "ied", "diamond", size=7.4)

    for index in range(len(stages) - 1):
        y_current = stages[index][0]
        y_next = stages[index + 1][0] + stages[index + 1][1]
        arrow(ax, (0.5, y_current - 0.006), (0.5, y_next + 0.006), lw=1.1)

    rounded_box(ax, x, 0.015, width, 0.038, face=COLORS["compact"] + "12", edge="compact", radius=0.008, lw=1)
    text(ax, 0.5, 0.039, "No train–validation subject overlap", size=8.0, weight="bold", color="compact", ha="center")
    text(ax, 0.5, 0.024, "Validation subjects never set the threshold", size=8.0, weight="bold", color="compact", ha="center")
    return fig


def main() -> None:
    desktop = build_desktop()
    export_figure(desktop, OUTPUT_DIR / "01_pipeline", web_width=760, web_height=1550)
    plt.close(desktop)

    mobile = build_mobile()
    export_figure(mobile, OUTPUT_DIR / "01_pipeline_mobile", web_width=678, web_height=2500)
    plt.close(mobile)

    for stem in ("01_pipeline", "01_pipeline_mobile"):
        for suffix in ("svg", "png", "webp"):
            print(f"Wrote {(OUTPUT_DIR / stem).with_suffix('.' + suffix).relative_to(OUTPUT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
