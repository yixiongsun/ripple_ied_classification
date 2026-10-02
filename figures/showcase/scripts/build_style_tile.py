"""Build the shared visual-system style tile for the portfolio showcase."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from showcase_style import COLORS, configure_matplotlib, export_figure


ROOT = Path(__file__).resolve().parents[1]

def rounded_box(ax, x, y, width, height, *, face, edge=None, radius=0.018, lw=1):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge or face,
        facecolor=face,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, *, size=12, weight="normal", color="ink", ha="left", va="center"):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        color=COLORS.get(color, color),
        ha=ha,
        va=va,
        fontfamily="sans-serif",
    )


def section_title(ax, x, y, number, title):
    label(ax, x, y, number, size=10, weight="bold", color="compact")
    label(ax, x + 0.034, y, title, size=15, weight="bold")


def draw_tile() -> plt.Figure:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.6, 9.2), dpi=100, facecolor=COLORS["page"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    label(ax, 0.055, 0.956, "RIPPLE / IED CLASSIFICATION", size=9, weight="bold", color="compact")
    label(ax, 0.055, 0.918, "Dark editorial system for the case study", size=23, weight="normal")
    label(
        ax,
        0.055,
        0.882,
        "Signal-first storytelling • subject-aware evidence • uncertainty shown, not hidden",
        size=10,
        color="secondary",
    )

    # Palette
    section_title(ax, 0.055, 0.822, "01", "Semantic palette")
    swatches = [
        ("Ripple", "ripple", "circle"),
        ("IED", "ied", "diamond"),
        ("Noise", "noise", "square"),
        ("Selected", "compact", "solid"),
        ("Baseline", "baseline", "dashed"),
        ("Body text", "limitation", "neutral"),
    ]
    start_x = 0.055
    swatch_y = 0.754
    gap = 0.149
    for i, (name, role, cue) in enumerate(swatches):
        x = start_x + i * gap
        rounded_box(ax, x, swatch_y, 0.126, 0.037, face=COLORS[role], radius=0.010, lw=0)
        label(ax, x, swatch_y - 0.016, name, size=8.5, weight="bold", va="top")
        label(ax, x, swatch_y - 0.037, cue, size=7.8, color="secondary", va="top")

    # Typography
    section_title(ax, 0.055, 0.662, "02", "Typography")
    rounded_box(ax, 0.055, 0.542, 0.89, 0.086, face=COLORS["surface"], edge=COLORS["rule"], radius=0.012)
    label(ax, 0.080, 0.596, "One finding per figure", size=18, weight="normal")
    label(ax, 0.480, 0.598, "Arial / Helvetica", size=10, weight="bold", color="compact")
    label(ax, 0.480, 0.568, "Sentence case • direct labels • tabular metrics 0.849", size=9, color="secondary")
    label(ax, 0.080, 0.558, "Minimum 12 px at the rendered width", size=8.5, color="secondary")

    # Figure grammar
    section_title(ax, 0.055, 0.490, "03", "Figure grammar")
    rounded_box(ax, 0.055, 0.382, 0.89, 0.074, face=COLORS["surface"], edge=COLORS["rule"], radius=0.012)
    ax.plot([0.082, 0.255], [0.424, 0.424], color=COLORS["compact"], lw=2.5, transform=ax.transAxes)
    ax.scatter([0.170], [0.424], s=34, color=COLORS["compact"], transform=ax.transAxes, zorder=3)
    label(ax, 0.280, 0.424, "Selected compact", size=9.5, weight="bold")
    ax.plot([0.520, 0.690], [0.424, 0.424], color=COLORS["baseline"], lw=2, dashes=(4, 3), transform=ax.transAxes)
    ax.scatter([0.605], [0.424], s=28, facecolors=COLORS["surface"], edgecolors=COLORS["baseline"], transform=ax.transAxes, zorder=3)
    label(ax, 0.715, 0.424, "Full baseline", size=9.5, color="secondary")
    label(ax, 0.082, 0.397, "Color + line style + direct label; no hover dependency", size=8.5, color="secondary")

    # Responsive composition
    section_title(ax, 0.055, 0.326, "04", "Responsive composition")
    label(ax, 0.055, 0.294, "Desktop • 760 px", size=9, weight="bold", color="secondary")
    desktop_x, desktop_y, desktop_w, desktop_h = 0.055, 0.116, 0.64, 0.150
    rounded_box(ax, desktop_x, desktop_y, desktop_w, desktop_h, face=COLORS["surface"], edge=COLORS["rule"], radius=0.012)
    stages = ["Recordings", "Detection", "Events", "Train", "Decision"]
    stage_colors = ["rule"] * len(stages)
    row_x = desktop_x + 0.024
    row_w = desktop_w - 0.048
    row_h = 0.020
    top_y = desktop_y + desktop_h - 0.032
    for i, (stage, role) in enumerate(zip(stages, stage_colors)):
        y = top_y - i * 0.025
        rounded_box(ax, row_x, y, row_w, row_h, face=COLORS["page"], edge=COLORS[role], radius=0.005)
        label(ax, row_x + 0.018, y + row_h / 2, f"{i + 1:02d}", size=6.6, color="secondary")
        label(ax, row_x + 0.070, y + row_h / 2, stage, size=7.1, weight="bold")
        if i < len(stages) - 1:
            ax.plot(
                [desktop_x + desktop_w / 2, desktop_x + desktop_w / 2],
                [y - 0.005, y],
                color=COLORS["secondary"],
                lw=1.0,
                transform=ax.transAxes,
            )

    # Mobile preview
    label(ax, 0.740, 0.294, "Mobile • 339 px", size=9, weight="bold", color="secondary")
    mobile_x, mobile_y, mobile_w, mobile_h = 0.740, 0.082, 0.205, 0.184
    rounded_box(ax, mobile_x, mobile_y, mobile_w, mobile_h, face=COLORS["surface"], edge=COLORS["rule"], radius=0.018)
    for i, (stage, role) in enumerate(zip(stages, stage_colors)):
        y = mobile_y + mobile_h - 0.032 - i * 0.030
        rounded_box(ax, mobile_x + 0.022, y, mobile_w - 0.044, 0.020, face=COLORS["page"], edge=COLORS[role], radius=0.006)
        label(ax, mobile_x + mobile_w / 2, y + 0.010, stage, size=6.7, weight="bold", ha="center")
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(mobile_x + mobile_w / 2, y - 0.007),
                xytext=(mobile_x + mobile_w / 2, y - 0.001),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color=COLORS["secondary"], lw=1),
            )
    label(ax, mobile_x + mobile_w / 2, mobile_y + 0.014, "No hover", size=7.2, color="secondary", ha="center")

    # Footer principle
    ax.plot([0.055, 0.945], [0.062, 0.062], color=COLORS["rule"], lw=1, transform=ax.transAxes)
    label(ax, 0.055, 0.032, "EDITORIAL TONE", size=8, weight="bold", color="compact")
    label(ax, 0.225, 0.032, "Precise, calm, candid; limitations remain visible.", size=9)

    return fig


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    fig = draw_tile()
    svg_path = ROOT / "00_style_tile.svg"
    png_path = ROOT / "00_style_tile.png"
    webp_path = ROOT / "00_style_tile.webp"

    export_figure(fig, ROOT / "00_style_tile", web_width=760, web_height=920)
    plt.close(fig)

    print(f"Wrote {svg_path.relative_to(ROOT.parent.parent)}")
    print(f"Wrote {png_path.relative_to(ROOT.parent.parent)}")
    print(f"Wrote {webp_path.relative_to(ROOT.parent.parent)}")


if __name__ == "__main__":
    main()
