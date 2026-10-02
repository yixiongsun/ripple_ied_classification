"""Shared visual tokens and export helpers for showcase figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure
from PIL import Image


COLORS = {
    "page": "#171817",
    "surface": "#111210",
    "panel": "#1D1F1D",
    "ink": "#E8E8E3",
    "secondary": "#B6B7B1",
    "muted": "#92938E",
    "rule": "#373936",
    "subtle_rule": "#2C2E2B",
    "ripple": "#6FB1E3",
    "ied": "#E8A15A",
    "noise": "#A89CC2",
    "compact": "#E8E8E3",
    "baseline": "#92938E",
    "limitation": "#B6B7B1",
}

FONT_STACK = ["Arial", "Helvetica", "DejaVu Sans"]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "svg.fonttype": "none",
            "svg.hashsalt": "ripple-ied-showcase-v1",
            "axes.unicode_minus": False,
        }
    )


def export_figure(
    figure: Figure,
    output_base: Path,
    *,
    web_width: int,
    web_height: int,
    png_dpi: int = 200,
) -> None:
    """Write editable SVG, high-resolution PNG, and optimized WebP."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_base.with_suffix(".svg"),
        format="svg",
        facecolor=COLORS["page"],
        bbox_inches=None,
        metadata={
            "Date": None,
            "Creator": "ripple_ied_classification showcase scripts",
        },
    )
    figure.savefig(
        output_base.with_suffix(".png"),
        format="png",
        dpi=png_dpi,
        facecolor=COLORS["page"],
        bbox_inches=None,
        metadata={"Software": "ripple_ied_classification showcase scripts"},
    )
    with Image.open(output_base.with_suffix(".png")) as image:
        image.convert("RGB").resize(
            (web_width, web_height), Image.Resampling.LANCZOS
        ).save(
            output_base.with_suffix(".webp"),
            format="WEBP",
            quality=88,
            method=6,
        )

