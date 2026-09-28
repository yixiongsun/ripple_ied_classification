"""Detect ripple/IED candidates and save the labeling JSON/JPG pairs.

This is the Python replacement for ``event_classification.ipynb``.  It works
with NumPy arrays and output paths only; loading project-specific subjects is
handled by :mod:`dataset_pipeline.subject_data`.

The detection thresholds and feature construction below follow the notebook's
final export cells.  Shared numerical and plotting details live in
``_event_extraction.py`` so ripple and IED candidates use the same processing
path.
"""

from __future__ import annotations

import numpy as np

from ._event_extraction import find_events, process_events


def detect_ripples(
    consensus_envelope: np.ndarray,
    sample_rate: float = 2_000,
) -> np.ndarray:
    """Return ripple ``[start, end)`` windows using the notebook settings."""
    return find_events(
        consensus_envelope,
        sample_rate,
        z_entry=3,
        z_exit=2,
        peak_threshold=7,
        min_duration=0.02,
        max_duration=0.25,
        merge_gap=0.015,
    )


def detect_ieds(
    consensus_envelope: np.ndarray,
    sample_rate: float = 2_000,
) -> np.ndarray:
    """Return IED ``[start, end)`` windows using the notebook settings."""
    return find_events(
        consensus_envelope,
        sample_rate,
        z_entry=5,
        z_exit=5,
        peak_threshold=5,
        min_duration=0.02,
        max_duration=0.25,
        merge_gap=0,
    )


def export_ripples(
    lfps,
    consensus_envelope,
    ripple_channel_envelopes,
    medians,
    iqrs,
    baseline_spectra,
    output_directory,
    *,
    sample_rate=2_000,
    workers=4,
    top_k=None,
    overwrite=False,
    show_progress=True,
):
    """Detect ripple candidates and write their labeling JPG/JSON pairs.

    ``ripple_channel_envelopes`` is used both to rank the three channels saved
    for each candidate and to populate the JSON ``ripple_env`` field.
    """
    events = detect_ripples(np.asarray(consensus_envelope), sample_rate)
    ripple_channel_envelopes = np.asarray(ripple_channel_envelopes)
    return process_events(
        events,
        event_name="ripple",
        detection_envelope=np.asarray(consensus_envelope),
        scoring_envelopes=ripple_channel_envelopes,
        context_envelopes=ripple_channel_envelopes,
        lfps=lfps,
        medians=np.asarray(medians),
        iqrs=np.asarray(iqrs),
        baseline_spectra=np.asarray(baseline_spectra),
        output_directory=output_directory,
        sample_rate=sample_rate,
        workers=workers,
        top_k=top_k,
        overwrite=overwrite,
        show_progress=show_progress,
    )


def export_ieds(
    lfps,
    consensus_envelope,
    ied_channel_envelopes,
    ripple_channel_envelopes,
    medians,
    iqrs,
    baseline_spectra,
    output_directory,
    *,
    sample_rate=2_000,
    workers=4,
    top_k=None,
    overwrite=False,
    show_progress=True,
):
    """Detect IED candidates and write their labeling JPG/JSON pairs.

    IED envelopes rank the three saved channels.  Ripple envelopes populate
    ``ripple_env`` to retain the notebook's ripple-band context for each IED.
    """
    events = detect_ieds(np.asarray(consensus_envelope), sample_rate)
    return process_events(
        events,
        event_name="ied",
        detection_envelope=np.asarray(consensus_envelope),
        scoring_envelopes=np.asarray(ied_channel_envelopes),
        context_envelopes=np.asarray(ripple_channel_envelopes),
        lfps=lfps,
        medians=np.asarray(medians),
        iqrs=np.asarray(iqrs),
        baseline_spectra=np.asarray(baseline_spectra),
        output_directory=output_directory,
        sample_rate=sample_rate,
        workers=workers,
        top_k=top_k,
        overwrite=overwrite,
        show_progress=show_progress,
    )
