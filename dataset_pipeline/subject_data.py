"""Connect the reusable envelope/detection functions to this project's subjects."""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .baseline_spectrograms import (
    InsufficientBaselineSamplesError,
    background_mask,
    compute_baseline_spectrogram,
)
from .event_classification import detect_ieds, detect_ripples, export_ieds, export_ripples
from .ied_envelope import consensus_ied_envelope, ied_envelope
from .kmeans_selection import select_session_candidates, write_label_csv
from .lfp_statistics import compute_lfp_statistics
from .ripple_envelope import consensus_ripple_envelope, ripple_envelope

try:
    import subjects
except ModuleNotFoundError:
    subjects = None


DEFAULT_SESSIONS = ("Sleep1", "Sleep2")
SAMPLE_RATE = 2_000


def get_subject_data(subject_id):
    """Return this project's base directory and hippocampal LFP channels."""
    if subjects is None:
        raise ModuleNotFoundError(
            "subject_data.py requires the project-specific subjects.py module"
        )
    subject = subjects.load(subject_id)
    return Path(subject["olm"]["directory"]), list(subject["olm"]["lfp_channels"])


def _cleaned_lfp(session_directory, channel):
    return np.load(session_directory / f"{channel}_cleaned_hpc.npy", mmap_mode="r")[1]


def _sleep_stages(session_directory):
    return np.load(session_directory / "sleep_stages.npy").reshape(-1)


def generate_lfp_statistics(
    subject_id,
    *,
    sessions=DEFAULT_SESSIONS,
    overwrite=False,
    show_progress=True,
):
    """Create each channel's stage-2 waveform-normalization statistics."""
    base_directory, channels = get_subject_data(subject_id)
    session_directories = [base_directory / session for session in sessions]

    channel_iterator = tqdm(
        channels,
        desc=f"{subject_id} LFP statistics",
        unit="channel",
        leave=False,
        disable=not show_progress,
    )
    for channel in channel_iterator:
        missing_stage_2 = []
        for session, directory in zip(sessions, session_directories):
            output = directory / f"{channel}_lfp_stats.npy"
            if output.exists() and not overwrite:
                continue

            lfp = _cleaned_lfp(directory, channel)
            stages = _sleep_stages(directory).reshape(-1)
            if len(lfp) != len(stages):
                raise ValueError(
                    f"LFP and sleep stages have different lengths for "
                    f"{subject_id} {session}"
                )
            if not np.any(stages == 2):
                missing_stage_2.append((session, output))
                continue

            np.save(output, compute_lfp_statistics(lfp, stages))

        for session, output in missing_stage_2:
            source = next(
                (
                    directory / f"{channel}_lfp_stats.npy"
                    for directory in session_directories
                    if directory / f"{channel}_lfp_stats.npy" != output
                    and (directory / f"{channel}_lfp_stats.npy").exists()
                ),
                None,
            )
            if source is None:
                raise ValueError(
                    f"No stage-2 samples are available to compute statistics "
                    f"for {subject_id} {channel} ({session})"
                )
            np.save(output, np.load(source, allow_pickle=True).item())


def _write_channel_envelopes(
    subject_id,
    envelope_function,
    output_suffix,
    consensus_function,
    consensus_filename,
    *,
    sessions=DEFAULT_SESSIONS,
    overwrite=False,
    consensus_kwargs=None,
    show_progress=True,
):
    base_directory, channels = get_subject_data(subject_id)
    session_directories = [base_directory / session for session in sessions]

    channel_iterator = tqdm(
        channels,
        desc=f"{subject_id} {output_suffix}",
        unit="channel",
        leave=False,
        disable=not show_progress,
    )
    for channel in channel_iterator:
        output_paths = [directory / f"{channel}_{output_suffix}.npy" for directory in session_directories]
        if not overwrite and all(path.exists() for path in output_paths):
            continue

        lfps = [_cleaned_lfp(directory, channel) for directory in session_directories]
        stages = [_sleep_stages(directory) for directory in session_directories]
        for lfp, stage, session in zip(lfps, stages, sessions):
            if len(lfp) != len(stage):
                raise ValueError(f"LFP and sleep stages have different lengths for {subject_id} {session}")

        lengths = [len(lfp) for lfp in lfps]
        combined_lfp = np.concatenate(lfps)
        baseline_mask = np.concatenate(stages) == 2
        combined_envelope = envelope_function(combined_lfp, baseline_mask, SAMPLE_RATE)

        start = 0
        for path, length in zip(output_paths, lengths):
            np.save(path, combined_envelope[start : start + length])
            start += length

    consensus_kwargs = consensus_kwargs or {}
    session_iterator = tqdm(
        session_directories,
        desc=f"{subject_id} consensus",
        unit="session",
        leave=False,
        disable=not show_progress,
    )
    for directory in session_iterator:
        output_path = directory / consensus_filename
        if output_path.exists() and not overwrite:
            continue
        channel_envelopes = np.stack(
            [np.load(directory / f"{channel}_{output_suffix}.npy") for channel in channels]
        )
        consensus = consensus_function(
            channel_envelopes,
            sample_rate=SAMPLE_RATE,
            **consensus_kwargs,
        )
        np.save(output_path, consensus)


def generate_ripple_envelopes(
    subject_id,
    *,
    sessions=DEFAULT_SESSIONS,
    channel_top_k=3,
    overwrite=False,
    show_progress=True,
):
    """Create per-channel and consensus ripple envelopes for one subject."""
    _write_channel_envelopes(
        subject_id,
        ripple_envelope,
        "ripple_zenv",
        consensus_ripple_envelope,
        "consensus_ripple_env.npy",
        sessions=sessions,
        overwrite=overwrite,
        consensus_kwargs={"top_k": channel_top_k},
        show_progress=show_progress,
    )


def generate_ied_envelopes(
    subject_id,
    *,
    sessions=DEFAULT_SESSIONS,
    overwrite=False,
    show_progress=True,
):
    """Create per-channel and consensus IED envelopes for one subject."""
    _write_channel_envelopes(
        subject_id,
        ied_envelope,
        "ied_zenv",
        consensus_ied_envelope,
        "consensus_ied_env.npy",
        sessions=sessions,
        overwrite=overwrite,
        show_progress=show_progress,
    )


def generate_baseline_spectrograms(
    subject_id,
    *,
    sessions=DEFAULT_SESSIONS,
    overwrite=False,
    show_progress=True,
):
    """Create each channel's event-free stage-2 baseline spectrogram."""
    base_directory, channels = get_subject_data(subject_id)
    sessions = tuple(sessions)
    session_directories = [base_directory / session for session in sessions]
    missing_baselines = []
    session_iterator = tqdm(
        zip(sessions, session_directories),
        total=len(sessions),
        desc=f"{subject_id} baseline spectrograms",
        unit="session",
        leave=False,
        disable=not show_progress,
    )
    for session, directory in session_iterator:
        outputs = [directory / f"{channel}_baseline_spec.npy" for channel in channels]
        if not overwrite and all(path.exists() for path in outputs):
            continue

        ripple_events = detect_ripples(
            np.load(directory / "consensus_ripple_env.npy"), SAMPLE_RATE
        )
        ied_events = detect_ieds(
            np.load(directory / "consensus_ied_env.npy"), SAMPLE_RATE
        )
        event_windows = np.concatenate((ripple_events, ied_events), axis=0)
        stages = _sleep_stages(directory)

        for channel, output in zip(channels, outputs):
            if output.exists() and not overwrite:
                continue
            lfp = _cleaned_lfp(directory, channel)
            sample_count = min(len(lfp), len(stages))
            mask = background_mask(stages[:sample_count], event_windows)
            try:
                baseline = compute_baseline_spectrogram(
                    lfp[:sample_count],
                    mask,
                    SAMPLE_RATE,
                    show_progress=show_progress,
                )
            except InsufficientBaselineSamplesError:
                missing_baselines.append((session, channel, output))
                continue
            np.save(output, baseline)

    missing_outputs = {output for _, _, output in missing_baselines}
    for session, channel, output in missing_baselines:
        source = next(
            (
                directory / f"{channel}_baseline_spec.npy"
                for directory in session_directories
                if directory / f"{channel}_baseline_spec.npy" != output
                and directory / f"{channel}_baseline_spec.npy" not in missing_outputs
                and (directory / f"{channel}_baseline_spec.npy").exists()
            ),
            None,
        )
        if source is None:
            raise InsufficientBaselineSamplesError(
                f"Neither requested session has enough baseline samples for "
                f"{subject_id} {channel}; unable to fill {session}"
            )
        np.save(output, np.load(source))
        tqdm.write(f"Copied baseline spectrogram: {source} -> {output}")


def _load_detection_data(subject_id, session):
    base_directory, channels = get_subject_data(subject_id)
    directory = base_directory / session
    lfps = [_cleaned_lfp(directory, channel) for channel in channels]
    stats = [
        np.load(directory / f"{channel}_lfp_stats.npy", allow_pickle=True).item()
        for channel in channels
    ]
    medians = np.asarray([item["median"] for item in stats])
    iqrs = np.asarray([item["iqr"] for item in stats])
    baseline_spectra = np.stack(
        [np.load(directory / f"{channel}_baseline_spec.npy") for channel in channels]
    )
    ripple_envelopes = np.stack(
        [np.load(directory / f"{channel}_ripple_zenv.npy") for channel in channels]
    )
    return directory, channels, lfps, medians, iqrs, baseline_spectra, ripple_envelopes


def extract_subject_ripples(
    subject_id,
    session,
    *,
    workers=4,
    event_top_k=None,
    overwrite=False,
    show_progress=True,
):
    """Load one subject/session and export its ripple candidates."""
    directory, _, lfps, medians, iqrs, baselines, ripple_envelopes = _load_detection_data(
        subject_id, session
    )
    consensus = np.load(directory / "consensus_ripple_env.npy")
    return export_ripples(
        lfps,
        consensus,
        ripple_envelopes,
        medians,
        iqrs,
        baselines,
        directory / "ripples",
        sample_rate=SAMPLE_RATE,
        workers=workers,
        top_k=event_top_k,
        overwrite=overwrite,
        show_progress=show_progress,
    )


def extract_subject_ieds(
    subject_id,
    session,
    *,
    workers=4,
    event_top_k=None,
    overwrite=False,
    show_progress=True,
):
    """Load one subject/session and export its IED candidates."""
    directory, channels, lfps, medians, iqrs, baselines, ripple_envelopes = _load_detection_data(
        subject_id, session
    )
    ied_envelopes = np.stack(
        [np.load(directory / f"{channel}_ied_zenv.npy") for channel in channels]
    )
    consensus = np.load(directory / "consensus_ied_env.npy")
    return export_ieds(
        lfps,
        consensus,
        ied_envelopes,
        ripple_envelopes,
        medians,
        iqrs,
        baselines,
        directory / "ieds",
        sample_rate=SAMPLE_RATE,
        workers=workers,
        top_k=event_top_k,
        overwrite=overwrite,
        show_progress=show_progress,
    )


def create_label_selection(
    subject_ids,
    *,
    sessions=DEFAULT_SESSIONS,
    output_path="extracted_samples.csv",
    count_per_event_type=100,
    random_state=0,
    show_progress=True,
):
    """Select representative ripple/IED images and write an unlabeled CSV."""
    rows = []
    subject_iterator = tqdm(
        subject_ids,
        desc="label candidates",
        unit="subject",
        disable=not show_progress,
    )
    for subject_id in subject_iterator:
        base_directory, _ = get_subject_data(subject_id)
        for session in sessions:
            rows.extend(
                select_session_candidates(
                    base_directory / session,
                    subject_id,
                    session,
                    count_per_event_type=count_per_event_type,
                    random_state=random_state,
                    show_progress=show_progress,
                )
            )
    write_label_csv(rows, output_path)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run subject-specific dataset pipeline steps.")
    parser.add_argument(
        "action",
        choices=(
            "ripple-envelope",
            "ied-envelope",
            "lfp-statistics",
            "baseline-spectrogram",
            "ripple-events",
            "ied-events",
            "label-candidates",
        ),
    )
    parser.add_argument("--subjects", nargs="+", dest="subject_ids")
    parser.add_argument("--sessions", nargs="+", default=list(DEFAULT_SESSIONS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--channel-top-k", type=int, default=3)
    parser.add_argument("--event-top-k", type=int)
    parser.add_argument("--label-count", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--output-csv", default="extracted_samples.csv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if subjects is None:
        raise ModuleNotFoundError("The project-specific subjects.py module is missing")
    subject_ids = args.subject_ids or subjects.get_subjects()

    if args.action == "label-candidates":
        rows = create_label_selection(
            subject_ids,
            sessions=args.sessions,
            output_path=args.output_csv,
            count_per_event_type=args.label_count,
            random_state=args.random_state,
            show_progress=not args.no_progress,
        )
        print(f"Wrote {len(rows)} candidates to {args.output_csv}")
        return

    subject_iterator = tqdm(
        subject_ids,
        desc=args.action,
        unit="subject",
        disable=args.no_progress,
    )
    for subject_id in subject_iterator:
        if args.action == "ripple-envelope":
            generate_ripple_envelopes(
                subject_id,
                sessions=args.sessions,
                channel_top_k=args.channel_top_k,
                overwrite=args.overwrite,
                show_progress=not args.no_progress,
            )
        elif args.action == "ied-envelope":
            generate_ied_envelopes(
                subject_id,
                sessions=args.sessions,
                overwrite=args.overwrite,
                show_progress=not args.no_progress,
            )
        elif args.action == "lfp-statistics":
            generate_lfp_statistics(
                subject_id,
                sessions=args.sessions,
                overwrite=args.overwrite,
                show_progress=not args.no_progress,
            )
        elif args.action == "baseline-spectrogram":
            generate_baseline_spectrograms(
                subject_id,
                sessions=args.sessions,
                overwrite=args.overwrite,
                show_progress=not args.no_progress,
            )
        else:
            extractor = extract_subject_ripples if args.action == "ripple-events" else extract_subject_ieds
            session_iterator = tqdm(
                args.sessions,
                desc=subject_id,
                unit="session",
                leave=False,
                disable=args.no_progress,
            )
            for session in session_iterator:
                counts = extractor(
                    subject_id,
                    session,
                    workers=args.workers,
                    event_top_k=args.event_top_k,
                    overwrite=args.overwrite,
                    show_progress=not args.no_progress,
                )
                tqdm.write(f"{subject_id} {session} {dict(counts)}")


if __name__ == "__main__":
    main()
