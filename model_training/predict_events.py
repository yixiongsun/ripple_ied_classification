"""Classify extracted ripple/IED candidates and save final event windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .loaders import load_event_samples, make_evaluation_loader
from .train import evaluate, load_checkpoint, predict_classes
from .train_full_dataset import resolve_device


LABEL_NAMES = {0: "ripple", 1: "ied", 2: "noise"}
SESSIONS = ("Sleep1", "Sleep2")


def prediction_rows(samples, predictions, energies) -> list[dict]:
    """Create the stable CSV schema used by downstream event cleanup."""
    rows = []
    for sample, prediction, energy in zip(samples, predictions, energies):
        path = Path(sample["file"])
        rows.append(
            {
                "subject_id": sample["animal_id"],
                "session_id": sample["session"],
                "event": int(prediction),
                "label": LABEL_NAMES[int(prediction)],
                "event_num": sample["event_num"],
                "energy": float(energy),
                "start": int(sample["window"][0]),
                "end": int(sample["window"][1]),
                "file": str(path),
                "image_path": str(path.with_suffix(".jpg")),
            }
        )
    return rows


def predict_directory(
    model,
    checkpoint: dict,
    device: torch.device,
    event_directory: Path,
    animal_id: str,
    session: str,
    candidate_label: int,
    output_stem: str,
    *,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> int:
    """Predict one subject/session candidate directory."""
    csv_path = event_directory.parent / f"{output_stem}_pred.csv"
    embedding_path = event_directory.parent / f"{output_stem}_embeddings.npy"
    if csv_path.exists() and embedding_path.exists() and not overwrite:
        return 0

    samples = load_event_samples(
        event_directory,
        animal_id,
        session,
        candidate_label,
        workers=num_workers,
    )
    if samples:
        loader = make_evaluation_loader(
            samples,
            batch_size,
            feature_mean=checkpoint["feat_mean"],
            feature_std=checkpoint["feat_std"],
            use_global_features=checkpoint.get("use_global_features", True),
            num_workers=num_workers,
        )
        probabilities, _, _, embeddings, energies = evaluate(model, loader, device)
        predictions = predict_classes(
            probabilities, energies, checkpoint["energy_threshold"]
        )
        rows = prediction_rows(samples, predictions, energies)
    else:
        embeddings = np.empty((0, 64), dtype=np.float32)
        rows = []

    columns = (
        "subject_id",
        "session_id",
        "event",
        "label",
        "event_num",
        "energy",
        "start",
        "end",
        "file",
        "image_path",
    )
    temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
    pd.DataFrame(rows, columns=columns).to_csv(temporary_csv, index=False)
    temporary_csv.replace(csv_path)
    temporary_embeddings = embedding_path.with_name(embedding_path.name + ".tmp.npy")
    np.save(temporary_embeddings, embeddings)
    temporary_embeddings.replace(embedding_path)
    return len(samples)


def empty_windows() -> np.ndarray:
    return np.empty((0, 2), dtype=np.int64)


def load_prediction_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    required = {"event", "start", "end"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing columns: {sorted(required - set(frame.columns))}")
    return frame


def windows_for_classes(frame: pd.DataFrame | None, classes: tuple[int, ...]) -> np.ndarray:
    if frame is None:
        return empty_windows()
    windows = frame.loc[frame["event"].isin(classes), ["start", "end"]].to_numpy()
    return windows.astype(np.int64) if len(windows) else empty_windows()


def overlaps_any(windows: np.ndarray, comparison: np.ndarray) -> np.ndarray:
    """Find interval overlaps without constructing an N-by-M boolean array."""
    if not len(windows) or not len(comparison):
        return np.zeros(len(windows), dtype=bool)
    order = np.argsort(comparison[:, 0], kind="stable")
    starts = comparison[order, 0]
    maximum_end = np.maximum.accumulate(comparison[order, 1])
    prior_index = np.searchsorted(starts, windows[:, 1], side="left") - 1
    overlaps = prior_index >= 0
    overlaps[overlaps] &= maximum_end[prior_index[overlaps]] > windows[overlaps, 0]
    return overlaps


def finalize_windows(session_directory: Path) -> dict[str, int]:
    """Apply the notebook's ripple/IED overlap rules and save NPY windows."""
    ripple_frame = load_prediction_csv(session_directory / "ripple_pred.csv")
    ied_frame = load_prediction_csv(session_directory / "ied_pred.csv")
    ripple_windows = windows_for_classes(ripple_frame, (0,))
    ripple_as_ied = windows_for_classes(ripple_frame, (1,))
    ied_windows = windows_for_classes(ied_frame, (1,))

    ripple_overlap = overlaps_any(ripple_windows, ied_windows)
    reclassified_overlap = overlaps_any(ripple_as_ied, ied_windows)
    true_ripples = ripple_windows[~ripple_overlap]
    true_ieds = np.vstack((ripple_as_ied[~reclassified_overlap], ied_windows))
    overlap_ripples = np.vstack(
        (ripple_windows[ripple_overlap], ripple_as_ied[reclassified_overlap])
    )
    candidate_ripples = windows_for_classes(ripple_frame, (0, 1, 2))

    np.save(session_directory / "true_ripple_windows.npy", true_ripples)
    np.save(session_directory / "true_ied_windows.npy", true_ieds)
    np.save(session_directory / "overlap_ripple_windows.npy", overlap_ripples)
    np.save(session_directory / "candidate_ripple_windows.npy", candidate_ripples)
    return {
        "ripples": len(true_ripples),
        "ieds": len(true_ieds),
        "overlaps": len(overlap_ripples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("predict", "finalize", "all"), default="all")
    parser.add_argument("--checkpoint", default="final_ripple_model.pt")
    parser.add_argument("--subjects", nargs="+", dest="subject_ids")
    parser.add_argument("--sessions", nargs="+", default=list(SESSIONS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        import subjects
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "predict_events.py requires the project-specific subjects.py module"
        ) from error

    subject_ids = args.subject_ids or subjects.get_subjects()
    device = resolve_device(args.device)
    model = checkpoint = None
    if args.action in ("predict", "all"):
        model, checkpoint = load_checkpoint(args.checkpoint, device)

    for animal_id in subject_ids:
        subject = subjects.load(animal_id)
        for session in args.sessions:
            session_directory = Path(subject["olm"]["directory"]) / session
            if args.action in ("predict", "all"):
                ripple_count = predict_directory(
                    model,
                    checkpoint,
                    device,
                    session_directory / "ripples",
                    animal_id,
                    session,
                    0,
                    "ripple",
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    overwrite=args.overwrite,
                )
                ied_count = predict_directory(
                    model,
                    checkpoint,
                    device,
                    session_directory / "ieds",
                    animal_id,
                    session,
                    1,
                    "ied",
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    overwrite=args.overwrite,
                )
                print(f"{animal_id} {session}: predicted {ripple_count} ripples, {ied_count} IEDs")
            if args.action in ("finalize", "all"):
                counts = finalize_windows(session_directory)
                print(f"{animal_id} {session}: final {counts}")


if __name__ == "__main__":
    main()
