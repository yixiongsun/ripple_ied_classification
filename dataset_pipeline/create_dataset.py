"""Build the final model dataset from labeled event JSON files.

The event exporter performs waveform transformation and spectrogram background
correction before labeling. This module packages those canonical JSON fields
with labels and metadata; it does not normalize them a second time.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm


LABELS = {
    "ripple": 0,
    "ied": 1,
    "ripple_ied": 1,
    "noise": 2,
}
REQUIRED_CSV_COLUMNS = {"subject_id", "session_id", "image_path", "label"}
REQUIRED_JSON_FIELDS = {
    "wave",
    "spec",
    "ripple_env",
    "amp",
    "entropy",
    "ratio",
}


def read_label_rows(label_csv: str | Path) -> list[dict[str, str]]:
    """Read and validate the labeling CSV."""
    label_csv = Path(label_csv)
    with label_csv.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        columns = set(reader.fieldnames or ())
        missing_columns = REQUIRED_CSV_COLUMNS - columns
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise ValueError(f"Label CSV is missing columns: {names}")
        rows = list(reader)

    for row_number, row in enumerate(rows, start=2):
        label = row["label"].strip()
        if label not in LABELS:
            raise ValueError(f"Unknown or empty label {label!r} on CSV row {row_number}")
    return rows


def json_path_for_row(row: dict[str, str], label_csv: str | Path) -> Path:
    """Resolve a row's event JSON path from its labeling image path."""
    path = Path(row["image_path"]).with_suffix(".json")
    if not path.is_absolute():
        path = Path(label_csv).resolve().parent / path
    return path


def find_missing_json(
    rows: list[dict[str, str]],
    label_csv: str | Path,
) -> list[Path]:
    """Return every labeled event whose canonical JSON file is unavailable."""
    return [
        path
        for row in rows
        if not (path := json_path_for_row(row, label_csv)).is_file()
    ]


def sample_from_record(
    record: dict,
    row: dict[str, str],
    json_path: Path,
) -> dict:
    """Convert one canonical event record into the training pickle schema."""
    missing_fields = REQUIRED_JSON_FIELDS - record.keys()
    if missing_fields:
        names = ", ".join(sorted(missing_fields))
        raise ValueError(f"{json_path} is missing JSON fields: {names}")

    wave = np.asarray(record["wave"], dtype=np.float32)
    spec = np.asarray(record["spec"], dtype=np.float32)
    ripple_env = np.asarray(record["ripple_env"], dtype=np.float32)
    amp = np.asarray(record["amp"], dtype=np.float32)
    entropy = np.asarray(record["entropy"], dtype=np.float32)
    ratio = np.asarray(record["ratio"], dtype=np.float32)

    if wave.shape != (3, 128):
        raise ValueError(f"Expected wave shape (3, 128) in {json_path}, got {wave.shape}")
    if spec.shape != (3, 100, 128):
        raise ValueError(
            f"Expected spec shape (3, 100, 128) in {json_path}, got {spec.shape}"
        )
    if ripple_env.shape != (3, 128):
        raise ValueError(
            f"Expected ripple_env shape (3, 128) in {json_path}, got {ripple_env.shape}"
        )
    for name, values in (("amp", amp), ("entropy", entropy), ("ratio", ratio)):
        if values.shape != (3,):
            raise ValueError(f"Expected {name} shape (3,) in {json_path}, got {values.shape}")

    return {
        "wave": wave,
        "spec": spec,
        "ripple_env": ripple_env,
        "label": LABELS[row["label"].strip()],
        "animal_id": row["subject_id"],
        "session": row["session_id"],
        "file": str(json_path),
        "amp": amp,
        "entropy": entropy,
        "ratio": ratio,
    }


def load_labeled_samples(
    label_csv: str | Path = "updated_labels.csv",
    *,
    skip_missing: bool = False,
    show_progress: bool = True,
) -> tuple[list[dict], list[Path]]:
    """Load final samples and return them with any missing JSON paths."""
    rows = read_label_rows(label_csv)
    missing = find_missing_json(rows, label_csv)
    if missing and not skip_missing:
        raise FileNotFoundError(
            f"{len(missing)} labeled JSON files are missing; first missing file: {missing[0]}"
        )

    missing_set = set(missing)
    samples = []
    iterator = tqdm(rows, desc="load labeled events", disable=not show_progress)
    for row in iterator:
        json_path = json_path_for_row(row, label_csv)
        if json_path in missing_set:
            continue
        with json_path.open("r", encoding="utf-8") as input_file:
            record = json.load(input_file)
        samples.append(sample_from_record(record, row, json_path))
    return samples, missing


def write_dataset(
    samples: list[dict],
    output_path: str | Path = "dataset_arcsinh.pkl",
    *,
    overwrite: bool = False,
) -> None:
    """Atomically serialize samples using Python's highest pickle protocol."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary.open("wb") as output_file:
            pickle.dump(samples, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_dataset(
    label_csv: str | Path = "updated_labels.csv",
    output_path: str | Path = "dataset_arcsinh.pkl",
    *,
    skip_missing: bool = False,
    overwrite: bool = False,
) -> tuple[list[dict], list[Path]]:
    """Load labeled events and write the final training dataset."""
    if Path(output_path).exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    samples, missing = load_labeled_samples(label_csv, skip_missing=skip_missing)
    write_dataset(samples, output_path, overwrite=overwrite)
    return samples, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final labeled LFP dataset.")
    parser.add_argument("--labels", default="updated_labels.csv")
    parser.add_argument("--output", default="dataset_arcsinh.pkl")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    samples, missing = create_dataset(
        args.labels,
        args.output,
        skip_missing=args.skip_missing,
        overwrite=args.overwrite,
    )
    label_counts = Counter(sample["label"] for sample in samples)
    print(f"Wrote {len(samples)} samples to {args.output}")
    print(f"Class counts: {dict(sorted(label_counts.items()))}")
    if missing:
        print(f"Skipped {len(missing)} missing JSON files")


if __name__ == "__main__":
    main()
