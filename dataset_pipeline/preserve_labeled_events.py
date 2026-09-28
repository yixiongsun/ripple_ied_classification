"""Copy labeled event JSON/JPG pairs into a self-contained local dataset."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


CSV_COLUMNS = ("subject_id", "session_id", "image_path", "label")
EVENT_DIRECTORIES = {"ripples", "ieds"}


def _safe_component(value: str, field: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"Unsafe {field}: {value!r}")
    return value


def _copy_if_needed(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return "skipped"
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return "copied"


def preserve(source_csv: Path, destination_root: Path) -> tuple[int, int, int]:
    source_csv = source_csv.resolve()
    destination_root = destination_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    with source_csv.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        missing_columns = set(CSV_COLUMNS) - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing_columns))}")
        rows = list(reader)

    # Keep a byte-for-byte snapshot of the input label file for provenance.
    snapshot = destination_root / "updated_labels_source.csv"
    temporary_snapshot = snapshot.with_name(snapshot.name + ".tmp")
    shutil.copyfile(source_csv, temporary_snapshot)
    os.replace(temporary_snapshot, snapshot)

    local_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []
    copied = skipped = 0

    for row in rows:
        subject = _safe_component(row["subject_id"], "subject_id")
        session = _safe_component(row["session_id"], "session_id")
        source_jpg = Path(row["image_path"])
        source_json = source_jpg.with_suffix(".json")
        event_directory = _safe_component(source_jpg.parent.name, "event directory")
        if event_directory not in EVENT_DIRECTORIES:
            raise ValueError(f"Unexpected event directory: {source_jpg.parent}")

        missing = [str(path) for path in (source_jpg, source_json) if not path.is_file()]
        if missing:
            missing_rows.append(
                {
                    **{column: row[column] for column in CSV_COLUMNS},
                    "missing_files": " | ".join(missing),
                }
            )
            continue

        relative_jpg = Path(subject) / session / event_directory / source_jpg.name
        destination_jpg = destination_root / relative_jpg
        destination_json = destination_jpg.with_suffix(".json")
        for source, destination in (
            (source_jpg, destination_jpg),
            (source_json, destination_json),
        ):
            status = _copy_if_needed(source, destination)
            copied += status == "copied"
            skipped += status == "skipped"

        # The labeler resolves paths from the repository root, so retain the
        # leading dataset directory while keeping the preserved set portable.
        local_image_path = destination_root.name / relative_jpg
        local_rows.append(
            {
                "subject_id": subject,
                "session_id": session,
                "image_path": str(local_image_path),
                "label": row["label"],
            }
        )

    local_csv = destination_root / "updated_labels.csv"
    temporary_csv = local_csv.with_name(local_csv.name + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(local_rows)
    os.replace(temporary_csv, local_csv)

    missing_csv = destination_root / "missing_sources.csv"
    temporary_missing = missing_csv.with_name(missing_csv.name + ".tmp")
    missing_columns = (*CSV_COLUMNS, "missing_files")
    with temporary_missing.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=missing_columns)
        writer.writeheader()
        writer.writerows(missing_rows)
    os.replace(temporary_missing, missing_csv)

    return len(local_rows), len(missing_rows), copied + skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("updated_labels.csv"))
    parser.add_argument("--destination", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    preserved, missing, files = preserve(args.labels, args.destination)
    print(f"Preserved {preserved} labeled events ({files} files).")
    print(f"Recorded {missing} rows with unavailable source files.")


if __name__ == "__main__":
    main()
