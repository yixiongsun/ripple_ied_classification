"""Score filename-derived event types against the hand labels."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


PREDICTED_CLASSES = ("ripple", "ied")


def filename_prediction(image_path: str) -> str:
    """Infer ripple or IED from the start of an image's filename."""
    filename = PurePosixPath(image_path.replace("\\", "/")).name.casefold()
    for event_type in PREDICTED_CLASSES:
        if filename.startswith(event_type):
            return event_type
    raise ValueError(f"Cannot infer an event type from filename: {image_path!r}")


def normalized_path(image_path: str) -> str:
    """Normalize separators and case for duplicate-path diagnostics."""
    return image_path.replace("\\", "/").casefold()


def evaluate(label_csv: str | Path) -> None:
    """Print naive filename-baseline metrics for a labeling CSV."""
    label_csv = Path(label_csv)
    with label_csv.open("r", newline="", encoding="utf-8-sig") as input_file:
        rows = list(csv.DictReader(input_file))

    required_columns = {"image_path", "label"}
    columns = set(rows[0]) if rows else set()
    if missing := required_columns - columns:
        raise ValueError(
            f"{label_csv} is missing required columns: {', '.join(sorted(missing))}"
        )
    if not rows:
        raise ValueError(f"{label_csv} contains no labeled rows")

    actual = [row["label"].strip().casefold() for row in rows]
    predicted = [filename_prediction(row["image_path"]) for row in rows]
    classes = sorted(set(actual) | set(predicted))

    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=classes,
        zero_division=0,
    )
    weighted_f1 = sum(score * count for score, count in zip(f1, support)) / sum(
        support
    )
    macro_f1 = sum(f1) / len(f1)

    labels_by_path: dict[str, set[str]] = defaultdict(set)
    for row, label in zip(rows, actual):
        labels_by_path[normalized_path(row["image_path"])].add(label)
    duplicate_rows = len(rows) - len(labels_by_path)
    conflicting_paths = sum(len(labels) > 1 for labels in labels_by_path.values())

    print(f"Labels: {label_csv}")
    print(f"Rows scored: {len(rows):,}")
    print(f"Actual labels: {dict(sorted(Counter(actual).items()))}")
    print(f"Naive predictions: {dict(sorted(Counter(predicted).items()))}")
    print(f"Duplicate rows: {duplicate_rows:,}")
    print(f"Paths with conflicting labels: {conflicting_paths:,}")
    print()
    print(f"Accuracy:    {accuracy_score(actual, predicted):.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print()
    print("Per-class metrics:")
    print(f"{'class':<12} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for label, class_precision, class_recall, class_f1, count in zip(
        classes, precision, recall, f1, support
    ):
        print(
            f"{label:<12} {class_precision:>10.4f} {class_recall:>10.4f} "
            f"{class_f1:>10.4f} {count:>10}"
        )

    matrix = confusion_matrix(actual, predicted, labels=classes)
    print()
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(f"{'':<12}" + "".join(f"{label:>12}" for label in classes))
    for label, counts in zip(classes, matrix):
        print(f"{label:<12}" + "".join(f"{count:>12}" for count in counts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare labels with the naive assumption that ripple* files are "
            "ripples and ied* files are IEDs."
        )
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("labels.csv"),
        help="label CSV to score (default: labels.csv)",
    )
    args = parser.parse_args()
    evaluate(args.labels)


if __name__ == "__main__":
    main()
