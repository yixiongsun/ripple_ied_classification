"""Aggregate per-subject metrics across repeated held-out evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


MODEL_NAMES = ("original_full_baseline", "frozen_compact_model")
CLASS_NAMES = ("ripple", "ied", "noise")
OVERALL_METRICS = ("accuracy", "weighted_f1", "macro_f1")
CLASS_METRICS = ("precision", "recall", "f1")


def summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "standard_deviation": pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_subject(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "repeat_count": len(rows),
        "sample_count": int(rows[0]["sample_count"]),
    }
    for metric in OVERALL_METRICS:
        result[metric] = summarize_values([float(row[metric]) for row in rows])
    result["per_class"] = {
        class_name: {
            metric: summarize_values(
                [float(row["per_class"][class_name][metric]) for row in rows]
            )
            for metric in CLASS_METRICS
        }
        for class_name in CLASS_NAMES
    }
    for class_name in CLASS_NAMES:
        result["per_class"][class_name]["support"] = int(
            rows[0]["per_class"][class_name]["support"]
        )
    return result


def build_summary(payload: dict[str, object], source: Path) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for model_name in MODEL_NAMES:
        for fold in payload["fold_results"][model_name]:
            for subject, metrics in fold["per_subject_metrics"].items():
                grouped.setdefault((model_name, subject), []).append(metrics)

    subjects = sorted({subject for _, subject in grouped})
    results = {}
    differences = []
    for subject in subjects:
        baseline = summarize_subject(grouped[(MODEL_NAMES[0], subject)])
        compact = summarize_subject(grouped[(MODEL_NAMES[1], subject)])
        delta = {
            metric: compact[metric]["mean"] - baseline[metric]["mean"]
            for metric in OVERALL_METRICS
        }
        delta["per_class_f1"] = {
            class_name: (
                compact["per_class"][class_name]["f1"]["mean"]
                - baseline["per_class"][class_name]["f1"]["mean"]
            )
            for class_name in CLASS_NAMES
        }
        differences.append(delta["macro_f1"])
        results[subject] = {
            "original_full_baseline": baseline,
            "frozen_compact_model": compact,
            "compact_minus_baseline": delta,
        }

    return {
        "schema_version": 1,
        "source": str(source.resolve()),
        "aggregation": (
            "Unweighted mean and population standard deviation across each "
            "subject's three held-out seed evaluations."
        ),
        "subject_count": len(subjects),
        "compact_macro_f1_wins": sum(value > 0 for value in differences),
        "compact_macro_f1_ties": sum(value == 0 for value in differences),
        "compact_macro_f1_losses": sum(value < 0 for value in differences),
        "subjects": results,
    }


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Per-subject metrics",
        "",
        summary["aggregation"],
        "",
        "| Subject | Samples | Baseline macro F1 | Compact macro F1 | Delta | "
        "Baseline accuracy | Compact accuracy | Baseline noise F1 | Compact noise F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for subject, values in summary["subjects"].items():
        baseline = values["original_full_baseline"]
        compact = values["frozen_compact_model"]
        delta = values["compact_minus_baseline"]["macro_f1"]
        lines.append(
            f"| {subject} | {baseline['sample_count']} | "
            f"{baseline['macro_f1']['mean']:.4f} ± {baseline['macro_f1']['standard_deviation']:.4f} | "
            f"{compact['macro_f1']['mean']:.4f} ± {compact['macro_f1']['standard_deviation']:.4f} | "
            f"{delta:+.4f} | {baseline['accuracy']['mean']:.4f} | "
            f"{compact['accuracy']['mean']:.4f} | "
            f"{baseline['per_class']['noise']['f1']['mean']:.4f} | "
            f"{compact['per_class']['noise']['f1']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Compact macro-F1 wins/ties/losses: "
            f"{summary['compact_macro_f1_wins']}/"
            f"{summary['compact_macro_f1_ties']}/"
            f"{summary['compact_macro_f1_losses']}.",
            "",
            "The companion JSON contains accuracy, weighted F1, macro F1, and "
            "per-class precision, recall, and F1 with variability for every subject.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    output_directory = args.output_directory or args.result.parent
    output_directory.mkdir(parents=True, exist_ok=True)
    summary = build_summary(payload, args.result)
    json_path = output_directory / "per_subject_metrics.json"
    markdown_path = output_directory / "per_subject_metrics.md"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(json_path.resolve())
    print(markdown_path.resolve())


if __name__ == "__main__":
    main()
