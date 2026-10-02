"""Validate and summarize Phase 10 SVM results against locked Phase 8 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np

from .ablation_test import CLASS_NAMES, SUMMARY_METRICS, save_json


REPORT_START = "<!-- PHASE10_CLASSICAL_BASELINE_START -->"
REPORT_END = "<!-- PHASE10_CLASSICAL_BASELINE_END -->"


def _bootstrap_interval(values: np.ndarray, *, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    return [float(value) for value in np.quantile(values[indices].mean(axis=1), (0.025, 0.975))]


def _exact_sign_flip_p_value(values: np.ndarray) -> float:
    count = 1 << len(values)
    patterns = np.arange(count, dtype=np.uint64)[:, None]
    bits = (patterns >> np.arange(len(values), dtype=np.uint64)) & 1
    signs = bits.astype(float) * 2 - 1
    observed = abs(float(values.mean()))
    permuted = np.abs((signs * values).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def _key(item: dict) -> tuple[int, int]:
    return int(item["repeat_seed"]), int(item["fold"])


def _assert_pairing(neural: list[dict], baseline: list[dict], model_name: str) -> None:
    neural_by_key = {_key(item): item for item in neural}
    baseline_by_key = {_key(item): item for item in baseline}
    if neural_by_key.keys() != baseline_by_key.keys():
        raise ValueError(f"Seed/fold keys differ for {model_name}")
    comparisons = (
        ("train_subjects", "model-fitting subjects"),
        ("calibration_subjects", "calibration subjects"),
        ("validation_subjects", "validation subjects"),
        ("validation_samples", "validation sample count"),
    )
    for key in sorted(neural_by_key):
        for field, description in comparisons:
            neural_value = neural_by_key[key][field]
            baseline_value = baseline_by_key[key][field]
            if neural_value != baseline_value:
                raise ValueError(
                    f"Pairing failure for {model_name}, seed/fold {key}: "
                    f"{description} differs"
                )


def _difference_summary(
    neural: list[dict], baseline: list[dict], extractor
) -> dict[str, object]:
    neural_by_key = {_key(item): item for item in neural}
    baseline_by_key = {_key(item): item for item in baseline}
    keys = sorted(neural_by_key)
    differences = np.asarray(
        [extractor(neural_by_key[key]) - extractor(baseline_by_key[key]) for key in keys],
        dtype=float,
    )
    seeds = sorted({key[0] for key in keys})
    seed_means = np.asarray(
        [differences[[key[0] == seed for key in keys]].mean() for seed in seeds]
    )
    return {
        "direction": "frozen_compact_model minus SVM baseline",
        "pair_count": len(differences),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "standard_deviation": float(differences.std(ddof=1)),
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
        "seed_mean_differences": seed_means.tolist(),
        "seed_block_bootstrap_95_interval": _bootstrap_interval(seed_means, seed=1),
        "seed_level_exact_sign_flip_p_value": _exact_sign_flip_p_value(seed_means),
    }


def _aggregate_confusion(results: list[dict]) -> list[list[int]]:
    return np.sum(
        [np.asarray(item["confusion_matrix"], dtype=np.int64) for item in results], axis=0
    ).tolist()


def _metric_table(results_by_name: dict[str, list[dict]]) -> dict:
    table = {}
    for name, results in results_by_name.items():
        table[name] = {}
        for metric in SUMMARY_METRICS:
            values = [float(item[metric]) for item in results]
            table[name][metric] = {
                "mean": mean(values),
                "standard_deviation": pstdev(values),
            }
    return table


def _per_class_table(results_by_name: dict[str, list[dict]]) -> dict:
    table = {}
    for name, results in results_by_name.items():
        table[name] = {}
        for class_name in CLASS_NAMES:
            values = [
                float(item["per_class"][class_name]["f1"]) for item in results
            ]
            table[name][class_name] = {
                "mean_f1": mean(values),
                "standard_deviation": pstdev(values),
            }
    return table


def _per_subject_differences(neural: list[dict], baseline: list[dict]) -> dict:
    neural_by_key = {_key(item): item for item in neural}
    baseline_by_key = {_key(item): item for item in baseline}
    values: dict[str, list[float]] = {}
    for key in sorted(neural_by_key):
        neural_subjects = neural_by_key[key]["per_subject_metrics"]
        baseline_subjects = baseline_by_key[key]["per_subject_metrics"]
        if neural_subjects.keys() != baseline_subjects.keys():
            raise ValueError(f"Per-subject keys differ for seed/fold {key}")
        for subject in neural_subjects:
            values.setdefault(subject, []).append(
                float(neural_subjects[subject]["macro_f1"])
                - float(baseline_subjects[subject]["macro_f1"])
            )
    return {
        subject: {
            "mean_macro_f1_difference": mean(differences),
            "differences": differences,
        }
        for subject, differences in sorted(values.items())
    }


def build_summary(classical: dict, locked: dict) -> dict[str, object]:
    """Build a strict paired summary, raising on any protocol mismatch."""
    neural = list(locked["fold_results"]["frozen_compact_model"])
    baselines = classical["fold_results"]
    results_by_name = {"frozen_compact_model": neural, **baselines}
    paired = {}
    for model_name, results in baselines.items():
        _assert_pairing(neural, results, model_name)
        paired[model_name] = {
            "metrics": {
                metric: _difference_summary(
                    neural, results, lambda item, name=metric: float(item[name])
                )
                for metric in SUMMARY_METRICS
            },
            "per_class_f1": {
                class_name: _difference_summary(
                    neural,
                    results,
                    lambda item, name=class_name: float(item["per_class"][name]["f1"]),
                )
                for class_name in CLASS_NAMES
            },
            "per_subject_macro_f1": _per_subject_differences(neural, results),
        }
    thresholds = [
        float(item["rejection_threshold"])
        for item in baselines["binary_rejection_svm"]
    ]
    efficiency = {}
    for name, results in results_by_name.items():
        parameter_counts = [
            float(item.get("parameter_count", item.get("support_vectors")))
            for item in results
        ]
        efficiency[name] = {
            "mean_total_fold_seconds": mean(
                float(item["duration_seconds"]) for item in results
            ),
            "mean_final_fit_seconds": (
                mean(float(item["fit_seconds"]) for item in results)
                if "fit_seconds" in results[0]
                else None
            ),
            "mean_inference_seconds": (
                mean(float(item["inference_seconds"]) for item in results)
                if "inference_seconds" in results[0]
                else None
            ),
            "mean_model_size_bytes": (
                mean(float(item["serialized_model_size_bytes"]) for item in results)
                if "serialized_model_size_bytes" in results[0]
                else None
            ),
            "model_size_basis": (
                "serialized sklearn pipeline"
                if "serialized_model_size_bytes" in results[0]
                else "not recorded in locked Phase 8 result"
            ),
            "mean_parameter_or_support_vector_count": mean(parameter_counts),
            "count_basis": (
                "support vectors"
                if "support_vectors" in results[0]
                else "trainable parameters"
            ),
        }
    return {
        "schema_version": 1,
        "comparison_direction": "frozen_compact_model minus SVM baseline",
        "qualification": (
            "Cross-validation folds and repeated seeds reuse subjects and are not "
            "independent experimental units. With only three seed blocks, intervals "
            "and sign-flip results are descriptive and have very low power."
        ),
        "aggregate_metrics": _metric_table(results_by_name),
        "per_class_metrics": _per_class_table(results_by_name),
        "aggregate_confusion_matrices": {
            name: _aggregate_confusion(results)
            for name, results in results_by_name.items()
        },
        "paired_analysis": paired,
        "rejection_threshold": {
            "mean": mean(thresholds),
            "median": median(thresholds),
            "standard_deviation": pstdev(thresholds),
            "minimum": min(thresholds),
            "maximum": max(thresholds),
            "values": thresholds,
        },
        "runtime_and_model_size": efficiency,
    }


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def render_markdown(summary: dict) -> str:
    metrics = summary["aggregate_metrics"]
    lines = [
        "## Post-hoc classical baseline comparison",
        "",
        "This analysis preserves the locked Phase 8 outer folds and calibration "
        "subjects. It is an internal post-hoc benchmark, not a new model-selection "
        "phase, and there is still no untouched test cohort.",
        "",
        "| Model | Accuracy | Weighted F1 | Macro F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "frozen_compact_model": "Frozen compact model",
        "binary_rejection_svm": "Binary RBF-SVM + rejection",
        "three_class_svm": "Direct three-class RBF-SVM",
    }
    for name in ("frozen_compact_model", "binary_rejection_svm", "three_class_svm"):
        row = metrics[name]
        lines.append(
            f"| {labels[name]} | "
            f"{_format_metric(row['accuracy']['mean'])} ± {_format_metric(row['accuracy']['standard_deviation'])} | "
            f"{_format_metric(row['weighted_f1']['mean'])} ± {_format_metric(row['weighted_f1']['standard_deviation'])} | "
            f"{_format_metric(row['macro_f1']['mean'])} ± {_format_metric(row['macro_f1']['standard_deviation'])} |"
        )
    lines.extend(["", "Paired macro-F1 differences (compact model minus baseline):", ""])
    for name in ("binary_rejection_svm", "three_class_svm"):
        item = summary["paired_analysis"][name]["metrics"]["macro_f1"]
        interval = item["seed_block_bootstrap_95_interval"]
        lines.append(
            f"- {labels[name]}: mean {_format_metric(item['mean_difference'])}, "
            f"median {_format_metric(item['median_difference'])}, "
            f"{item['wins']} wins / {item['ties']} ties / {item['losses']} losses; "
            f"seed-block 95% interval [{_format_metric(interval[0])}, "
            f"{_format_metric(interval[1])}], exact seed-level sign-flip "
            f"p={item['seed_level_exact_sign_flip_p_value']:.4f}."
        )
    threshold = summary["rejection_threshold"]
    lines.extend(
        [
            "",
            f"The rejection threshold averaged {threshold['mean']:.3f} "
            f"(range {threshold['minimum']:.3f}–{threshold['maximum']:.3f}).",
            "",
            "Mean class F1 values were:",
            "",
            "| Model | Ripple | IED | Noise |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in ("frozen_compact_model", "binary_rejection_svm", "three_class_svm"):
        row = summary["per_class_metrics"][name]
        lines.append(
            f"| {labels[name]} | {row['ripple']['mean_f1']:.4f} | "
            f"{row['ied']['mean_f1']:.4f} | {row['noise']['mean_f1']:.4f} |"
        )
    lines.extend(["", summary["qualification"]])
    return "\n".join(lines) + "\n"


def update_phase9_report(path: str | Path, section: str) -> None:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    replacement = f"{REPORT_START}\n{section.rstrip()}\n{REPORT_END}"
    if REPORT_START in text and REPORT_END in text:
        before, remainder = text.split(REPORT_START, 1)
        _, after = remainder.split(REPORT_END, 1)
        text = before.rstrip() + "\n\n" + replacement + after
    else:
        text = text.rstrip() + "\n\n" + replacement + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--locked",
        default=(
            "final_confirmation_results/phase8_final_confirmation_20260930_233532/"
            "phase8_final_confirmation.json"
        ),
    )
    parser.add_argument("--output")
    parser.add_argument("--markdown")
    parser.add_argument("--phase9-report", default="final_confirmation_results/PHASE9_REPORT.md")
    parser.add_argument("--no-report-update", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    baseline_path = Path(args.baseline)
    output = Path(args.output) if args.output else baseline_path.with_name(
        baseline_path.stem + "_summary.json"
    )
    markdown_path = Path(args.markdown) if args.markdown else output.with_suffix(".md")
    if markdown_path.exists() and not args.overwrite:
        raise FileExistsError(f"Summary already exists: {markdown_path}")
    classical = json.loads(baseline_path.read_text(encoding="utf-8"))
    locked = json.loads(Path(args.locked).read_text(encoding="utf-8"))
    summary = build_summary(classical, locked)
    save_json(summary, output, overwrite=args.overwrite)
    section = render_markdown(summary)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(section, encoding="utf-8")
    if not args.no_report_update:
        update_phase9_report(args.phase9_report, section)
    print(f"Saved JSON summary to {output.resolve()}")
    print(f"Saved Markdown summary to {markdown_path.resolve()}")


if __name__ == "__main__":
    main()

