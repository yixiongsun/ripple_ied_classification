"""Extract reproducible, non-sensitive plotting tables for showcase figures.

The source JSON files remain the scientific record. This script flattens only
the fields needed by the visual showcase and fails if the expected paired-run
structure or locked Phase 9 headline values change.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "figures" / "showcase" / "data"

FINAL_RESULTS = (
    REPO_ROOT
    / "final_confirmation_results"
    / "phase8_final_confirmation_20260930_233532"
    / "phase8_final_confirmation.json"
)
PER_SUBJECT_RESULTS = (
    REPO_ROOT
    / "final_confirmation_results"
    / "phase8_final_confirmation_20260930_233532"
    / "per_subject_metrics.json"
)
THRESHOLD_RESULTS = (
    REPO_ROOT
    / "threshold_results"
    / "phase7_threshold_robustness_20260930_214618"
    / "phase7_threshold_robustness.json"
)
SVM_SUMMARY = REPO_ROOT / "baseline_results" / "phase10_svm_comparison_summary.json"
LABELS_CSV = REPO_ROOT / "labels.csv"
ABLATION_SCREEN = (
    REPO_ROOT
    / "ablation_results"
    / "phase1_screen_20260928T233427.780964Z"
    / "phase1_screen.json"
)
ABLATION_CONFIRMATION_DIR = (
    REPO_ROOT
    / "ablation_results"
    / "phase1_confirmation_20260929T024757.286631Z"
)

CLASS_ORDER = ["ripple", "ied", "noise"]
MODEL_ORDER = ["frozen_compact_model", "original_full_baseline"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_key(row: Mapping[str, Any], *, repeat_seed: int | None = None) -> tuple[int, int]:
    seed = repeat_seed if repeat_seed is not None else int(row["repeat_seed"])
    return seed, int(row["fold"])


def normalized_model_runs(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for run in results["fold_results"][model]:
            rows.append(
                {
                    "repeat_seed": run["repeat_seed"],
                    "training_seed": run["training_seed"],
                    "fold": run["fold"],
                    "model": model,
                    "architecture": run["architecture"],
                    "accuracy": run["accuracy"],
                    "weighted_f1": run["weighted_f1"],
                    "macro_f1": run["macro_f1"],
                    "energy_threshold": run["energy_threshold"],
                    "training_time_seconds": run["duration_seconds"],
                    "parameter_count": run["parameter_count"],
                    "calibration_fraction": run["calibration_fraction"],
                    "validation_samples": run["validation_samples"],
                }
            )
    return rows


def normalized_class_metrics(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for run in results["fold_results"][model]:
            for class_name in CLASS_ORDER:
                metrics = run["per_class"][class_name]
                rows.append(
                    {
                        "repeat_seed": run["repeat_seed"],
                        "fold": run["fold"],
                        "model": model,
                        "class": class_name,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "support": metrics["support"],
                    }
                )
    return rows


def normalized_subject_runs(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    subject_ids = sorted(
        {
            subject_id
            for model in MODEL_ORDER
            for run in results["fold_results"][model]
            for subject_id in run["per_subject_metrics"]
        }
    )
    aliases = {subject_id: f"subject_{index:02d}" for index, subject_id in enumerate(subject_ids, 1)}

    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for run in results["fold_results"][model]:
            for subject_id, metrics in sorted(run["per_subject_metrics"].items()):
                rows.append(
                    {
                        "subject": aliases[subject_id],
                        "repeat_seed": run["repeat_seed"],
                        "fold": run["fold"],
                        "model": model,
                        "sample_count": metrics["sample_count"],
                        "accuracy": metrics["accuracy"],
                        "weighted_f1": metrics["weighted_f1"],
                        "macro_f1": metrics["macro_f1"],
                    }
                )
    return rows


def normalized_threshold_runs(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = sorted(
        results["fold_results"],
        key=lambda name: results["candidates"][name]["calibration_fraction"],
    )
    for candidate in candidates:
        for run in results["fold_results"][candidate]:
            rows.append(
                {
                    "calibration_fraction": run["calibration_fraction"],
                    "repeat_seed": run["repeat_seed"],
                    "training_seed": run["training_seed"],
                    "fold": run["fold"],
                    "accuracy": run["accuracy"],
                    "weighted_f1": run["weighted_f1"],
                    "macro_f1": run["macro_f1"],
                    "energy_threshold": run["energy_threshold"],
                    "calibration_macro_f1": run["calibration_macro_f1"],
                    "calibration_samples": run["calibration_samples"],
                    "training_time_seconds": run["duration_seconds"],
                    "parameter_count": run["parameter_count"],
                }
            )
    return rows


def append_ablation_rows(
    rows: list[dict[str, Any]],
    results: Mapping[str, Any],
    *,
    phase: str,
    evidence: str,
) -> None:
    repeat_seed = int(results["parameters"]["seed"])
    for configuration in sorted(results["fold_results"]):
        config = results["ablation_configs"][configuration]
        for run in results["fold_results"][configuration]:
            rows.append(
                {
                    "phase": phase,
                    "evidence": evidence,
                    "configuration": configuration,
                    "repeat_seed": repeat_seed,
                    "training_seed": run["seed"],
                    "fold": run["fold"],
                    "accuracy": run["accuracy"],
                    "weighted_f1": run["weighted_f1"],
                    "macro_f1": run["macro_f1"],
                    "energy_threshold": run["energy_threshold"],
                    "training_time_seconds": run["duration_seconds"],
                    "parameter_count": run["parameter_count"],
                    "modality": config["modality"],
                    "use_temporal_conv": config["use_temporal_conv"],
                    "use_channel_attention": config["use_channel_attention"],
                    "use_global_features": config["use_global_features"],
                    "use_spike_oscillation": config["use_spike_oscillation"],
                }
            )


def normalized_ablation_runs(
    screen: Mapping[str, Any], confirmation_results: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    append_ablation_rows(
        rows,
        screen,
        phase="phase1_screen",
        evidence="exploratory_3fold_1seed",
    )
    for result in confirmation_results:
        append_ablation_rows(
            rows,
            result,
            phase="phase1_confirmation",
            evidence="confirmed_5fold_3seed",
        )
    return rows


def dataset_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    compact_runs = results["fold_results"]["frozen_compact_model"]
    first_seed = min(int(run["repeat_seed"]) for run in compact_runs)
    first_seed_runs = [run for run in compact_runs if int(run["repeat_seed"]) == first_seed]
    return [
        {
            "class": class_name,
            "class_order": index,
            "event_count": sum(run["per_class"][class_name]["support"] for run in first_seed_runs),
            "subject_count": results["dataset"]["subject_count"],
        }
        for index, class_name in enumerate(CLASS_ORDER, 1)
    ]


def assert_close(actual: float, expected: float, *, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"Expected {expected}, got {actual}")


def mean(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    return statistics.mean(float(row[field]) for row in rows)


def population_sd(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    return statistics.pstdev(float(row[field]) for row in rows)


def assert_paired(
    result: Mapping[str, Any],
    groups: Sequence[str],
    *,
    repeat_seed: int | None = None,
    require_same_calibration: bool = False,
) -> None:
    indexed: dict[str, dict[tuple[int, int], Mapping[str, Any]]] = {}
    for group in groups:
        indexed[group] = {
            run_key(row, repeat_seed=repeat_seed): row for row in result["fold_results"][group]
        }
    expected_keys = set(indexed[groups[0]])
    for group in groups[1:]:
        if set(indexed[group]) != expected_keys:
            raise AssertionError(f"Unpaired seed/fold keys for {groups[0]} and {group}")
    for key in expected_keys:
        reference = indexed[groups[0]][key]
        for group in groups[1:]:
            candidate = indexed[group][key]
            if candidate["validation_subjects"] != reference["validation_subjects"]:
                raise AssertionError(f"Validation subjects differ at run {key}")
            if require_same_calibration and candidate["calibration_subjects"] != reference["calibration_subjects"]:
                raise AssertionError(f"Calibration subjects differ at run {key}")


def validate_sources(
    final: Mapping[str, Any],
    per_subject: Mapping[str, Any],
    threshold: Mapping[str, Any],
    screen: Mapping[str, Any],
    confirmations: Sequence[Mapping[str, Any]],
    model_runs: Sequence[Mapping[str, Any]],
    class_metrics: Sequence[Mapping[str, Any]],
    subject_runs: Sequence[Mapping[str, Any]],
    threshold_runs: Sequence[Mapping[str, Any]],
    ablation_runs: Sequence[Mapping[str, Any]],
    dataset: Sequence[Mapping[str, Any]],
) -> list[str]:
    checks: list[str] = []

    all_results = [final, threshold, screen, *confirmations]
    if any(result["class_order"] != CLASS_ORDER for result in all_results):
        raise AssertionError("Class order differs from ripple, IED, noise")
    checks.append("Class order is ripple, IED, noise in every source result file.")

    if final["dataset"]["sample_count"] != 7701 or final["dataset"]["subject_count"] != 30:
        raise AssertionError("Dataset headline counts do not match Phase 9")
    observed_counts = {row["class"]: row["event_count"] for row in dataset}
    if observed_counts != {"ripple": 4178, "ied": 1212, "noise": 2311}:
        raise AssertionError(f"Unexpected class counts: {observed_counts}")
    checks.append("Dataset totals reproduce 7,701 events from 30 subjects (4,178 ripple; 1,212 IED; 2,311 noise).")

    assert_paired(final, MODEL_ORDER, require_same_calibration=True)
    for model in MODEL_ORDER:
        if len(final["fold_results"][model]) != 15:
            raise AssertionError(f"Expected 15 final runs for {model}")
    checks.append("Final comparison contains 15 exactly paired seed/fold runs per model with identical validation and calibration subjects.")

    threshold_groups = sorted(threshold["fold_results"])
    assert_paired(threshold, threshold_groups)
    if any(len(threshold["fold_results"][group]) != 15 for group in threshold_groups):
        raise AssertionError("Expected 15 threshold runs per calibration fraction")
    checks.append("Threshold comparison contains 15 paired outer-validation runs for each calibration fraction.")

    assert_paired(screen, sorted(screen["fold_results"]), repeat_seed=int(screen["parameters"]["seed"]), require_same_calibration=True)
    for confirmation in confirmations:
        assert_paired(
            confirmation,
            sorted(confirmation["fold_results"]),
            repeat_seed=int(confirmation["parameters"]["seed"]),
            require_same_calibration=True,
        )
    checks.append("Ablation comparisons use matched fold/seed splits; screening and confirmation evidence remain separately labeled.")

    by_model = {model: [row for row in model_runs if row["model"] == model] for model in MODEL_ORDER}
    compact = by_model["frozen_compact_model"]
    baseline = by_model["original_full_baseline"]
    expected_summary = {
        "frozen_compact_model": {
            "macro_f1": (0.8488723806399198, 0.021082414046704846),
            "accuracy": (0.8611270459054595, 0.015621007010775786),
            "weighted_f1": (0.8607857850575258, 0.014403463590492983),
        },
        "original_full_baseline": {
            "macro_f1": (0.8484440953697612, 0.029816205765290302),
            "accuracy": (0.8619468578280369, 0.02349717660350543),
            "weighted_f1": (0.8617971087496259, 0.022272482971996528),
        },
    }
    for model, metrics in expected_summary.items():
        for metric, (expected_mean, expected_sd) in metrics.items():
            assert_close(mean(by_model[model], metric), expected_mean)
            assert_close(population_sd(by_model[model], metric), expected_sd)
    checks.append("Final macro F1, accuracy, and weighted F1 means and population standard deviations reproduce the locked Phase 9 values.")

    compact_index = {(row["repeat_seed"], row["fold"]): row for row in compact}
    baseline_index = {(row["repeat_seed"], row["fold"]): row for row in baseline}
    differences = [
        compact_index[key]["macro_f1"] - baseline_index[key]["macro_f1"]
        for key in sorted(compact_index)
    ]
    assert_close(statistics.mean(differences), 0.00042828527015862744)
    if (sum(value > 0 for value in differences), sum(value < 0 for value in differences)) != (8, 7):
        raise AssertionError("Unexpected compact-model paired win/loss count")
    checks.append("Paired macro-F1 difference is +0.000428 with 8 compact wins and 7 losses.")

    compact_class = [row for row in class_metrics if row["model"] == "frozen_compact_model"]
    expected_class_f1 = {
        "ripple": 0.8956960440039418,
        "ied": 0.8881840988891467,
        "noise": 0.7627369990266708,
    }
    for class_name, expected in expected_class_f1.items():
        assert_close(mean((row for row in compact_class if row["class"] == class_name), "f1"), expected)
    checks.append("Compact-model class F1 reproduces ripple 0.896, IED 0.888, and noise 0.763 after display rounding.")

    thresholds = [float(row["energy_threshold"]) for row in compact]
    assert_close(statistics.mean(thresholds), 2.7684781887878973)
    assert_close(statistics.pstdev(thresholds), 1.398331073668575)
    assert_close(min(thresholds), 0.828703286126256)
    assert_close(max(thresholds), 5.2763300612568855)
    checks.append("Compact-model threshold mean, standard deviation, and range reproduce 2.77 ± 1.40 and 0.83–5.28 after display rounding.")

    if set(row["parameter_count"] for row in compact) != {73985}:
        raise AssertionError("Unexpected compact parameter count")
    if set(row["parameter_count"] for row in baseline) != {113985}:
        raise AssertionError("Unexpected baseline parameter count")
    parameter_reduction = 1 - 73985 / 113985
    time_reduction = 1 - mean(compact, "training_time_seconds") / mean(baseline, "training_time_seconds")
    assert_close(parameter_reduction, 0.3509233671097074)
    assert_close(time_reduction, 0.4954360433447502)
    checks.append("Efficiency values reproduce 73,985 versus 113,985 parameters (−35.1%) and 112 versus 222 mean seconds (−49.5%).")

    if len(subject_runs) != 30 * 3 * 2 or len({row["subject"] for row in subject_runs}) != 30:
        raise AssertionError("Expected 180 de-identified subject/seed/model rows")
    if per_subject["subject_count"] != 30:
        raise AssertionError("Per-subject summary does not contain 30 subjects")
    checks.append("Subject table contains 180 rows: 30 de-identified subjects × 3 seeds × 2 models.")

    expected_lengths = {
        "final_model_runs.csv": (len(model_runs), 30),
        "final_class_metrics.csv": (len(class_metrics), 90),
        "threshold_runs.csv": (len(threshold_runs), 45),
        "ablation_runs.csv": (len(ablation_runs), 39),
    }
    for name, (actual, expected) in expected_lengths.items():
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected} rows, got {actual}")
    checks.append("All derived tables have the expected row counts.")
    return checks


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def cross_validation_web_payload(
    model_runs: Sequence[Mapping[str, Any]],
    class_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the compact JSON payload consumed by the portfolio website."""
    compact = [row for row in model_runs if row["model"] == "frozen_compact_model"]
    baseline = [row for row in model_runs if row["model"] == "original_full_baseline"]
    compact_by_pair = {(int(row["repeat_seed"]), int(row["fold"])): row for row in compact}
    baseline_by_pair = {(int(row["repeat_seed"]), int(row["fold"])): row for row in baseline}
    pair_keys = sorted(compact_by_pair)
    if set(pair_keys) != set(baseline_by_pair):
        raise AssertionError("Compact and baseline runs are not exactly paired")

    paired_runs = []
    differences = []
    for repeat_seed, fold in pair_keys:
        compact_score = float(compact_by_pair[(repeat_seed, fold)]["macro_f1"])
        baseline_score = float(baseline_by_pair[(repeat_seed, fold)]["macro_f1"])
        differences.append(compact_score - baseline_score)
        paired_runs.append(
            {
                "seed": repeat_seed + 1,
                "fold": fold,
                "compact_macro_f1": compact_score,
                "baseline_macro_f1": baseline_score,
            }
        )

    compact_class = [row for row in class_metrics if row["model"] == "frozen_compact_model"]
    class_f1 = [
        {
            "class": class_name,
            "f1": statistics.mean(
                float(row["f1"]) for row in compact_class if row["class"] == class_name
            ),
        }
        for class_name in CLASS_ORDER
    ]
    return {
        "schema_version": 1,
        "evaluation": "five subject-level folds × three seeds; internal cross-validation",
        "paired_runs": paired_runs,
        "summary": {
            "compact_macro_f1_mean": statistics.mean(float(row["macro_f1"]) for row in compact),
            "compact_macro_f1_sd": statistics.pstdev(float(row["macro_f1"]) for row in compact),
            "baseline_macro_f1_mean": statistics.mean(float(row["macro_f1"]) for row in baseline),
            "baseline_macro_f1_sd": statistics.pstdev(float(row["macro_f1"]) for row in baseline),
            "compact_accuracy_mean": statistics.mean(float(row["accuracy"]) for row in compact),
            "paired_macro_f1_difference": statistics.mean(differences),
            "compact_wins": sum(difference > 0 for difference in differences),
            "compact_losses": sum(difference < 0 for difference in differences),
        },
        "class_f1": class_f1,
    }


def robustness_web_payload(
    model_runs: Sequence[Mapping[str, Any]],
    threshold_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return efficiency and threshold-sensitivity data for the website."""
    models = []
    for model, label in (
        ("original_full_baseline", "Full baseline"),
        ("frozen_compact_model", "Compact selected"),
    ):
        rows = [row for row in model_runs if row["model"] == model]
        models.append(
            {
                "model": model,
                "label": label,
                "macro_f1_mean": statistics.mean(float(row["macro_f1"]) for row in rows),
                "parameter_count": int(rows[0]["parameter_count"]),
                "training_time_seconds_mean": statistics.mean(
                    float(row["training_time_seconds"]) for row in rows
                ),
            }
        )

    full, compact = models
    fractions = []
    for fraction in (0.15, 0.20, 0.25):
        rows = [
            row
            for row in threshold_runs
            if math.isclose(float(row["calibration_fraction"]), fraction)
        ]
        fractions.append(
            {
                "fraction": fraction,
                "selected": math.isclose(fraction, 0.20),
                "macro_f1_mean": statistics.mean(float(row["macro_f1"]) for row in rows),
                "macro_f1_sd": statistics.pstdev(float(row["macro_f1"]) for row in rows),
                "threshold_mean": statistics.mean(float(row["energy_threshold"]) for row in rows),
                "threshold_sd": statistics.pstdev(float(row["energy_threshold"]) for row in rows),
                "thresholds": [float(row["energy_threshold"]) for row in rows],
            }
        )

    return {
        "schema_version": 1,
        "models": models,
        "efficiency": {
            "parameter_reduction": 1 - compact["parameter_count"] / full["parameter_count"],
            "training_time_reduction": 1
            - compact["training_time_seconds_mean"] / full["training_time_seconds_mean"],
        },
        "calibration_fractions": fractions,
        "threshold_axis": {"minimum": 0.0, "maximum": 6.0},
    }


def scientific_comparison_web_payload(
    model_runs: Sequence[Mapping[str, Any]],
    svm_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a like-for-like three-class comparison for the conclusion."""
    actual: list[str] = []
    predicted: list[str] = []
    with LABELS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row["label"].strip().casefold()
            actual.append("ied" if label == "ripple_ied" else label)
            filename = row["image_path"].replace("\\", "/").rsplit("/", 1)[-1].casefold()
            predicted.append("ripple" if filename.startswith("ripple") else "ied")

    accuracy = sum(a == p for a, p in zip(actual, predicted)) / len(actual)
    class_f1 = []
    for class_name in CLASS_ORDER:
        true_positive = sum(a == class_name and p == class_name for a, p in zip(actual, predicted))
        false_positive = sum(a != class_name and p == class_name for a, p in zip(actual, predicted))
        false_negative = sum(a == class_name and p != class_name for a, p in zip(actual, predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_f1.append(f1)

    compact_rows = [row for row in model_runs if row["model"] == "frozen_compact_model"]
    svm = svm_summary["aggregate_metrics"]["three_class_svm"]
    payload = {
        "schema_version": 1,
        "class_order": CLASS_ORDER,
        "comparisons": [
            {
                "model": "threshold_candidate_rule",
                "label": "Threshold-derived candidate rule",
                "accuracy": accuracy,
                "macro_f1": statistics.mean(class_f1),
                "note": "Historical ripple_ied labels are merged into IED to match the final three-class task.",
            },
            {
                "model": "three_class_svm",
                "label": "Direct three-class SVM",
                "accuracy": float(svm["accuracy"]["mean"]),
                "macro_f1": float(svm["macro_f1"]["mean"]),
            },
            {
                "model": "frozen_compact_model",
                "label": "Waveform model",
                "accuracy": statistics.mean(float(row["accuracy"]) for row in compact_rows),
                "macro_f1": statistics.mean(float(row["macro_f1"]) for row in compact_rows),
            },
        ],
    }
    assert_close(payload["comparisons"][0]["accuracy"], 0.6269315673289183)
    assert_close(payload["comparisons"][0]["macro_f1"], 0.4349991818589402)
    return payload


def main() -> None:
    final = load_json(FINAL_RESULTS)
    per_subject = load_json(PER_SUBJECT_RESULTS)
    threshold = load_json(THRESHOLD_RESULTS)
    screen = load_json(ABLATION_SCREEN)
    confirmation_paths = sorted(ABLATION_CONFIRMATION_DIR.glob("phase1_confirmation_seed*.json"))
    confirmations = [load_json(path) for path in confirmation_paths]
    if len(confirmations) != 3:
        raise AssertionError(f"Expected three ablation confirmation files, found {len(confirmations)}")

    tables = {
        "dataset_summary.csv": dataset_rows(final),
        "final_model_runs.csv": normalized_model_runs(final),
        "final_class_metrics.csv": normalized_class_metrics(final),
        "final_subject_runs.csv": normalized_subject_runs(final),
        "threshold_runs.csv": normalized_threshold_runs(threshold),
        "ablation_runs.csv": normalized_ablation_runs(screen, confirmations),
    }

    checks = validate_sources(
        final,
        per_subject,
        threshold,
        screen,
        confirmations,
        tables["final_model_runs.csv"],
        tables["final_class_metrics.csv"],
        tables["final_subject_runs.csv"],
        tables["threshold_runs.csv"],
        tables["ablation_runs.csv"],
        tables["dataset_summary.csv"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, rows in tables.items():
        write_csv(OUTPUT_DIR / filename, rows)

    web_results = cross_validation_web_payload(
        tables["final_model_runs.csv"], tables["final_class_metrics.csv"]
    )
    with (OUTPUT_DIR / "cross_validation_web.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(web_results, handle, indent=2)
        handle.write("\n")

    robustness_results = robustness_web_payload(
        tables["final_model_runs.csv"], tables["threshold_runs.csv"]
    )
    with (OUTPUT_DIR / "robustness_web.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(robustness_results, handle, indent=2)
        handle.write("\n")

    svm_summary = load_json(SVM_SUMMARY)
    scientific_comparison = scientific_comparison_web_payload(
        tables["final_model_runs.csv"], svm_summary
    )
    with (OUTPUT_DIR / "scientific_comparison_web.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(scientific_comparison, handle, indent=2)
        handle.write("\n")

    source_paths = [
        FINAL_RESULTS,
        PER_SUBJECT_RESULTS,
        THRESHOLD_RESULTS,
        SVM_SUMMARY,
        LABELS_CSV,
        ABLATION_SCREEN,
        *confirmation_paths,
    ]
    manifest = {
        "schema_version": 1,
        "description": "Reproducible plotting tables for the ripple/IED portfolio showcase.",
        "class_order": CLASS_ORDER,
        "subject_identifiers": "Deterministic aliases based on sorted source identifiers; no source subject IDs are exported.",
        "sources": [source_record(path) for path in source_paths],
        "tables": {
            filename: {"row_count": len(rows), "columns": list(rows[0].keys())}
            for filename, rows in tables.items()
        },
        "web_artifacts": {
            "cross_validation_web.json": {
                "paired_run_count": len(web_results["paired_runs"]),
                "class_order": CLASS_ORDER,
            },
            "robustness_web.json": {
                "model_count": len(robustness_results["models"]),
                "calibration_fraction_count": len(robustness_results["calibration_fractions"]),
            },
            "scientific_comparison_web.json": {
                "comparison_count": len(scientific_comparison["comparisons"]),
                "class_order": CLASS_ORDER,
            },
        },
    }
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    report_lines = [
        "# Showcase data validation",
        "",
        "Generated by `figures/showcase/scripts/extract_showcase_data.py`.",
        "",
        *[f"- PASS — {check}" for check in checks],
        "",
        "All values in CSV files retain source precision. Rounding belongs in figure code only.",
        "",
    ]
    (OUTPUT_DIR / "VALIDATION.md").write_text("\n".join(report_lines), encoding="utf-8")

    for filename, rows in tables.items():
        print(f"{filename}: {len(rows)} rows")
    print(f"validation checks: {len(checks)} passed")


if __name__ == "__main__":
    main()
