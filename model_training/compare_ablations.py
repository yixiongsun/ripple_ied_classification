"""Compare saved ablation runs and create tabular and graphical summaries."""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_NAMES = ("ripple", "ied", "noise")
METRICS = ("accuracy", "weighted_f1", "macro_f1")
COMPARABILITY_FIELDS = (
    "folds",
    "epochs",
    "batch_size",
    "learning_rate",
    "contrast_weight",
    "contrast_temperature",
    "noise_weight",
    "repulsion_weight",
    "noise_margin",
    "tune_threshold",
    "calibration_fraction",
)


def resolve_result_paths(values: list[str]) -> list[Path]:
    """Resolve files, directories, and shell-independent glob patterns."""
    if not values:
        values = ["experiment_results/architecture"]
    paths = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            matches = sorted(path.glob("ablations_*.json"))
        elif any(character in value for character in "*?["):
            matches = [Path(match) for match in sorted(glob.glob(value))]
        else:
            matches = [path]
        paths.extend(matches)

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)
    if not unique_paths:
        raise FileNotFoundError("No ablation result JSON files were found")
    missing = [path for path in unique_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Ablation result does not exist: {missing[0]}")
    return unique_paths


def load_runs(paths: list[Path]) -> list[dict]:
    """Load and minimally validate ablation result payloads."""
    runs = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {"fold_results", "parameters", "class_order"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        if tuple(payload["class_order"]) != CLASS_NAMES:
            raise ValueError(
                f"{path} uses class order {payload['class_order']}, expected {CLASS_NAMES}"
            )
        runs.append({"path": path, "name": path.stem, "payload": payload})
    return runs


def comparability_warnings(runs: list[dict]) -> list[str]:
    """Describe training settings that differ between supplied result files."""
    warnings = []
    for field in COMPARABILITY_FIELDS:
        values = {
            json.dumps(run["payload"]["parameters"].get(field), sort_keys=True)
            for run in runs
        }
        if len(values) > 1:
            readable = ", ".join(sorted(values))
            warnings.append(f"{field} differs across runs: {readable}")
    dataset_paths = {
        run["payload"].get("dataset", {}).get("path") for run in runs
    }
    if len(dataset_paths) > 1:
        warnings.append("dataset path differs across runs")
    return warnings


def build_frames(runs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one row per variant and one row per fold."""
    summary_rows = []
    fold_rows = []
    multiple_runs = len(runs) > 1

    for run in runs:
        baseline_folds = run["payload"]["fold_results"].get("baseline", [])
        baseline_macro = (
            float(np.mean([fold["macro_f1"] for fold in baseline_folds]))
            if baseline_folds
            else np.nan
        )
        for variant, folds in run["payload"]["fold_results"].items():
            if not folds:
                continue
            display_name = (
                f"{run['name']} / {variant}" if multiple_runs else variant
            )
            row = {
                "run": run["name"],
                "variant": variant,
                "display_name": display_name,
                "folds": len(folds),
                "parameters": int(folds[0]["parameter_count"]),
            }
            for metric in METRICS:
                values = np.asarray([fold[metric] for fold in folds], dtype=float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=0))
            row["macro_f1_delta_from_baseline"] = (
                row["macro_f1_mean"] - baseline_macro
                if np.isfinite(baseline_macro)
                else np.nan
            )
            for class_name in CLASS_NAMES:
                values = np.asarray(
                    [fold["per_class"][class_name]["f1"] for fold in folds],
                    dtype=float,
                )
                row[f"{class_name}_f1_mean"] = float(values.mean())
                row[f"{class_name}_f1_std"] = float(values.std(ddof=0))
            summary_rows.append(row)

            for fold in folds:
                fold_row = {
                    "run": run["name"],
                    "variant": variant,
                    "display_name": display_name,
                    "fold": int(fold["fold"]),
                    "energy_threshold": float(fold["energy_threshold"]),
                }
                for metric in METRICS:
                    fold_row[metric] = float(fold[metric])
                for class_name in CLASS_NAMES:
                    fold_row[f"{class_name}_f1"] = float(
                        fold["per_class"][class_name]["f1"]
                    )
                fold_rows.append(fold_row)

    if not summary_rows:
        raise ValueError("No fold results were found in the supplied files")
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def print_ranking(summary: pd.DataFrame, metric: str) -> str:
    """Return a compact ranking suitable for both stdout and a text file."""
    ranked = summary.sort_values(f"{metric}_mean", ascending=False).copy()
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked["score"] = ranked.apply(
        lambda row: f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}",
        axis=1,
    )
    ranked["delta_macro_f1"] = ranked["macro_f1_delta_from_baseline"].map(
        lambda value: "n/a" if pd.isna(value) else f"{value:+.4f}"
    )
    table = ranked[
        ["rank", "run", "variant", "score", "delta_macro_f1", "parameters"]
    ].rename(
        columns={
            "run": "Run",
            "variant": "Variant",
            "score": metric,
            "delta_macro_f1": "Delta macro F1",
            "parameters": "Parameters",
        }
    )
    return table.to_string(index=False)


def plot_comparison(
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """Save a fold-aware metric plot and a per-class F1 heatmap."""
    ordered = summary.sort_values(f"{metric}_mean", ascending=True).reset_index(drop=True)
    figure_height = max(6.5, 0.48 * len(ordered) + 2.5)
    figure, (metric_axis, class_axis) = plt.subplots(
        1,
        2,
        figsize=(15, figure_height),
        gridspec_kw={"width_ratios": (1.3, 1)},
        constrained_layout=True,
    )

    positions = np.arange(len(ordered))
    means = ordered[f"{metric}_mean"].to_numpy()
    deviations = ordered[f"{metric}_std"].to_numpy()
    metric_axis.errorbar(
        means,
        positions,
        xerr=deviations,
        fmt="o",
        color="#2457a6",
        ecolor="#7896c4",
        capsize=4,
        markersize=7,
        label="mean ± SD",
        zorder=3,
    )
    rng = np.random.default_rng(0)
    for position, display_name in enumerate(ordered["display_name"]):
        values = folds.loc[folds["display_name"] == display_name, metric].to_numpy()
        jitter = rng.uniform(-0.09, 0.09, size=len(values))
        metric_axis.scatter(
            values,
            np.full(len(values), position) + jitter,
            color="#ef8a3a",
            s=24,
            alpha=0.8,
            label="individual folds" if position == 0 else None,
            zorder=4,
        )
    metric_axis.set_yticks(positions, ordered["display_name"].str.replace("_", " "))
    metric_axis.set_xlabel(metric.replace("_", " ").title())
    metric_axis.set_ylabel("Ablation variant")
    metric_axis.set_title("Overall performance by variant")
    metric_axis.grid(axis="x", alpha=0.25)
    metric_axis.legend(loc="upper left", frameon=False)

    class_values = ordered[
        [f"{class_name}_f1_mean" for class_name in CLASS_NAMES]
    ].to_numpy()
    image = class_axis.imshow(
        class_values,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        vmin=max(0.0, float(class_values.min()) - 0.05),
        vmax=min(1.0, float(class_values.max()) + 0.02),
    )
    class_axis.set_xticks(
        np.arange(len(CLASS_NAMES)),
        ["Ripple", "IED", "Noise"],
    )
    class_axis.set_yticks(positions, ordered["display_name"].str.replace("_", " "))
    class_axis.set_xlabel("Class")
    class_axis.set_ylabel("Ablation variant")
    class_axis.set_title("Mean class F1")
    midpoint = (image.norm.vmin + image.norm.vmax) / 2
    for row in range(class_values.shape[0]):
        for column in range(class_values.shape[1]):
            value = class_values[row, column]
            class_axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="black" if value > midpoint else "white",
                fontsize=9,
            )
    figure.colorbar(image, ax=class_axis, label="F1", fraction=0.046, pad=0.04)
    figure.suptitle("Ripple/IED model ablation comparison", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def compare_runs(
    result_paths: list[Path],
    *,
    metric: str = "macro_f1",
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Compare result files and save CSV, text, and PNG summaries."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of: {', '.join(METRICS)}")
    runs = load_runs(result_paths)
    summary, folds = build_frames(runs)
    warnings = comparability_warnings(runs)

    if output_directory is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output_directory = Path("experiment_results/architecture") / f"comparison_{stamp}"
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)

    ranking = print_ranking(summary, metric)
    report_parts = [f"Ranking by {metric}", "", ranking]
    if warnings:
        report_parts.extend(("", "Comparability warnings:", *[f"- {item}" for item in warnings]))
    report = "\n".join(report_parts) + "\n"
    print(report, end="")

    summary_path = output_directory / "ablation_summary.csv"
    folds_path = output_directory / "ablation_folds.csv"
    report_path = output_directory / "ablation_ranking.txt"
    plot_path = output_directory / "ablation_comparison.png"
    summary.sort_values(f"{metric}_mean", ascending=False).to_csv(summary_path, index=False)
    folds.to_csv(folds_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    plot_comparison(summary, folds, metric, plot_path)
    print(f"Saved comparison outputs to {output_directory.resolve()}")
    return {
        "summary": summary_path.resolve(),
        "folds": folds_path.resolve(),
        "ranking": report_path.resolve(),
        "plot": plot_path.resolve(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="*",
        help=(
            "Result JSON files, directories, or glob patterns. Defaults to all "
            "ablations_*.json files under experiment_results/architecture/."
        ),
    )
    parser.add_argument("--metric", choices=METRICS, default="macro_f1")
    parser.add_argument(
        "--output-dir",
        help="New output directory; defaults to a timestamped comparison directory",
    )
    args = parser.parse_args()
    compare_runs(
        resolve_result_paths(args.results),
        metric=args.metric,
        output_directory=args.output_dir,
    )


if __name__ == "__main__":
    main()
