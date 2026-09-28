"""Summarize and plot JSON outputs produced by ``compare_losses.py``."""

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
LOSS_COMPONENTS = (
    "supervised",
    "weighted_contrast",
    "weighted_noise",
    "weighted_repulsion",
    "total",
)
COMPARABILITY_FIELDS = (
    "architecture",
    "folds",
    "epochs",
    "batch_size",
    "learning_rate",
    "subjects_per_batch",
    "tune_threshold",
    "calibration_fraction",
)


def resolve_result_paths(values: list[str]) -> list[Path]:
    """Resolve files, directories, and Windows-safe glob expressions."""
    if not values:
        values = ["loss_comparison_results"]
    paths = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            matches = sorted(path.glob("losses_*.json"))
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
        raise FileNotFoundError("No loss comparison JSON files were found")
    missing = [path for path in unique_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Loss comparison result does not exist: {missing[0]}")
    return unique_paths


def load_runs(paths: list[Path]) -> list[dict]:
    """Load and validate loss-comparison payloads."""
    runs = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {"fold_results", "parameters", "class_order", "loss_configs"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        if payload.get("experiment_type") != "loss_formulation_comparison":
            raise ValueError(f"{path} is not a loss-formulation result file")
        if tuple(payload["class_order"]) != CLASS_NAMES:
            raise ValueError(
                f"{path} uses class order {payload['class_order']}, expected {CLASS_NAMES}"
            )
        runs.append({"path": path, "name": path.stem, "payload": payload})
    return runs


def comparability_warnings(runs: list[dict]) -> list[str]:
    """Report settings that make supplied runs unsuitable for direct pooling."""
    warnings = []
    for field in COMPARABILITY_FIELDS:
        values = {
            json.dumps(run["payload"]["parameters"].get(field), sort_keys=True)
            for run in runs
        }
        if len(values) > 1:
            warnings.append(f"{field} differs across runs: {', '.join(sorted(values))}")
    dataset_paths = {
        run["payload"].get("dataset", {}).get("path") for run in runs
    }
    if len(dataset_paths) > 1:
        warnings.append("dataset path differs across runs")
    return warnings


def build_frames(
    runs: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build formulation, fold, and epoch-level data frames."""
    summary_rows = []
    fold_rows = []
    history_rows = []
    multiple_runs = len(runs) > 1

    for run in runs:
        current_folds = run["payload"]["fold_results"].get("current", [])
        current_macro = (
            float(np.mean([fold["macro_f1"] for fold in current_folds]))
            if current_folds
            else np.nan
        )
        for formulation, folds in run["payload"]["fold_results"].items():
            if not folds:
                continue
            display_name = (
                f"{run['name']} / {formulation}" if multiple_runs else formulation
            )
            row = {
                "run": run["name"],
                "formulation": formulation,
                "display_name": display_name,
                "folds": len(folds),
                "parameters": int(folds[0]["parameter_count"]),
            }
            for metric in METRICS:
                values = np.asarray([fold[metric] for fold in folds], dtype=float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=0))
            row["macro_f1_delta_from_current"] = (
                row["macro_f1_mean"] - current_macro
                if np.isfinite(current_macro)
                else np.nan
            )
            for class_name in CLASS_NAMES:
                values = np.asarray(
                    [fold["per_class"][class_name]["f1"] for fold in folds],
                    dtype=float,
                )
                row[f"{class_name}_f1_mean"] = float(values.mean())
                row[f"{class_name}_f1_std"] = float(values.std(ddof=0))
            thresholds = [
                float(fold["energy_threshold"])
                for fold in folds
                if fold.get("energy_threshold") is not None
            ]
            row["threshold_mean"] = float(np.mean(thresholds)) if thresholds else np.nan
            row["threshold_std"] = float(np.std(thresholds)) if thresholds else np.nan
            summary_rows.append(row)

            for fold in folds:
                fold_row = {
                    "run": run["name"],
                    "formulation": formulation,
                    "display_name": display_name,
                    "fold": int(fold["fold"]),
                    "energy_threshold": fold.get("energy_threshold"),
                }
                for metric in METRICS:
                    fold_row[metric] = float(fold[metric])
                for class_name in CLASS_NAMES:
                    fold_row[f"{class_name}_f1"] = float(
                        fold["per_class"][class_name]["f1"]
                    )
                fold_rows.append(fold_row)

                for epoch in fold["training_history"]:
                    history_row = {
                        "run": run["name"],
                        "formulation": formulation,
                        "display_name": display_name,
                        "fold": int(fold["fold"]),
                        "epoch": int(epoch["epoch"]),
                    }
                    for component in LOSS_COMPONENTS:
                        history_row[component] = float(epoch.get(component, 0.0))
                    history_rows.append(history_row)

    if not summary_rows:
        raise ValueError("No loss formulation results were found")
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(history_rows),
    )


def ranking_text(summary: pd.DataFrame, metric: str) -> str:
    """Return a compact ranked text table."""
    ranked = summary.sort_values(f"{metric}_mean", ascending=False).copy()
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked["score"] = ranked.apply(
        lambda row: f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}",
        axis=1,
    )
    ranked["delta"] = ranked["macro_f1_delta_from_current"].map(
        lambda value: "n/a" if pd.isna(value) else f"{value:+.4f}"
    )
    table = ranked[
        ["rank", "run", "formulation", "score", "delta", "parameters"]
    ].rename(
        columns={
            "run": "Run",
            "formulation": "Formulation",
            "score": metric,
            "delta": "Delta macro F1",
            "parameters": "Parameters",
        }
    )
    return table.to_string(index=False)


def plot_performance(
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    metric: str,
    path: Path,
) -> None:
    """Plot fold-level performance and mean class F1."""
    ordered = summary.sort_values(f"{metric}_mean", ascending=True).reset_index(drop=True)
    height = max(6.5, 0.5 * len(ordered) + 2.5)
    figure, (metric_axis, class_axis) = plt.subplots(
        1,
        2,
        figsize=(15, height),
        gridspec_kw={"width_ratios": (1.3, 1)},
        constrained_layout=True,
    )
    positions = np.arange(len(ordered))
    metric_axis.errorbar(
        ordered[f"{metric}_mean"],
        positions,
        xerr=ordered[f"{metric}_std"],
        fmt="o",
        color="#2457a6",
        ecolor="#7896c4",
        capsize=4,
        markersize=7,
        label="mean +/- SD",
        zorder=3,
    )
    rng = np.random.default_rng(0)
    for position, name in enumerate(ordered["display_name"]):
        values = folds.loc[folds["display_name"] == name, metric].to_numpy()
        metric_axis.scatter(
            values,
            np.full(len(values), position) + rng.uniform(-0.09, 0.09, len(values)),
            color="#ef8a3a",
            s=24,
            alpha=0.85,
            label="individual folds" if position == 0 else None,
            zorder=4,
        )
    labels = ordered["display_name"].str.replace("_", " ")
    metric_axis.set_yticks(positions, labels)
    metric_axis.set_xlabel(metric.replace("_", " ").title())
    metric_axis.set_ylabel("Loss formulation")
    metric_axis.set_title("Overall validation performance")
    metric_axis.grid(axis="x", alpha=0.25)
    metric_axis.legend(loc="upper left", frameon=False)

    class_values = ordered[
        [f"{class_name}_f1_mean" for class_name in CLASS_NAMES]
    ].to_numpy()
    image = class_axis.imshow(
        class_values,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=max(0.0, float(class_values.min()) - 0.05),
        vmax=min(1.0, float(class_values.max()) + 0.02),
    )
    class_axis.set_xticks(np.arange(3), ["Ripple", "IED", "Noise"])
    class_axis.set_yticks(positions, labels)
    class_axis.set_xlabel("Class")
    class_axis.set_ylabel("Loss formulation")
    class_axis.set_title("Mean validation class F1")
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
    figure.suptitle("Loss formulation comparison", fontsize=15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_training_curves(history: pd.DataFrame, path: Path) -> None:
    """Plot mean weighted loss components for each formulation."""
    names = list(dict.fromkeys(history["display_name"]))
    columns = 2
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, max(4.2 * rows, 5)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {
        "supervised": "#2457a6",
        "weighted_contrast": "#7b4ab5",
        "weighted_noise": "#df7f24",
        "weighted_repulsion": "#2d8b57",
        "total": "#333333",
    }
    labels = {
        "supervised": "supervised",
        "weighted_contrast": "contrast",
        "weighted_noise": "noise",
        "weighted_repulsion": "repulsion",
        "total": "total",
    }
    for axis, name in zip(axes.flat, names):
        selected = history.loc[history["display_name"] == name]
        grouped = selected.groupby("epoch", sort=True)
        epochs = np.asarray(sorted(selected["epoch"].unique()))
        for component in LOSS_COMPONENTS:
            means = grouped[component].mean().reindex(epochs).to_numpy()
            if component != "total" and np.allclose(means, 0):
                continue
            axis.plot(
                epochs,
                means,
                color=colors[component],
                linewidth=2.0 if component == "total" else 1.5,
                linestyle="--" if component == "total" else "-",
                label=labels[component],
            )
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean weighted loss")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes.flat[len(names) :]:
        axis.remove()
    figure.suptitle("Training loss components across folds", fontsize=15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_thresholds(folds: pd.DataFrame, path: Path) -> bool:
    """Plot calibrated binary-rejection thresholds; return false if absent."""
    selected = folds.dropna(subset=["energy_threshold"])
    if selected.empty:
        return False
    order = (
        selected.groupby("display_name")["energy_threshold"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    figure, axis = plt.subplots(
        figsize=(max(9, 1.25 * len(order)), 5.5),
        constrained_layout=True,
    )
    rng = np.random.default_rng(1)
    for position, name in enumerate(order):
        values = selected.loc[
            selected["display_name"] == name, "energy_threshold"
        ].to_numpy(dtype=float)
        axis.scatter(
            np.full(len(values), position) + rng.uniform(-0.08, 0.08, len(values)),
            values,
            color="#ef8a3a",
            s=42,
            alpha=0.9,
            zorder=3,
        )
        axis.hlines(
            values.mean(),
            position - 0.22,
            position + 0.22,
            color="#2457a6",
            linewidth=3,
            zorder=4,
        )
    axis.set_xticks(np.arange(len(order)), [name.replace("_", " ") for name in order])
    axis.tick_params(axis="x", rotation=25)
    axis.set_xlabel("Binary loss formulation")
    axis.set_ylabel("Calibrated absolute-logit threshold")
    axis.set_title("Noise-rejection threshold stability across folds")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_loss_results(
    result_paths: list[Path],
    *,
    metric: str = "macro_f1",
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Read saved loss results and write tables and plots."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of: {', '.join(METRICS)}")
    runs = load_runs(result_paths)
    summary, folds, history = build_frames(runs)
    warnings = comparability_warnings(runs)
    if output_directory is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output_directory = Path("loss_comparison_results") / f"comparison_{stamp}"
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)

    ranking = ranking_text(summary, metric)
    report_parts = [f"Ranking by {metric}", "", ranking]
    if warnings:
        report_parts.extend(("", "Comparability warnings:", *[f"- {x}" for x in warnings]))
    report = "\n".join(report_parts) + "\n"
    print(report, end="")

    paths = {
        "summary": (output_directory / "loss_summary.csv").resolve(),
        "folds": (output_directory / "loss_folds.csv").resolve(),
        "history": (output_directory / "loss_training_history.csv").resolve(),
        "ranking": (output_directory / "loss_ranking.txt").resolve(),
        "performance_plot": (output_directory / "loss_performance.png").resolve(),
        "training_plot": (output_directory / "loss_training_curves.png").resolve(),
    }
    summary.sort_values(f"{metric}_mean", ascending=False).to_csv(
        paths["summary"], index=False
    )
    folds.to_csv(paths["folds"], index=False)
    history.to_csv(paths["history"], index=False)
    paths["ranking"].write_text(report, encoding="utf-8")
    plot_performance(summary, folds, metric, paths["performance_plot"])
    plot_training_curves(history, paths["training_plot"])

    threshold_path = (output_directory / "loss_thresholds.png").resolve()
    if plot_thresholds(folds, threshold_path):
        paths["threshold_plot"] = threshold_path
    print(f"Saved loss comparison plots to {output_directory.resolve()}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="*",
        help=(
            "Loss JSON files, directories, or glob patterns. Defaults to all "
            "losses_*.json files under loss_comparison_results/."
        ),
    )
    parser.add_argument("--metric", choices=METRICS, default="macro_f1")
    parser.add_argument(
        "--output-dir",
        help="New output directory; defaults to a timestamped comparison directory",
    )
    args = parser.parse_args()
    plot_loss_results(
        resolve_result_paths(args.results),
        metric=args.metric,
        output_directory=args.output_dir,
    )


if __name__ == "__main__":
    main()
