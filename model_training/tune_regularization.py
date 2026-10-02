"""Screen regularization, batch composition, or capacity on frozen settings."""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter

import numpy as np
import sklearn
import torch

from .ablation_model import ABLATION_CONFIGS, AblationModel
from .ablation_test import (
    _git_revision,
    calculate_metrics,
    calculate_per_subject_metrics,
    save_json,
    split_calibration_subjects,
    tune_energy_threshold,
)
from .compare_losses import evaluate_loss_model, train_loss_epoch
from .experiment_config import (
    FROZEN_ARCHITECTURE,
    FROZEN_BATCH_SIZE,
    FROZEN_CALIBRATION_FRACTION,
    FROZEN_DROPOUT,
    FROZEN_EPOCHS,
    FROZEN_FEATURE_DIM,
    FROZEN_LEARNING_RATE,
    FROZEN_LOSS,
    FROZEN_MAX_GRAD_NORM,
    FROZEN_SCHEDULE,
    FROZEN_SUBJECTS_PER_BATCH,
    FROZEN_THRESHOLD_SELECTION,
    FROZEN_WEIGHT_DECAY,
)
from .loaders import load_samples
from .train import predict_classes, replace_batch_norm_with_group_norm, set_seed, subject_kfold
from .train_full_dataset import resolve_device
from .tune_loss_weights import make_fold_loaders, prepare_fold_data


SUMMARY_METRICS = ("accuracy", "weighted_f1", "macro_f1")


@dataclass(frozen=True)
class DropoutCandidate:
    """One paired training configuration."""

    name: str
    dropout: float
    max_grad_norm: float | None = None
    batch_size: int | None = None
    subjects_per_batch: int | None = None
    feat_dim: int | None = None
    calibration_fraction: float | None = None
    architecture: str | None = None


def build_dropout_candidates(values: list[float]) -> list[DropoutCandidate]:
    """Build uniquely named paired dropout candidates."""
    if not values or len(values) != len(set(values)):
        raise ValueError("dropout values must be non-empty and unique")
    if any(not 0 <= value < 1 for value in values):
        raise ValueError("dropout values must be in [0, 1)")
    return [
        DropoutCandidate(f"dropout_{value:g}", value)
        for value in values
    ]


def build_gradient_clip_candidates() -> list[DropoutCandidate]:
    """Build the planned disabled-versus-max-norm-1 clipping comparison."""
    return [
        DropoutCandidate("gradient_clip_disabled", FROZEN_DROPOUT, None),
        DropoutCandidate("gradient_clip_norm_1", FROZEN_DROPOUT, 1.0),
    ]


def build_batch_composition_candidates() -> list[DropoutCandidate]:
    """Build the three batch compositions specified for Phase 5."""
    return [
        DropoutCandidate(
            "batch_32_subjects_8",
            FROZEN_DROPOUT,
            FROZEN_MAX_GRAD_NORM,
            batch_size=32,
            subjects_per_batch=8,
        ),
        DropoutCandidate(
            "batch_64_subjects_8",
            FROZEN_DROPOUT,
            FROZEN_MAX_GRAD_NORM,
            batch_size=64,
            subjects_per_batch=8,
        ),
        DropoutCandidate(
            "batch_64_subjects_16",
            FROZEN_DROPOUT,
            FROZEN_MAX_GRAD_NORM,
            batch_size=64,
            subjects_per_batch=16,
        ),
    ]


def build_embedding_dimension_candidates() -> list[DropoutCandidate]:
    """Build the first Phase 6 representation-capacity screen."""
    return [
        DropoutCandidate(
            f"embedding_dim_{feat_dim}",
            FROZEN_DROPOUT,
            FROZEN_MAX_GRAD_NORM,
            batch_size=FROZEN_BATCH_SIZE,
            subjects_per_batch=FROZEN_SUBJECTS_PER_BATCH,
            feat_dim=feat_dim,
        )
        for feat_dim in (64, 128, 256)
    ]


def build_threshold_robustness_candidates() -> list[DropoutCandidate]:
    """Build the Phase 7 calibration-fraction sensitivity candidates."""
    return [
        DropoutCandidate(
            f"calibration_fraction_{fraction:g}",
            FROZEN_DROPOUT,
            FROZEN_MAX_GRAD_NORM,
            batch_size=FROZEN_BATCH_SIZE,
            subjects_per_batch=FROZEN_SUBJECTS_PER_BATCH,
            feat_dim=FROZEN_FEATURE_DIM,
            calibration_fraction=fraction,
        )
        for fraction in (0.15, 0.20, 0.25)
    ]


def build_final_confirmation_candidates() -> list[DropoutCandidate]:
    """Return only the original baseline and fully frozen selected model."""
    common = {
        "dropout": FROZEN_DROPOUT,
        "max_grad_norm": FROZEN_MAX_GRAD_NORM,
        "batch_size": FROZEN_BATCH_SIZE,
        "subjects_per_batch": FROZEN_SUBJECTS_PER_BATCH,
        "feat_dim": FROZEN_FEATURE_DIM,
        "calibration_fraction": FROZEN_CALIBRATION_FRACTION,
    }
    return [
        DropoutCandidate(
            "original_full_baseline",
            architecture="baseline",
            **common,
        ),
        DropoutCandidate(
            "frozen_compact_model",
            architecture=FROZEN_ARCHITECTURE,
            **common,
        ),
    ]


def _bootstrap_interval(values: np.ndarray, *, seed: int = 0) -> list[float]:
    """Return a deterministic percentile interval for a mean."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _exact_sign_flip_p_value(values: np.ndarray) -> float:
    """Two-sided exact paired sign-flip p-value for a mean difference."""
    count = 1 << len(values)
    patterns = np.arange(count, dtype=np.uint64)[:, None]
    bits = (patterns >> np.arange(len(values), dtype=np.uint64)) & 1
    signs = bits.astype(np.float64) * 2.0 - 1.0
    permuted = np.abs((signs * values).mean(axis=1))
    observed = abs(float(values.mean()))
    return float(np.mean(permuted >= observed - 1e-15))


def build_final_paired_analysis(
    results: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    """Summarize paired final-confirmation differences with caveats."""
    baseline = sorted(
        results["original_full_baseline"],
        key=lambda item: (item["repeat_seed"], item["fold"]),
    )
    selected = sorted(
        results["frozen_compact_model"],
        key=lambda item: (item["repeat_seed"], item["fold"]),
    )
    baseline_keys = [(item["repeat_seed"], item["fold"]) for item in baseline]
    selected_keys = [(item["repeat_seed"], item["fold"]) for item in selected]
    if baseline_keys != selected_keys:
        raise ValueError("Final confirmation results are not correctly paired")

    metric_results = {}
    for metric in SUMMARY_METRICS:
        differences = np.asarray(
            [
                float(selected_item[metric]) - float(baseline_item[metric])
                for baseline_item, selected_item in zip(baseline, selected)
            ],
            dtype=float,
        )
        seed_means = np.asarray(
            [
                differences[
                    np.asarray([key[0] == seed for key in baseline_keys], dtype=bool)
                ].mean()
                for seed in sorted({key[0] for key in baseline_keys})
            ]
        )
        metric_results[metric] = {
            "direction": "frozen_compact_model minus original_full_baseline",
            "pair_count": len(differences),
            "mean_difference": float(differences.mean()),
            "median_difference": float(np.median(differences)),
            "standard_deviation": float(differences.std(ddof=1)),
            "wins": int((differences > 0).sum()),
            "ties": int((differences == 0).sum()),
            "losses": int((differences < 0).sum()),
            "paired_bootstrap_95_interval": _bootstrap_interval(differences),
            "fold_pair_exact_sign_flip_p_value": _exact_sign_flip_p_value(
                differences
            ),
            "seed_mean_differences": [float(value) for value in seed_means],
            "seed_block_bootstrap_95_interval": _bootstrap_interval(
                seed_means, seed=1
            ),
            "seed_level_exact_sign_flip_p_value": _exact_sign_flip_p_value(
                seed_means
            ),
        }
    return {
        "metrics": metric_results,
        "qualification": (
            "Cross-validation folds and repeated seeds reuse subjects and are not "
            "independent experimental units. Fold-pair intervals and tests are "
            "descriptive; seed-level inference has only three units and very low "
            "power. Interpret effect sizes and class-level behavior alongside them."
        ),
    }


def run_dropout_tuning(
    dataset_path: str | Path,
    candidates: list[DropoutCandidate],
    *,
    stage: str = "dropout",
    folds: int = 3,
    epochs: int = FROZEN_EPOCHS,
    batch_size: int = FROZEN_BATCH_SIZE,
    subjects_per_batch: int = FROZEN_SUBJECTS_PER_BATCH,
    default_energy_threshold: float = 3.13,
    calibration_fraction: float = FROZEN_CALIBRATION_FRACTION,
    num_workers: int = 0,
    seeds: list[int] | None = None,
    device_name: str = "auto",
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Evaluate dropout candidates on paired grouped folds and seeds."""
    if stage not in {
        "dropout",
        "gradient_clipping",
        "batch_composition",
        "embedding_dimension",
        "threshold_robustness",
        "final_confirmation",
    }:
        raise ValueError(
            "stage must be dropout, gradient_clipping, batch_composition, "
            "embedding_dimension, threshold_robustness, or final_confirmation"
        )
    if FROZEN_SCHEDULE != "constant":
        raise ValueError("Phase 4 expects the frozen constant schedule")
    if not candidates:
        raise ValueError("At least one dropout candidate is required")
    seeds = [0] if seeds is None else seeds
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a non-empty list of unique integers")

    started_at = datetime.now(timezone.utc)
    if output_path is None:
        stamp = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
        output_path = Path("regularization_results") / f"{stage}_{stamp}.json"
    elif Path(output_path).exists() and not overwrite:
        raise FileExistsError(f"Results already exist: {output_path}")
    output_path = Path(output_path)
    partial_output_path = output_path.with_name(
        f"{output_path.stem}.partial{output_path.suffix}"
    )

    device = resolve_device(device_name)
    dataset_path = Path(dataset_path)
    samples = load_samples(dataset_path)
    architecture_config = ABLATION_CONFIGS[FROZEN_ARCHITECTURE]
    candidate_architectures = {
        candidate.name: ABLATION_CONFIGS[
            candidate.architecture or FROZEN_ARCHITECTURE
        ]
        for candidate in candidates
    }
    if len(
        {config.use_global_features for config in candidate_architectures.values()}
    ) != 1:
        raise ValueError("Candidates must agree on global-feature preprocessing")
    results: dict[str, list[dict[str, object]]] = {
        candidate.name: [] for candidate in candidates
    }

    for repeat_seed in seeds:
        outer_splits = subject_kfold(samples, k=folds, seed=repeat_seed)
        for fold_index, (outer_training, validation_samples) in enumerate(
            outer_splits, start=1
        ):
            training_seed = repeat_seed * 10_000 + fold_index
            print(
                f"\n######## seed {repeat_seed} | fold {fold_index}/{folds} | "
                "preparing features ########"
            )
            validation_subject_names = sorted(
                {str(sample["animal_id"]) for sample in validation_samples}
            )
            shared_split = None
            shared_prepared = None
            if stage != "threshold_robustness":
                shared_split = split_calibration_subjects(
                    outer_training, calibration_fraction, training_seed
                )
                shared_prepared = prepare_fold_data(
                    shared_split[0],
                    shared_split[1],
                    validation_samples,
                    use_global_features=architecture_config.use_global_features,
                )

            for candidate in candidates:
                candidate_started = perf_counter()
                print(f"\n===== {candidate.name} =====")
                set_seed(training_seed)
                candidate_batch_size = candidate.batch_size or batch_size
                candidate_subjects_per_batch = (
                    candidate.subjects_per_batch or subjects_per_batch
                )
                candidate_feat_dim = candidate.feat_dim or FROZEN_FEATURE_DIM
                candidate_architecture = (
                    candidate.architecture or FROZEN_ARCHITECTURE
                )
                candidate_calibration_fraction = (
                    candidate.calibration_fraction or calibration_fraction
                )
                if shared_split is None or shared_prepared is None:
                    train_samples, calibration_samples = split_calibration_subjects(
                        outer_training,
                        candidate_calibration_fraction,
                        training_seed,
                    )
                    prepared = prepare_fold_data(
                        train_samples,
                        calibration_samples,
                        validation_samples,
                        use_global_features=architecture_config.use_global_features,
                    )
                else:
                    train_samples, calibration_samples = shared_split
                    prepared = shared_prepared
                train_loader, calibration_loader, validation_loader = (
                    make_fold_loaders(
                        prepared,
                        batch_size=candidate_batch_size,
                        subjects_per_batch=candidate_subjects_per_batch,
                        num_workers=num_workers,
                        seed=training_seed,
                    )
                )
                model = AblationModel(
                    candidate_architectures[candidate.name],
                    feat_dim=candidate_feat_dim,
                    n_classes=1,
                    dropout=candidate.dropout,
                )
                replace_batch_norm_with_group_norm(model)
                model.to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=FROZEN_LEARNING_RATE,
                    weight_decay=FROZEN_WEIGHT_DECAY,
                )
                history = []
                for epoch in range(1, epochs + 1):
                    stats = train_loss_epoch(
                        model,
                        train_loader,
                        optimizer,
                        device,
                        FROZEN_LOSS,
                        max_grad_norm=candidate.max_grad_norm,
                    )
                    history.append(
                        {
                            "epoch": epoch,
                            **{name: float(value) for name, value in stats.items()},
                        }
                    )
                    print(
                        f"Epoch {epoch:03d}/{epochs:03d} | "
                        f"total={stats['total']:.4f} | "
                        f"supervised={stats['supervised']:.4f} | "
                        f"contrast={stats['weighted_contrast']:.4f} | "
                        f"noise={stats['weighted_noise']:.4f} | "
                        f"repulsion={stats['weighted_repulsion']:.4f}"
                    )

                calibration = evaluate_loss_model(
                    model, calibration_loader, device, FROZEN_LOSS
                )
                validation = evaluate_loss_model(
                    model, validation_loader, device, FROZEN_LOSS
                )
                threshold, calibration_macro_f1 = tune_energy_threshold(
                    calibration["probabilities"],
                    calibration["labels"],
                    calibration["energies"],
                    default_energy_threshold,
                )
                predictions = predict_classes(
                    validation["probabilities"], validation["energies"], threshold
                )
                metrics = calculate_metrics(validation["labels"], predictions)
                result = {
                    "repeat_seed": repeat_seed,
                    "fold": fold_index,
                    "training_seed": training_seed,
                    "dropout": candidate.dropout,
                    "max_grad_norm": candidate.max_grad_norm,
                    "batch_size": candidate_batch_size,
                    "subjects_per_batch": candidate_subjects_per_batch,
                    "feat_dim": candidate_feat_dim,
                    "architecture": candidate_architecture,
                    "calibration_fraction": candidate_calibration_fraction,
                    "train_subjects": sorted(
                        {str(sample["animal_id"]) for sample in train_samples}
                    ),
                    "calibration_subjects": sorted(
                        {str(sample["animal_id"]) for sample in calibration_samples}
                    ),
                    "validation_subjects": validation_subject_names,
                    "train_samples": len(train_samples),
                    "calibration_samples": len(calibration_samples),
                    "validation_samples": len(validation_samples),
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "energy_threshold": threshold,
                    "calibration_macro_f1": calibration_macro_f1,
                    "duration_seconds": perf_counter() - candidate_started,
                    "training_history": history,
                    "per_subject_metrics": calculate_per_subject_metrics(
                        validation["labels"],
                        predictions,
                        validation["subjects"],
                        validation_subject_names,
                    ),
                    **metrics,
                }
                results[candidate.name].append(result)
                completed_runs = sum(len(items) for items in results.values())
                save_json(
                    {
                        "schema_version": 1,
                        "experiment_type": f"{stage}_tuning_partial",
                        "started_at_utc": started_at.isoformat(),
                        "completed_runs": completed_runs,
                        "expected_runs": len(seeds) * folds * len(candidates),
                        "candidates": {
                            item.name: asdict(item) for item in candidates
                        },
                        "fold_results": results,
                    },
                    partial_output_path,
                    overwrite=True,
                )
                print(
                    f"threshold={threshold:.4f} | "
                    f"accuracy={metrics['accuracy']:.4f} | "
                    f"macro_f1={metrics['macro_f1']:.4f}"
                )
                del model, optimizer, train_loader, calibration_loader, validation_loader
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary = {}
    for candidate in candidates:
        candidate_results = results[candidate.name]
        summary[candidate.name] = asdict(candidate)
        for metric in SUMMARY_METRICS:
            values = [float(result[metric]) for result in candidate_results]
            summary[candidate.name][metric] = {
                "mean": mean(values),
                "standard_deviation": pstdev(values),
            }
        thresholds = [
            float(result["energy_threshold"]) for result in candidate_results
        ]
        summary[candidate.name]["energy_threshold"] = {
            "mean": mean(thresholds),
            "median": median(thresholds),
            "standard_deviation": pstdev(thresholds),
            "minimum": min(thresholds),
            "maximum": max(thresholds),
        }

    ranking = sorted(
        candidates,
        key=lambda item: summary[item.name]["macro_f1"]["mean"],
        reverse=True,
    )
    print("\nMean cross-validation metrics")
    for candidate in ranking:
        metrics = summary[candidate.name]
        print(
            f"{candidate.name:>16}: "
            f"macro_f1={metrics['macro_f1']['mean']:.4f} | "
            f"accuracy={metrics['accuracy']['mean']:.4f}"
        )

    finished_at = datetime.now(timezone.utc)
    dataset_stat = dataset_path.stat()
    payload = {
        "schema_version": 1,
        "experiment_type": f"{stage}_tuning",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "dataset": {
            "path": str(dataset_path.resolve()),
            "size_bytes": dataset_stat.st_size,
            "modified_at_utc": datetime.fromtimestamp(
                dataset_stat.st_mtime, timezone.utc
            ).isoformat(),
            "sample_count": len(samples),
            "subject_count": len({sample["animal_id"] for sample in samples}),
        },
        "parameters": {
            "architecture": FROZEN_ARCHITECTURE,
            "learning_rate": FROZEN_LEARNING_RATE,
            "weight_decay": FROZEN_WEIGHT_DECAY,
            "schedule": FROZEN_SCHEDULE,
            "stage": stage,
            "folds": folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "subjects_per_batch": subjects_per_batch,
            "calibration_fraction": calibration_fraction,
            "threshold_selection_rule": FROZEN_THRESHOLD_SELECTION,
            "threshold_tie_break": "closest to default energy threshold",
            "false_positive_constraint": None,
            "num_workers": num_workers,
            "seeds": seeds,
            "device_name": device_name,
            "optimizer": "Adam",
        },
        "architecture_config": architecture_config.to_dict(),
        "architecture_configs": {
            name: config.to_dict()
            for name, config in candidate_architectures.items()
        },
        "loss_config": FROZEN_LOSS.to_dict(),
        "candidates": {item.name: asdict(item) for item in candidates},
        "runtime": {
            "command_line": [sys.executable, *sys.argv],
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "resolved_device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "class_order": ["ripple", "ied", "noise"],
        "fold_results": results,
        "summary": summary,
    }
    if stage == "final_confirmation":
        payload["paired_analysis"] = build_final_paired_analysis(results)
    saved_path = save_json(payload, output_path, overwrite)
    print(f"Saved {stage} tuning results to {saved_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset_arcsinh.pkl")
    parser.add_argument(
        "--stage",
        choices=(
            "dropout",
            "gradient_clipping",
            "batch_composition",
            "embedding_dimension",
            "threshold_robustness",
            "final_confirmation",
        ),
        default="dropout",
    )
    parser.add_argument("--dropouts", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    parser.add_argument(
        "--configurations",
        nargs="+",
        help=(
            "Optional candidate names to run, useful for restricting a "
            "confirmation to selected configurations."
        ),
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=FROZEN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FROZEN_BATCH_SIZE)
    parser.add_argument(
        "--subjects-per-batch", type=int, default=FROZEN_SUBJECTS_PER_BATCH
    )
    parser.add_argument("--energy-threshold", type=float, default=3.13)
    parser.add_argument(
        "--calibration-fraction", type=float, default=FROZEN_CALIBRATION_FRACTION
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.stage == "gradient_clipping":
        candidates = build_gradient_clip_candidates()
    elif args.stage == "batch_composition":
        candidates = build_batch_composition_candidates()
    elif args.stage == "embedding_dimension":
        candidates = build_embedding_dimension_candidates()
    elif args.stage == "threshold_robustness":
        candidates = build_threshold_robustness_candidates()
    elif args.stage == "final_confirmation":
        candidates = build_final_confirmation_candidates()
    else:
        candidates = build_dropout_candidates(args.dropouts)
    if args.configurations:
        requested = set(args.configurations)
        available = {candidate.name for candidate in candidates}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(
                f"Unknown configurations: {unknown}; available: {sorted(available)}"
            )
        candidates = [
            candidate for candidate in candidates if candidate.name in requested
        ]
    print(f"Prepared {len(candidates)} {args.stage} configurations")
    run_dropout_tuning(
        args.dataset,
        candidates,
        stage=args.stage,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        subjects_per_batch=args.subjects_per_batch,
        default_energy_threshold=args.energy_threshold,
        calibration_fraction=args.calibration_fraction,
        num_workers=args.workers,
        seeds=args.seeds,
        device_name=args.device,
        output_path=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
