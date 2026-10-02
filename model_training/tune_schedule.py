"""Screen training schedules and holdout-based early stopping."""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
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
    FROZEN_EPOCHS,
    FROZEN_LEARNING_RATE,
    FROZEN_LOSS,
    FROZEN_SUBJECTS_PER_BATCH,
    FROZEN_WEIGHT_DECAY,
)
from .loaders import load_samples, make_evaluation_loader
from .train import predict_classes, replace_batch_norm_with_group_norm, set_seed, subject_kfold
from .train_full_dataset import resolve_device
from .tune_loss_weights import make_fold_loaders, prepare_fold_data


SUMMARY_METRICS = ("accuracy", "weighted_f1", "macro_f1")


@dataclass(frozen=True)
class ScheduleCandidate:
    """One validation-independent training schedule."""

    name: str
    scheduler: str
    patience: int | None = None


SCHEDULE_CANDIDATES = (
    ScheduleCandidate("constant_40", "constant"),
    ScheduleCandidate("cosine_40", "cosine"),
    ScheduleCandidate("early_stop_patience_5", "constant", 5),
    ScheduleCandidate("early_stop_patience_10", "constant", 10),
)


def split_monitor_subjects(
    samples: list[dict], fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Reserve complete development subjects solely for epoch selection."""
    return split_calibration_subjects(samples, fraction, seed)


def _monitor_score(model, loader, device) -> tuple[float, float]:
    """Return monitor macro F1 and its temporary holdout-only threshold."""
    evaluation = evaluate_loss_model(model, loader, device, FROZEN_LOSS)
    threshold, _ = tune_energy_threshold(
        evaluation["probabilities"],
        evaluation["labels"],
        evaluation["energies"],
        3.13,
    )
    predictions = predict_classes(
        evaluation["probabilities"], evaluation["energies"], threshold
    )
    metrics = calculate_metrics(evaluation["labels"], predictions)
    return float(metrics["macro_f1"]), float(threshold)


def run_schedule_tuning(
    dataset_path: str | Path,
    candidates: list[ScheduleCandidate],
    *,
    folds: int = 3,
    epochs: int = FROZEN_EPOCHS,
    batch_size: int = FROZEN_BATCH_SIZE,
    subjects_per_batch: int = FROZEN_SUBJECTS_PER_BATCH,
    default_energy_threshold: float = 3.13,
    calibration_fraction: float = FROZEN_CALIBRATION_FRACTION,
    monitor_fraction: float = 0.2,
    num_workers: int = 0,
    seeds: list[int] | None = None,
    device_name: str = "auto",
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Evaluate schedule candidates without using outer validation for epochs."""
    if not candidates or len({item.name for item in candidates}) != len(candidates):
        raise ValueError("schedule candidates must be non-empty and uniquely named")
    seeds = [0] if seeds is None else seeds
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a non-empty list of unique integers")
    if not 0 < monitor_fraction < 1:
        raise ValueError("monitor_fraction must be between 0 and 1")

    started_at = datetime.now(timezone.utc)
    if output_path is None:
        stamp = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
        output_path = Path("experiment_results/training_schedule") / f"schedules_{stamp}.json"
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
    results: dict[str, list[dict[str, object]]] = {
        candidate.name: [] for candidate in candidates
    }

    for repeat_seed in seeds:
        outer_splits = subject_kfold(samples, k=folds, seed=repeat_seed)
        for fold_index, (outer_training, validation_samples) in enumerate(
            outer_splits, start=1
        ):
            training_seed = repeat_seed * 10_000 + fold_index
            development_samples, calibration_samples = split_calibration_subjects(
                outer_training, calibration_fraction, training_seed
            )
            train_samples, monitor_samples = split_monitor_subjects(
                development_samples, monitor_fraction, training_seed + 1_000_000
            )
            print(
                f"\n######## seed {repeat_seed} | fold {fold_index}/{folds} | "
                "preparing features ########"
            )
            prepared = prepare_fold_data(
                train_samples,
                calibration_samples,
                validation_samples,
                use_global_features=architecture_config.use_global_features,
            )
            validation_subject_names = sorted(
                {str(sample["animal_id"]) for sample in validation_samples}
            )

            for candidate in candidates:
                candidate_started = perf_counter()
                print(f"\n===== {candidate.name} =====")
                set_seed(training_seed)
                train_loader, calibration_loader, validation_loader = (
                    make_fold_loaders(
                        prepared,
                        batch_size=batch_size,
                        subjects_per_batch=subjects_per_batch,
                        num_workers=num_workers,
                        seed=training_seed,
                    )
                )
                monitor_loader = make_evaluation_loader(
                    monitor_samples,
                    batch_size,
                    feature_mean=prepared["feature_mean"],
                    feature_std=prepared["feature_std"],
                    use_global_features=architecture_config.use_global_features,
                    num_workers=num_workers,
                )
                model = AblationModel(architecture_config, feat_dim=64, n_classes=1)
                replace_batch_norm_with_group_norm(model)
                model.to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=FROZEN_LEARNING_RATE,
                    weight_decay=FROZEN_WEIGHT_DECAY,
                )
                scheduler = (
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=epochs
                    )
                    if candidate.scheduler == "cosine"
                    else None
                )
                history = []
                best_score = -np.inf
                best_epoch = 0
                best_state = None
                stale_epochs = 0

                for epoch in range(1, epochs + 1):
                    learning_rate = float(optimizer.param_groups[0]["lr"])
                    stats = train_loss_epoch(
                        model, train_loader, optimizer, device, FROZEN_LOSS
                    )
                    monitor_macro_f1, monitor_threshold = _monitor_score(
                        model, monitor_loader, device
                    )
                    history.append(
                        {
                            "epoch": epoch,
                            "learning_rate": learning_rate,
                            "monitor_macro_f1": monitor_macro_f1,
                            "monitor_energy_threshold": monitor_threshold,
                            **{name: float(value) for name, value in stats.items()},
                        }
                    )
                    print(
                        f"Epoch {epoch:03d}/{epochs:03d} | "
                        f"lr={learning_rate:.6g} | total={stats['total']:.4f} | "
                        f"monitor_macro_f1={monitor_macro_f1:.4f}"
                    )
                    if scheduler is not None:
                        scheduler.step()
                    if candidate.patience is not None:
                        if monitor_macro_f1 > best_score + 1e-6:
                            best_score = monitor_macro_f1
                            best_epoch = epoch
                            best_state = {
                                name: value.detach().cpu().clone()
                                for name, value in model.state_dict().items()
                            }
                            stale_epochs = 0
                        else:
                            stale_epochs += 1
                        if stale_epochs >= candidate.patience:
                            print(
                                f"Early stop at epoch {epoch}; "
                                f"restoring epoch {best_epoch}"
                            )
                            break

                if candidate.patience is not None:
                    if best_state is None:
                        raise RuntimeError("Early stopping did not capture a model state")
                    model.load_state_dict(best_state)
                    selected_epoch = best_epoch
                else:
                    selected_epoch = history[-1]["epoch"]

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
                    "selected_epoch": selected_epoch,
                    "epochs_trained": len(history),
                    "train_subjects": sorted(
                        {str(sample["animal_id"]) for sample in train_samples}
                    ),
                    "monitor_subjects": sorted(
                        {str(sample["animal_id"]) for sample in monitor_samples}
                    ),
                    "calibration_subjects": sorted(
                        {str(sample["animal_id"]) for sample in calibration_samples}
                    ),
                    "validation_subjects": validation_subject_names,
                    "train_samples": len(train_samples),
                    "monitor_samples": len(monitor_samples),
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
                        "experiment_type": "schedule_tuning_partial",
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
                    f"selected_epoch={selected_epoch} | threshold={threshold:.4f} | "
                    f"accuracy={metrics['accuracy']:.4f} | "
                    f"macro_f1={metrics['macro_f1']:.4f}"
                )
                del model, optimizer, scheduler, train_loader
                del monitor_loader, calibration_loader, validation_loader
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary = {}
    for candidate in candidates:
        candidate_results = results[candidate.name]
        summary[candidate.name] = asdict(candidate)
        summary[candidate.name]["selected_epoch"] = {
            "mean": mean(result["selected_epoch"] for result in candidate_results),
            "standard_deviation": pstdev(
                result["selected_epoch"] for result in candidate_results
            ),
        }
        for metric in SUMMARY_METRICS:
            values = [float(result[metric]) for result in candidate_results]
            summary[candidate.name][metric] = {
                "mean": mean(values),
                "standard_deviation": pstdev(values),
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
            f"{candidate.name:>24}: "
            f"macro_f1={metrics['macro_f1']['mean']:.4f} | "
            f"selected_epoch={metrics['selected_epoch']['mean']:.1f}"
        )

    finished_at = datetime.now(timezone.utc)
    dataset_stat = dataset_path.stat()
    payload = {
        "schema_version": 1,
        "experiment_type": "schedule_tuning",
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
            "folds": folds,
            "maximum_epochs": epochs,
            "batch_size": batch_size,
            "subjects_per_batch": subjects_per_batch,
            "calibration_fraction": calibration_fraction,
            "monitor_fraction": monitor_fraction,
            "num_workers": num_workers,
            "seeds": seeds,
            "device_name": device_name,
            "optimizer": "Adam",
        },
        "architecture_config": architecture_config.to_dict(),
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
    saved_path = save_json(payload, output_path, overwrite)
    print(f"Saved schedule tuning results to {saved_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset_arcsinh.pkl")
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=tuple(item.name for item in SCHEDULE_CANDIDATES),
        default=[item.name for item in SCHEDULE_CANDIDATES],
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
    parser.add_argument("--monitor-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    by_name = {item.name: item for item in SCHEDULE_CANDIDATES}
    candidates = [by_name[name] for name in args.candidates]
    print(f"Prepared {len(candidates)} schedule configurations")
    run_schedule_tuning(
        args.dataset,
        candidates,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        subjects_per_batch=args.subjects_per_batch,
        default_energy_threshold=args.energy_threshold,
        calibration_fraction=args.calibration_fraction,
        monitor_fraction=args.monitor_fraction,
        num_workers=args.workers,
        seeds=args.seeds,
        device_name=args.device,
        output_path=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
