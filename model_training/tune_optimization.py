"""Screen learning-rate and weight-decay pairs on the frozen architecture."""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
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
    FROZEN_LOSS,
    FROZEN_SUBJECTS_PER_BATCH,
)
from .loaders import load_samples
from .train import predict_classes, replace_batch_norm_with_group_norm, set_seed, subject_kfold
from .train_full_dataset import resolve_device
from .tune_loss_weights import make_fold_loaders, prepare_fold_data


SUMMARY_METRICS = ("accuracy", "weighted_f1", "macro_f1")


@dataclass(frozen=True)
class OptimizationCandidate:
    """One Adam learning-rate and weight-decay configuration."""

    name: str
    learning_rate: float
    weight_decay: float


def build_candidates(
    learning_rates: list[float],
    weight_decays: list[float],
) -> list[OptimizationCandidate]:
    """Build the paired Cartesian optimization screen."""
    if not learning_rates or not weight_decays:
        raise ValueError("learning_rates and weight_decays cannot be empty")
    if any(value <= 0 for value in learning_rates):
        raise ValueError("learning rates must be greater than zero")
    if any(value < 0 for value in weight_decays):
        raise ValueError("weight decay cannot be negative")
    pairs = list(product(learning_rates, weight_decays))
    if len(pairs) != len(set(pairs)):
        raise ValueError("learning-rate and weight-decay pairs must be unique")
    return [
        OptimizationCandidate(
            name=f"lr_{learning_rate:g}_wd_{weight_decay:g}",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        for learning_rate, weight_decay in pairs
    ]


def parse_candidate_pairs(values: list[str]) -> list[OptimizationCandidate]:
    """Parse exact ``learning_rate:weight_decay`` pairs from the CLI."""
    pairs = []
    for value in values:
        try:
            learning_rate_text, weight_decay_text = value.split(":", maxsplit=1)
            learning_rate = float(learning_rate_text)
            weight_decay = float(weight_decay_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid optimization pair {value!r}; expected learning_rate:weight_decay"
            ) from error
        pairs.append((learning_rate, weight_decay))
    if len(pairs) != len(set(pairs)):
        raise ValueError("optimization pairs must be unique")
    candidates = []
    for learning_rate, weight_decay in pairs:
        if learning_rate <= 0:
            raise ValueError("learning rates must be greater than zero")
        if weight_decay < 0:
            raise ValueError("weight decay cannot be negative")
        candidates.append(
            OptimizationCandidate(
                name=f"lr_{learning_rate:g}_wd_{weight_decay:g}",
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        )
    return candidates


def run_optimization_tuning(
    dataset_path: str | Path,
    candidates: list[OptimizationCandidate],
    *,
    folds: int = 3,
    epochs: int = FROZEN_EPOCHS,
    batch_size: int = FROZEN_BATCH_SIZE,
    subjects_per_batch: int = FROZEN_SUBJECTS_PER_BATCH,
    default_energy_threshold: float = 3.13,
    calibration_fraction: float = FROZEN_CALIBRATION_FRACTION,
    num_workers: int = 0,
    pin_memory: bool = True,
    seeds: list[int] | None = None,
    device_name: str = "auto",
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Evaluate paired optimization candidates on grouped folds and seeds."""
    if not candidates:
        raise ValueError("At least one optimization candidate is required")
    seeds = [0] if seeds is None else seeds
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a non-empty list of unique integers")

    started_at = datetime.now(timezone.utc)
    if output_path is None:
        stamp = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
        output_path = Path("optimization_results") / f"optimization_{stamp}.json"
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
            train_samples, calibration_samples = split_calibration_subjects(
                outer_training,
                calibration_fraction,
                training_seed,
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
                        pin_memory=pin_memory,
                    )
                )
                model = AblationModel(architecture_config, feat_dim=64, n_classes=1)
                replace_batch_norm_with_group_norm(model)
                model.to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=candidate.learning_rate,
                    weight_decay=candidate.weight_decay,
                )
                history = []
                for epoch in range(1, epochs + 1):
                    stats = train_loss_epoch(
                        model,
                        train_loader,
                        optimizer,
                        device,
                        FROZEN_LOSS,
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
                    validation["probabilities"],
                    validation["energies"],
                    threshold,
                )
                metrics = calculate_metrics(validation["labels"], predictions)
                result = {
                    "repeat_seed": repeat_seed,
                    "fold": fold_index,
                    "training_seed": training_seed,
                    "learning_rate": candidate.learning_rate,
                    "weight_decay": candidate.weight_decay,
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
                        "experiment_type": "optimization_tuning_partial",
                        "started_at_utc": started_at.isoformat(),
                        "completed_runs": completed_runs,
                        "expected_runs": len(seeds) * folds * len(candidates),
                        "parameters": {
                            "architecture": FROZEN_ARCHITECTURE,
                            "folds": folds,
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "subjects_per_batch": subjects_per_batch,
                            "calibration_fraction": calibration_fraction,
                            "pin_memory": pin_memory,
                            "seeds": seeds,
                            "optimizer": "Adam",
                            "scheduler": None,
                        },
                        "architecture_config": architecture_config.to_dict(),
                        "loss_config": FROZEN_LOSS.to_dict(),
                        "candidates": {
                            item.name: asdict(item) for item in candidates
                        },
                        "class_order": ["ripple", "ied", "noise"],
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

    ranking = sorted(
        candidates,
        key=lambda candidate: summary[candidate.name]["macro_f1"]["mean"],
        reverse=True,
    )
    print("\nMean cross-validation metrics")
    for candidate in ranking:
        metrics = summary[candidate.name]
        print(
            f"{candidate.name:>24}: "
            f"macro_f1={metrics['macro_f1']['mean']:.4f} | "
            f"accuracy={metrics['accuracy']['mean']:.4f}"
        )

    finished_at = datetime.now(timezone.utc)
    dataset_stat = dataset_path.stat()
    payload = {
        "schema_version": 1,
        "experiment_type": "optimization_tuning",
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
            "folds": folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "subjects_per_batch": subjects_per_batch,
            "default_energy_threshold": default_energy_threshold,
            "calibration_fraction": calibration_fraction,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "seeds": seeds,
            "device_name": device_name,
            "optimizer": "Adam",
            "scheduler": None,
        },
        "architecture_config": architecture_config.to_dict(),
        "loss_config": FROZEN_LOSS.to_dict(),
        "candidates": {
            candidate.name: asdict(candidate) for candidate in candidates
        },
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
    print(f"Saved optimization tuning results to {saved_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset_arcsinh.pkl")
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3]
    )
    parser.add_argument(
        "--weight-decays", type=float, nargs="+", default=[0.0, 1e-4, 1e-3]
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        help=(
            "Exact learning_rate:weight_decay pairs. When supplied, this "
            "overrides the Cartesian --learning-rates/--weight-decays screen."
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
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable page-locked loader buffers when system commit is constrained",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    candidates = (
        parse_candidate_pairs(args.pairs)
        if args.pairs
        else build_candidates(args.learning_rates, args.weight_decays)
    )
    print(f"Prepared {len(candidates)} optimization configurations")
    run_optimization_tuning(
        args.dataset,
        candidates,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        subjects_per_batch=args.subjects_per_batch,
        default_energy_threshold=args.energy_threshold,
        calibration_fraction=args.calibration_fraction,
        num_workers=args.workers,
        pin_memory=not args.no_pin_memory,
        seeds=args.seeds,
        device_name=args.device,
        output_path=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
