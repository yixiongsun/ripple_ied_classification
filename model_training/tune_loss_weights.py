"""Tune binary-rejection loss weights and margin with grouped cross-validation."""

from __future__ import annotations

import argparse
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import sklearn
import torch
from torch.utils.data import DataLoader

from .ablation_model import ABLATION_CONFIGS, AblationModel
from .ablation_test import (
    calculate_metrics,
    save_json,
    split_calibration_subjects,
    tune_energy_threshold,
)
from .compare_losses import LossConfig, evaluate_loss_model, train_loss_epoch
from .loaders import (
    LFPDataset,
    SubjectClassBalancedBatchSampler,
    load_samples,
    precompute_features,
)
from .train import predict_classes, replace_batch_norm_with_group_norm, set_seed, subject_kfold
from .train_full_dataset import resolve_device


SUMMARY_METRICS = ("accuracy", "weighted_f1", "macro_f1")


@dataclass(frozen=True)
class Candidate:
    """Named loss configuration used by the tuning experiment."""

    name: str
    loss: LossConfig
    varied_parameter: str


def _number_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _signature(loss: LossConfig) -> tuple[float, float, float, float, float]:
    margin = loss.noise_margin if loss.repulsion_weight > 0 else 0.0
    return (
        loss.contrast_weight,
        loss.contrast_temperature,
        loss.noise_weight,
        loss.repulsion_weight,
        margin,
    )


def _make_loss(
    name: str,
    contrast_weight: float,
    contrast_temperature: float,
    noise_weight: float,
    repulsion_weight: float,
    noise_margin: float,
) -> LossConfig:
    if repulsion_weight == 0:
        noise_margin = 0.0
    return LossConfig(
        name=name,
        contrast_weight=contrast_weight,
        contrast_temperature=contrast_temperature,
        noise_weight=noise_weight,
        repulsion_weight=repulsion_weight,
        noise_margin=noise_margin,
    )


def build_candidates(
    search: str,
    *,
    contrast_weights: list[float],
    noise_weights: list[float],
    repulsion_weights: list[float],
    noise_margins: list[float],
    base_contrast_weight: float,
    base_noise_weight: float,
    base_repulsion_weight: float,
    base_noise_margin: float,
    contrast_temperature: float,
    max_configurations: int,
) -> list[Candidate]:
    """Create a paired one-at-a-time screen or a full Cartesian grid."""
    if search not in {"one_at_a_time", "grid"}:
        raise ValueError("search must be one_at_a_time or grid")
    for name, values in (
        ("contrast_weights", contrast_weights),
        ("noise_weights", noise_weights),
        ("repulsion_weights", repulsion_weights),
        ("noise_margins", noise_margins),
    ):
        if not values:
            raise ValueError(f"{name} cannot be empty")
        if any(value < 0 for value in values):
            raise ValueError(f"{name} cannot contain negative values")

    base = _make_loss(
        "base",
        base_contrast_weight,
        contrast_temperature,
        base_noise_weight,
        base_repulsion_weight,
        base_noise_margin,
    )
    raw_candidates: list[Candidate] = [Candidate("base", base, "base")]

    if search == "one_at_a_time":
        dimensions = (
            ("contrast", contrast_weights),
            ("noise", noise_weights),
            ("repulsion", repulsion_weights),
            ("margin", noise_margins),
        )
        for parameter, values in dimensions:
            for value in values:
                settings = {
                    "contrast_weight": base.contrast_weight,
                    "noise_weight": base.noise_weight,
                    "repulsion_weight": base.repulsion_weight,
                    "noise_margin": base.noise_margin,
                }
                field = {
                    "contrast": "contrast_weight",
                    "noise": "noise_weight",
                    "repulsion": "repulsion_weight",
                    "margin": "noise_margin",
                }[parameter]
                settings[field] = value
                candidate_name = f"{parameter}_{_number_token(value)}"
                loss = _make_loss(
                    candidate_name,
                    settings["contrast_weight"],
                    contrast_temperature,
                    settings["noise_weight"],
                    settings["repulsion_weight"],
                    settings["noise_margin"],
                )
                raw_candidates.append(Candidate(candidate_name, loss, parameter))
    else:
        for contrast, noise, repulsion, margin in product(
            contrast_weights,
            noise_weights,
            repulsion_weights,
            noise_margins,
        ):
            name = (
                f"c{_number_token(contrast)}_n{_number_token(noise)}_"
                f"r{_number_token(repulsion)}_m{_number_token(margin)}"
            )
            raw_candidates.append(
                Candidate(
                    name,
                    _make_loss(
                        name,
                        contrast,
                        contrast_temperature,
                        noise,
                        repulsion,
                        margin,
                    ),
                    "grid",
                )
            )

    candidates = []
    seen = set()
    for candidate in raw_candidates:
        signature = _signature(candidate.loss)
        if signature not in seen:
            seen.add(signature)
            candidates.append(candidate)
    if len(candidates) > max_configurations:
        raise ValueError(
            f"Search expands to {len(candidates)} unique configurations, exceeding "
            f"--max-configurations={max_configurations}. Narrow the value lists or "
            "raise the limit explicitly."
        )
    return candidates


def _loader_options(
    num_workers: int,
    pin_memory: bool | None = None,
) -> dict[str, object]:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }


def prepare_fold_data(
    train_samples: list[dict],
    calibration_samples: list[dict],
    validation_samples: list[dict],
    *,
    use_global_features: bool,
) -> dict[str, object]:
    """Precompute expensive global features once for all loss candidates."""
    train_features, feature_mean, feature_std = precompute_features(
        train_samples, use_global_features
    )
    calibration_features, _, _ = precompute_features(
        calibration_samples, use_global_features
    )
    validation_features, _, _ = precompute_features(
        validation_samples, use_global_features
    )
    subject_ids = sorted({sample["animal_id"] for sample in train_samples})
    subject_to_index = {
        subject_id: index for index, subject_id in enumerate(subject_ids)
    }
    return {
        "train_samples": train_samples,
        "calibration_samples": calibration_samples,
        "validation_samples": validation_samples,
        "train_features": train_features,
        "calibration_features": calibration_features,
        "validation_features": validation_features,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "subject_to_index": subject_to_index,
        "use_global_features": use_global_features,
    }


def make_fold_loaders(
    prepared: dict[str, object],
    *,
    batch_size: int,
    subjects_per_batch: int,
    num_workers: int,
    seed: int,
    pin_memory: bool | None = None,
):
    """Create fresh paired loaders while reusing precomputed features."""
    common = {
        "feature_mean": prepared["feature_mean"],
        "feature_std": prepared["feature_std"],
        "use_global_features": prepared["use_global_features"],
    }
    train_dataset = LFPDataset(
        prepared["train_samples"],
        prepared["train_features"],
        subject_to_index=prepared["subject_to_index"],
        **common,
    )
    calibration_dataset = LFPDataset(
        prepared["calibration_samples"],
        prepared["calibration_features"],
        **common,
    )
    validation_dataset = LFPDataset(
        prepared["validation_samples"],
        prepared["validation_features"],
        **common,
    )
    sampler = SubjectClassBalancedBatchSampler(
        prepared["train_samples"],
        batch_size,
        subjects_per_batch=subjects_per_batch,
        seed=seed,
    )
    options = _loader_options(num_workers, pin_memory)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, **options)
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
        **options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        **options,
    )
    return train_loader, calibration_loader, validation_loader


def run_weight_tuning(
    dataset_path: str | Path,
    candidates: list[Candidate],
    *,
    architecture: str = "baseline",
    folds: int = 3,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    subjects_per_batch: int = 8,
    default_energy_threshold: float = 3.13,
    calibration_fraction: float = 0.2,
    num_workers: int = 0,
    seeds: list[int] | None = None,
    device_name: str = "auto",
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Evaluate loss candidates on paired folds and seeds."""
    if architecture not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown architecture: {architecture}")
    seeds = [0] if seeds is None else seeds
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a non-empty list of unique integers")

    started_at = datetime.now(timezone.utc)
    if output_path is None:
        stamp = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
        output_path = Path("loss_weight_results") / f"weights_{stamp}.json"
    elif Path(output_path).exists() and not overwrite:
        raise FileExistsError(f"Results already exist: {output_path}")

    device = resolve_device(device_name)
    dataset_path = Path(dataset_path)
    samples = load_samples(dataset_path)
    architecture_config = ABLATION_CONFIGS[architecture]
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

            for candidate in candidates:
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
                model = AblationModel(architecture_config, feat_dim=64, n_classes=1)
                replace_batch_norm_with_group_norm(model)
                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                history = []
                for epoch in range(1, epochs + 1):
                    stats = train_loss_epoch(
                        model,
                        train_loader,
                        optimizer,
                        device,
                        candidate.loss,
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
                    model, calibration_loader, device, candidate.loss
                )
                validation = evaluate_loss_model(
                    model, validation_loader, device, candidate.loss
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
                    "train_subjects": sorted(
                        {str(sample["animal_id"]) for sample in train_samples}
                    ),
                    "calibration_subjects": sorted(
                        {str(sample["animal_id"]) for sample in calibration_samples}
                    ),
                    "validation_subjects": sorted(
                        {str(sample["animal_id"]) for sample in validation_samples}
                    ),
                    "train_samples": len(train_samples),
                    "calibration_samples": len(calibration_samples),
                    "validation_samples": len(validation_samples),
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "energy_threshold": threshold,
                    "calibration_macro_f1": calibration_macro_f1,
                    "training_history": history,
                    **metrics,
                }
                results[candidate.name].append(result)
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
        summary[candidate.name] = {
            "loss_config": candidate.loss.to_dict(),
            "varied_parameter": candidate.varied_parameter,
        }
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
            f"{candidate.name:>20}: "
            f"macro_f1={metrics['macro_f1']['mean']:.4f} | "
            f"accuracy={metrics['accuracy']['mean']:.4f}"
        )

    finished_at = datetime.now(timezone.utc)
    dataset_stat = dataset_path.stat()
    payload = {
        "schema_version": 1,
        "experiment_type": "binary_loss_weight_tuning",
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
            "architecture": architecture,
            "folds": folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "subjects_per_batch": subjects_per_batch,
            "default_energy_threshold": default_energy_threshold,
            "calibration_fraction": calibration_fraction,
            "num_workers": num_workers,
            "seeds": seeds,
            "device_name": device_name,
        },
        "architecture_config": architecture_config.to_dict(),
        "candidates": {
            candidate.name: {
                "loss_config": candidate.loss.to_dict(),
                "varied_parameter": candidate.varied_parameter,
            }
            for candidate in candidates
        },
        "runtime": {
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
    print(f"Saved loss-weight tuning results to {saved_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset_arcsinh.pkl")
    parser.add_argument(
        "--search",
        choices=("one_at_a_time", "grid"),
        default="one_at_a_time",
        help="Paired isolated changes (default) or a Cartesian grid",
    )
    parser.add_argument("--contrast-weights", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5])
    parser.add_argument("--noise-weights", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--repulsion-weights", type=float, nargs="+", default=[0, 0.05, 0.1, 0.25])
    parser.add_argument("--noise-margins", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--base-contrast-weight", type=float, default=0.25)
    parser.add_argument("--base-noise-weight", type=float, default=0.5)
    parser.add_argument("--base-repulsion-weight", type=float, default=0.1)
    parser.add_argument("--base-noise-margin", type=float, default=0.5)
    parser.add_argument("--contrast-temperature", type=float, default=0.1)
    parser.add_argument("--max-configurations", type=int, default=64)
    parser.add_argument(
        "--architecture",
        choices=tuple(ABLATION_CONFIGS),
        default="baseline",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--subjects-per-batch", type=int, default=8)
    parser.add_argument("--energy-threshold", type=float, default=3.13)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    candidates = build_candidates(
        args.search,
        contrast_weights=args.contrast_weights,
        noise_weights=args.noise_weights,
        repulsion_weights=args.repulsion_weights,
        noise_margins=args.noise_margins,
        base_contrast_weight=args.base_contrast_weight,
        base_noise_weight=args.base_noise_weight,
        base_repulsion_weight=args.base_repulsion_weight,
        base_noise_margin=args.base_noise_margin,
        contrast_temperature=args.contrast_temperature,
        max_configurations=args.max_configurations,
    )
    print(f"Prepared {len(candidates)} unique loss configurations")
    run_weight_tuning(
        args.dataset,
        candidates,
        architecture=args.architecture,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
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
