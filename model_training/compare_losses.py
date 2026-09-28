"""Compare seven loss formulations with paired subject-wise cross-validation."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import sklearn
import torch
import torch.nn.functional as F

from .ablation_model import ABLATION_CONFIGS, AblationModel
from .ablation_test import (
    calculate_metrics,
    save_json,
    split_calibration_subjects,
    tune_energy_threshold,
)
from .loaders import (
    load_samples,
    make_evaluation_loader,
    make_subject_class_balanced_loader,
)
from .train import (
    predict_classes,
    replace_batch_norm_with_group_norm,
    set_seed,
    subject_kfold,
    supervised_contrastive_loss,
)
from .train_full_dataset import resolve_device


CLASS_NAMES = ("ripple", "ied", "noise")
SUMMARY_METRICS = ("accuracy", "weighted_f1", "macro_f1")


@dataclass(frozen=True)
class LossConfig:
    """One complete training objective for the comparison."""

    name: str
    objective: str = "binary_rejection"
    contrast_weight: float = 0.0
    contrast_temperature: float = 0.1
    noise_weight: float = 0.5
    repulsion_weight: float = 0.0
    noise_margin: float = 0.5

    def __post_init__(self) -> None:
        if self.objective not in {"binary_rejection", "three_class_ce"}:
            raise ValueError(
                "objective must be either 'binary_rejection' or 'three_class_ce'"
            )
        if self.contrast_temperature <= 0:
            raise ValueError("contrast_temperature must be greater than zero")
        for field_name in (
            "contrast_weight",
            "noise_weight",
            "repulsion_weight",
            "noise_margin",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


LOSS_CONFIGS = {
    "current": LossConfig(
        name="current",
        contrast_weight=1.0,
        noise_weight=0.5,
        repulsion_weight=0.5,
        noise_margin=5.0,
    ),
    "corrected_margin": LossConfig(
        name="corrected_margin",
        contrast_weight=1.0,
        noise_weight=0.5,
        repulsion_weight=0.5,
        noise_margin=0.5,
    ),
    "bce_noise": LossConfig(
        name="bce_noise",
        contrast_weight=0.0,
        noise_weight=0.5,
        repulsion_weight=0.0,
    ),
    "bce_contrast_noise": LossConfig(
        name="bce_contrast_noise",
        contrast_weight=0.25,
        noise_weight=0.5,
        repulsion_weight=0.0,
    ),
    "bce_repulsion_noise": LossConfig(
        name="bce_repulsion_noise",
        contrast_weight=0.0,
        noise_weight=0.5,
        repulsion_weight=0.1,
        noise_margin=0.5,
    ),
    "moderate_full": LossConfig(
        name="moderate_full",
        contrast_weight=0.25,
        noise_weight=0.5,
        repulsion_weight=0.1,
        noise_margin=0.5,
    ),
    "three_class_ce": LossConfig(
        name="three_class_ce",
        objective="three_class_ce",
        noise_weight=0.0,
        noise_margin=0.0,
    ),
}


def get_loss_config(name: str) -> LossConfig:
    """Return a named objective configuration."""
    try:
        return LOSS_CONFIGS[name]
    except KeyError as error:
        choices = ", ".join(LOSS_CONFIGS)
        raise ValueError(f"Unknown loss formulation {name!r}; choose from: {choices}") from error


def train_loss_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: LossConfig,
) -> dict[str, float]:
    """Train one epoch using exactly one configured loss formulation."""
    model.train()
    names = (
        "supervised",
        "contrast",
        "noise",
        "repulsion",
        "weighted_contrast",
        "weighted_noise",
        "weighted_repulsion",
        "total",
    )
    totals = {name: 0.0 for name in names}

    for wave, spec, global_features, labels, subject_ids in loader:
        wave = wave.to(device)
        spec = spec.to(device)
        global_features = global_features.to(device)
        labels = labels.to(device)
        subject_ids = subject_ids.to(device)
        logits, embeddings, _ = model(wave, spec, global_features)
        embeddings = F.normalize(embeddings, dim=1)

        if config.objective == "three_class_ce":
            supervised = F.cross_entropy(logits, labels)
            contrast = logits.new_tensor(0.0)
            noise = logits.new_tensor(0.0)
            repulsion = logits.new_tensor(0.0)
        else:
            noise_mask = labels == 2
            known_mask = ~noise_mask
            if known_mask.any():
                supervised = F.binary_cross_entropy_with_logits(
                    logits[known_mask], labels[known_mask].float()
                )
                if config.contrast_weight > 0:
                    contrast = supervised_contrastive_loss(
                        embeddings[known_mask],
                        labels[known_mask],
                        subject_ids[known_mask],
                        temperature=config.contrast_temperature,
                    )
                else:
                    contrast = logits.new_tensor(0.0)
            else:
                supervised = logits.new_tensor(0.0)
                contrast = logits.new_tensor(0.0)

            if noise_mask.any():
                noise_probability = torch.sigmoid(logits[noise_mask])
                noise = F.mse_loss(
                    noise_probability,
                    torch.full_like(noise_probability, 0.5),
                )
            else:
                noise = logits.new_tensor(0.0)

            if (
                config.repulsion_weight > 0
                and noise_mask.any()
                and known_mask.any()
            ):
                distances = torch.cdist(
                    embeddings[noise_mask], embeddings[known_mask]
                )
                repulsion = torch.clamp(
                    config.noise_margin - distances.min(dim=1).values,
                    min=0,
                ).mean()
            else:
                repulsion = logits.new_tensor(0.0)

        weighted_contrast = config.contrast_weight * contrast
        weighted_noise = config.noise_weight * noise
        weighted_repulsion = config.repulsion_weight * repulsion
        loss = supervised + weighted_contrast + weighted_noise + weighted_repulsion

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        values = {
            "supervised": supervised,
            "contrast": contrast,
            "noise": noise,
            "repulsion": repulsion,
            "weighted_contrast": weighted_contrast,
            "weighted_noise": weighted_noise,
            "weighted_repulsion": weighted_repulsion,
            "total": loss,
        }
        for name, value in values.items():
            totals[name] += float(value.detach())

    if not len(loader):
        raise ValueError("Training loader produced no batches")
    return {name: value / len(loader) for name, value in totals.items()}


@torch.no_grad()
def evaluate_loss_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: LossConfig,
) -> dict[str, np.ndarray]:
    """Collect predictions and scores for either objective family."""
    model.eval()
    labels = []
    subjects = []
    embeddings = []
    probabilities = []
    energies = []
    predictions = []

    for wave, spec, global_features, batch_labels, subject_ids in loader:
        logits, batch_embeddings, _ = model(
            wave.to(device),
            spec.to(device),
            global_features.to(device),
        )
        labels.append(batch_labels.numpy())
        subjects.append(subject_ids.numpy())
        embeddings.append(batch_embeddings.cpu().numpy())
        if config.objective == "three_class_ce":
            batch_probabilities = torch.softmax(logits, dim=1)
            probabilities.append(batch_probabilities.cpu().numpy())
            predictions.append(batch_probabilities.argmax(dim=1).cpu().numpy())
        else:
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            energies.append(torch.abs(logits).cpu().numpy())

    result = {
        "labels": np.concatenate(labels),
        "subjects": np.concatenate(subjects),
        "embeddings": np.concatenate(embeddings),
        "probabilities": np.concatenate(probabilities),
    }
    if config.objective == "three_class_ce":
        result["predictions"] = np.concatenate(predictions)
    else:
        result["energies"] = np.concatenate(energies)
    return result


def _print_confusion_matrix(matrix: list[list[int]]) -> None:
    print("\nConfusion matrix:")
    print(" " * 15 + " ".join(f"{name:>12}" for name in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, matrix):
        print(f"{name:>15} " + " ".join(f"{value:12d}" for value in row))


def run_loss_comparison(
    dataset_path: str | Path,
    formulations: list[str],
    *,
    architecture: str = "baseline",
    folds: int = 3,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    subjects_per_batch: int = 8,
    default_energy_threshold: float = 3.13,
    tune_threshold: bool = True,
    calibration_fraction: float = 0.2,
    num_workers: int = 0,
    seed: int = 0,
    device_name: str = "auto",
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Compare loss formulations using identical architecture and data splits."""
    unknown = sorted(set(formulations) - set(LOSS_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown loss formulations: {unknown}")
    if len(formulations) != len(set(formulations)):
        raise ValueError("Each loss formulation may be specified only once")
    if architecture not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown architecture: {architecture}")

    started_at = datetime.now(timezone.utc)
    if output_path is None:
        stamp = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
        output_path = Path("loss_comparison_results") / f"losses_{stamp}.json"
    elif Path(output_path).exists() and not overwrite:
        raise FileExistsError(f"Results already exist: {output_path}")

    device = resolve_device(device_name)
    set_seed(seed)
    dataset_path = Path(dataset_path)
    samples = load_samples(dataset_path)
    outer_splits = subject_kfold(samples, k=folds, seed=seed)
    architecture_config = ABLATION_CONFIGS[architecture]
    results: dict[str, list[dict[str, object]]] = {}

    for formulation in formulations:
        loss_config = get_loss_config(formulation)
        print(f"\n######## Loss formulation: {formulation} ########")
        fold_results = []
        for fold_index, (outer_training, validation_samples) in enumerate(
            outer_splits, start=1
        ):
            fold_seed = seed + fold_index
            set_seed(fold_seed)
            train_samples, calibration_samples = split_calibration_subjects(
                outer_training,
                calibration_fraction,
                seed + fold_index,
            )
            print(f"\n===== {formulation}: fold {fold_index}/{folds} =====")

            class_count = 3 if loss_config.objective == "three_class_ce" else 1
            model = AblationModel(
                architecture_config,
                feat_dim=64,
                n_classes=class_count,
            )
            replace_batch_norm_with_group_norm(model)
            model.to(device)

            train_loader, feature_mean, feature_std, _ = (
                make_subject_class_balanced_loader(
                    train_samples,
                    batch_size,
                    use_global_features=architecture_config.use_global_features,
                    subjects_per_batch=subjects_per_batch,
                    num_workers=num_workers,
                    seed=fold_seed,
                )
            )
            calibration_loader = make_evaluation_loader(
                calibration_samples,
                batch_size,
                feature_mean=feature_mean,
                feature_std=feature_std,
                use_global_features=architecture_config.use_global_features,
                num_workers=num_workers,
            )
            validation_loader = make_evaluation_loader(
                validation_samples,
                batch_size,
                feature_mean=feature_mean,
                feature_std=feature_std,
                use_global_features=architecture_config.use_global_features,
                num_workers=num_workers,
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            history = []
            for epoch in range(1, epochs + 1):
                stats = train_loss_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    loss_config,
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
                model, calibration_loader, device, loss_config
            )
            validation = evaluate_loss_model(
                model, validation_loader, device, loss_config
            )
            energy_threshold = None
            if loss_config.objective == "three_class_ce":
                calibration_predictions = calibration["predictions"]
                validation_predictions = validation["predictions"]
            else:
                energy_threshold = default_energy_threshold
                if tune_threshold:
                    energy_threshold, _ = tune_energy_threshold(
                        calibration["probabilities"],
                        calibration["labels"],
                        calibration["energies"],
                        default_energy_threshold,
                    )
                calibration_predictions = predict_classes(
                    calibration["probabilities"],
                    calibration["energies"],
                    energy_threshold,
                )
                validation_predictions = predict_classes(
                    validation["probabilities"],
                    validation["energies"],
                    energy_threshold,
                )

            calibration_metrics = calculate_metrics(
                calibration["labels"], calibration_predictions
            )
            validation_metrics = calculate_metrics(
                validation["labels"], validation_predictions
            )
            fold_result = {
                "fold": fold_index,
                "seed": fold_seed,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
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
                "energy_threshold": energy_threshold,
                "calibration_metrics": calibration_metrics,
                "training_history": history,
                **validation_metrics,
            }
            fold_results.append(fold_result)
            _print_confusion_matrix(validation_metrics["confusion_matrix"])
            print(
                f"threshold={energy_threshold} | "
                f"accuracy={validation_metrics['accuracy']:.4f} | "
                f"macro_f1={validation_metrics['macro_f1']:.4f}"
            )
            del model, optimizer, train_loader, calibration_loader, validation_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[formulation] = fold_results

    summary = {}
    for formulation, fold_results in results.items():
        summary[formulation] = {}
        for metric in SUMMARY_METRICS:
            values = [float(result[metric]) for result in fold_results]
            summary[formulation][metric] = {
                "mean": mean(values),
                "standard_deviation": pstdev(values),
            }

    print("\nMean cross-validation metrics")
    for formulation in formulations:
        metrics = summary[formulation]
        print(
            f"{formulation:>24}: "
            f"macro_f1={metrics['macro_f1']['mean']:.4f} | "
            f"accuracy={metrics['accuracy']['mean']:.4f}"
        )

    finished_at = datetime.now(timezone.utc)
    dataset_stat = dataset_path.stat()
    payload = {
        "schema_version": 1,
        "experiment_type": "loss_formulation_comparison",
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
            "formulations": formulations,
            "architecture": architecture,
            "folds": folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "subjects_per_batch": subjects_per_batch,
            "default_energy_threshold": default_energy_threshold,
            "tune_threshold": tune_threshold,
            "calibration_fraction": calibration_fraction,
            "num_workers": num_workers,
            "seed": seed,
            "device_name": device_name,
        },
        "architecture_config": architecture_config.to_dict(),
        "loss_configs": {
            name: get_loss_config(name).to_dict() for name in formulations
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
        "class_order": list(CLASS_NAMES),
        "fold_results": results,
        "summary": summary,
    }
    saved_path = save_json(payload, output_path, overwrite)
    print(f"Saved loss comparison results to {saved_path}")
    return payload


def parse_formulations(values: list[str]) -> list[str]:
    """Expand the special ``all`` selection."""
    if values == ["all"]:
        return list(LOSS_CONFIGS)
    if "all" in values:
        raise ValueError("Use --formulations all by itself")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset_arcsinh.pkl")
    parser.add_argument(
        "--formulations",
        nargs="+",
        choices=(*LOSS_CONFIGS, "all"),
        default=["all"],
        help="Loss formulations to compare; defaults to all seven",
    )
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
    parser.add_argument(
        "--fixed-threshold",
        action="store_true",
        help="Use --energy-threshold instead of calibration for binary objectives",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_loss_comparison(
        args.dataset,
        parse_formulations(args.formulations),
        architecture=args.architecture,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        subjects_per_batch=args.subjects_per_batch,
        default_energy_threshold=args.energy_threshold,
        tune_threshold=not args.fixed_threshold,
        calibration_fraction=args.calibration_fraction,
        num_workers=args.workers,
        seed=args.seed,
        device_name=args.device,
        output_path=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
