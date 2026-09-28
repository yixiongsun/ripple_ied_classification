"""Run subject-wise cross-validation for the full-dataset CNN configuration."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .loaders import (
    load_samples,
    make_evaluation_loader,
    make_subject_class_balanced_loader,
)
from .model import SharedModalityModel
from .train import (
    SubjectAdversary,
    evaluate,
    predict_classes,
    print_confusion_matrix,
    replace_batch_norm_with_group_norm,
    set_seed,
    subject_kfold,
    train_epoch,
)
from .train_full_dataset import resolve_device


METRIC_NAMES = ("accuracy", "weighted_f1", "macro_f1")


def _default_results_path(started_at: datetime) -> Path:
    timestamp = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("cross_validation_results") / f"cross_validation_{timestamp}.json"


def _save_results(payload: dict, output_path: str | Path, overwrite: bool) -> Path:
    """Atomically write a JSON result file and return its resolved path."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Results already exist: {output_path}. "
            "Pass --overwrite or choose another --output path."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path.resolve()


def _configure_gpu_memory(
    device: torch.device, gpu_memory_fraction: float | None
) -> None:
    """Apply an optional per-process CUDA memory limit before model allocation."""
    if gpu_memory_fraction is None:
        return
    if not 0 < gpu_memory_fraction <= 1:
        raise ValueError("gpu_memory_fraction must be greater than 0 and at most 1")
    if device.type != "cuda":
        raise ValueError("gpu_memory_fraction can only be used with a CUDA device")
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(
        gpu_memory_fraction, device=device_index
    )


def cross_validate(
    dataset_path: str | Path,
    *,
    folds: int = 3,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    subjects_per_batch: int = 8,
    energy_threshold: float = 3.13,
    contrast_weight: float = 1.0,
    noise_weight: float = 0.5,
    repulsion_weight: float = 0.5,
    noise_margin: float = 5.0,
    subject_weight: float = 0.0,
    prefer_cross_subject: bool = False,
    num_workers: int = 0,
    seed: int = 0,
    device_name: str = "auto",
    gpu_memory_fraction: float | None = None,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    """Cross-validate the training configuration and save a reproducible record."""
    started_at = datetime.now(timezone.utc)
    if output_path is None:
        output_path = _default_results_path(started_at)
    elif Path(output_path).exists() and not overwrite:
        raise FileExistsError(
            f"Results already exist: {output_path}. "
            "Pass --overwrite or choose another --output path."
        )

    dataset_path = Path(dataset_path)
    run_parameters = {
        "folds": folds,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "subjects_per_batch": subjects_per_batch,
        "energy_threshold": energy_threshold,
        "contrast_weight": contrast_weight,
        "noise_weight": noise_weight,
        "repulsion_weight": repulsion_weight,
        "noise_margin": noise_margin,
        "subject_weight": subject_weight,
        "prefer_cross_subject": prefer_cross_subject,
        "num_workers": num_workers,
        "seed": seed,
        "device_name": device_name,
        "gpu_memory_fraction": gpu_memory_fraction,
    }
    device = resolve_device(device_name)
    _configure_gpu_memory(device, gpu_memory_fraction)
    set_seed(seed)
    samples = load_samples(dataset_path)
    splits = subject_kfold(samples, k=folds, seed=seed)
    results = []

    for fold_index, (train_samples, validation_samples) in enumerate(splits, start=1):
        print(f"\n===== Fold {fold_index}/{folds} =====")
        model = SharedModalityModel(feat_dim=64, n_classes=1, use_global_features=True)
        replace_batch_norm_with_group_norm(model)
        model.to(device)

        train_loader, feature_mean, feature_std, subject_to_index = (
            make_subject_class_balanced_loader(
                train_samples,
                batch_size,
                subjects_per_batch=subjects_per_batch,
                num_workers=num_workers,
                seed=seed + fold_index,
            )
        )
        validation_loader = make_evaluation_loader(
            validation_samples,
            batch_size,
            feature_mean=feature_mean,
            feature_std=feature_std,
            num_workers=num_workers,
        )

        adversary = (
            SubjectAdversary(64, len(subject_to_index)).to(device)
            if subject_weight > 0
            else None
        )
        trainable_parameters = list(model.parameters())
        if adversary is not None:
            trainable_parameters.extend(adversary.parameters())
        optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
        training_history = []
        for epoch in range(1, epochs + 1):
            stats = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                contrast_weight=contrast_weight,
                noise_weight=noise_weight,
                repulsion_weight=repulsion_weight,
                noise_margin=noise_margin,
                prefer_cross_subject=prefer_cross_subject,
                subject_adversary=adversary,
                subject_weight=subject_weight,
            )
            training_history.append(
                {"epoch": epoch, **{name: float(value) for name, value in stats.items()}}
            )
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | total={stats['total']:.4f} | "
                f"supervised={stats['supervised']:.4f} | "
                f"contrast={stats['contrast']:.4f} | noise={stats['noise']:.4f} | "
                f"repulsion={stats['repulsion']:.4f} | subject={stats['subject']:.4f}"
            )

        probabilities, labels, _, _, energies = evaluate(
            model, validation_loader, device
        )
        predictions = predict_classes(probabilities, energies, energy_threshold)
        matrix = confusion_matrix(labels, predictions, labels=(0, 1, 2))
        print_confusion_matrix(matrix, ("ripple", "ied", "noise"))
        result = {
            "fold": fold_index,
            "train_subjects": sorted(
                {str(sample["animal_id"]) for sample in train_samples}
            ),
            "validation_subjects": sorted(
                {str(sample["animal_id"]) for sample in validation_samples}
            ),
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "training_history": training_history,
            "accuracy": float(accuracy_score(labels, predictions)),
            "weighted_f1": float(
                f1_score(labels, predictions, average="weighted", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(labels, predictions, average="macro", zero_division=0)
            ),
            "confusion_matrix": matrix.tolist(),
        }
        results.append(result)
        print(
            f"accuracy={result['accuracy']:.4f} | "
            f"weighted_f1={result['weighted_f1']:.4f} | "
            f"macro_f1={result['macro_f1']:.4f}"
        )

    print("\nMean CV metrics")
    summary = {}
    for metric in METRIC_NAMES:
        values = [result[metric] for result in results]
        summary[metric] = {
            "mean": mean(values),
            "standard_deviation": pstdev(values),
        }
        print(f"{metric}: {summary[metric]['mean']:.4f}")

    dataset_stat = dataset_path.stat()
    finished_at = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
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
            "subject_count": len({str(sample["animal_id"]) for sample in samples}),
        },
        "parameters": run_parameters,
        "model": {
            "class": "SharedModalityModel",
            "feat_dim": 64,
            "n_classes": 1,
            "use_global_features": True,
            "normalization": "group_norm",
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
    saved_path = _save_results(payload, output_path, overwrite)
    print(f"Saved cross-validation results to {saved_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset_arcsinh.pkl")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--subjects-per-batch", type=int, default=8)
    parser.add_argument("--energy-threshold", type=float, default=3.13)
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument("--noise-weight", type=float, default=0.5)
    parser.add_argument("--repulsion-weight", type=float, default=0.5)
    parser.add_argument("--noise-margin", type=float, default=5.0)
    parser.add_argument("--subject-weight", type=float, default=0.0)
    parser.add_argument("--prefer-cross-subject", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Data-loading subprocesses (default: 0 to avoid duplicating the "
            "in-memory dataset on Windows)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        help="Maximum fraction of CUDA memory available to this process (0 < value <= 1)",
    )
    parser.add_argument(
        "--output",
        help=(
            "JSON results path (default: a timestamped file under "
            "cross_validation_results/)"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cross_validate(
        args.dataset,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        subjects_per_batch=args.subjects_per_batch,
        energy_threshold=args.energy_threshold,
        contrast_weight=args.contrast_weight,
        noise_weight=args.noise_weight,
        repulsion_weight=args.repulsion_weight,
        noise_margin=args.noise_margin,
        subject_weight=args.subject_weight,
        prefer_cross_subject=args.prefer_cross_subject,
        num_workers=args.workers,
        seed=args.seed,
        device_name=args.device,
        gpu_memory_fraction=args.gpu_memory_fraction,
        output_path=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
