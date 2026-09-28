"""Run reproducible subject-wise cross-validation for architecture ablations."""

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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .ablation_model import ABLATION_CONFIGS, AblationModel, get_ablation_config
from .loaders import (
    load_samples,
    make_evaluation_loader,
    make_subject_class_balanced_loader,
)
from .train import (
    evaluate,
    predict_classes,
    replace_batch_norm_with_group_norm,
    set_seed,
    subject_kfold,
    train_epoch,
)
from .train_full_dataset import resolve_device


CLASS_NAMES = ("ripple", "ied", "noise")
SUMMARY_METRICS = ("accuracy", "weighted_f1", "macro_f1")


def split_calibration_subjects(
    samples: list[dict], fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Reserve complete subjects for threshold calibration."""
    if not 0 < fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    subjects = np.asarray(sorted({sample["animal_id"] for sample in samples}), dtype=object)
    if len(subjects) < 3:
        raise ValueError("At least three training subjects are required for calibration")
    np.random.default_rng(seed).shuffle(subjects)
    calibration_count = min(len(subjects) - 1, max(1, round(len(subjects) * fraction)))
    calibration_subjects = set(subjects[:calibration_count].tolist())
    training = [
        sample for sample in samples if sample["animal_id"] not in calibration_subjects
    ]
    calibration = [
        sample for sample in samples if sample["animal_id"] in calibration_subjects
    ]
    missing_training_classes = {0, 1, 2} - {sample["label"] for sample in training}
    if missing_training_classes:
        raise ValueError(
            "Calibration split removed all training examples for classes "
            f"{sorted(missing_training_classes)}"
        )
    return training, calibration


def tune_energy_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    energies: np.ndarray,
    default: float,
    candidate_count: int = 257,
) -> tuple[float, float]:
    """Select a noise-rejection threshold using calibration macro F1."""
    if candidate_count < 3:
        raise ValueError("candidate_count must be at least 3")
    quantiles = np.linspace(0, 1, candidate_count)
    candidates = np.unique(
        np.concatenate(([0.0, default], np.quantile(energies, quantiles)))
    )
    scored = []
    for threshold in candidates:
        predictions = predict_classes(probabilities, energies, float(threshold))
        score = f1_score(
            labels,
            predictions,
            labels=(0, 1, 2),
            average="macro",
            zero_division=0,
        )
        scored.append((float(score), -abs(float(threshold) - default), float(threshold)))
    best_score, _, best_threshold = max(scored)
    return best_threshold, best_score


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    """Calculate aggregate and per-class metrics in a JSON-safe form."""
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=(0, 1, 2),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=(0, 1, 2),
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=(0, 1, 2)
        ).tolist(),
    }


def save_json(payload: dict, path: str | Path, overwrite: bool) -> Path:
    """Atomically save an ablation result payload."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Results already exist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.resolve()


def run_ablations(
    dataset_path: str | Path,
    variants: list[str],
    *,
    folds: int = 3,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    subjects_per_batch: int = 8,
    default_energy_threshold: float = 3.13,
    tune_threshold: bool = True,
    calibration_fraction: float = 0.2,
    contrast_weight: float = 1.0,
    noise_weight: float = 0.5,
    repulsion_weight: float = 0.5,
    noise_margin: float = 5.0,
    num_workers: int = 0,
    seed: int = 0,
    device_name: str = "auto",
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Train and evaluate every requested ablation on identical outer folds."""
    unknown = sorted(set(variants) - set(ABLATION_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown ablation variants: {unknown}")
    if len(set(variants)) != len(variants):
        raise ValueError("Each ablation variant may be specified only once")

    started_at = datetime.now(timezone.utc)
    if output_path is None:
        stamp = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
        output_path = Path("ablation_results") / f"ablations_{stamp}.json"
    elif Path(output_path).exists() and not overwrite:
        raise FileExistsError(f"Results already exist: {output_path}")

    set_seed(seed)
    device = resolve_device(device_name)
    dataset_path = Path(dataset_path)
    samples = load_samples(dataset_path)
    outer_splits = subject_kfold(samples, k=folds, seed=seed)
    results: dict[str, list[dict[str, object]]] = {}

    for variant in variants:
        config = get_ablation_config(variant)
        print(f"\n######## Ablation: {variant} ########")
        fold_results = []
        for fold_index, (outer_training, validation_samples) in enumerate(
            outer_splits, start=1
        ):
            # Use the same fold seed for every architecture so initialization,
            # batch sampling, and augmentation randomness are paired wherever
            # the corresponding modules exist.
            fold_seed = seed + fold_index
            set_seed(fold_seed)
            if tune_threshold:
                train_samples, calibration_samples = split_calibration_subjects(
                    outer_training,
                    calibration_fraction,
                    seed + fold_index,
                )
            else:
                train_samples = outer_training
                calibration_samples = []

            print(f"\n===== {variant}: fold {fold_index}/{folds} =====")
            model = AblationModel(config, feat_dim=64, n_classes=1)
            replace_batch_norm_with_group_norm(model)
            model.to(device)

            train_loader, feature_mean, feature_std, _ = (
                make_subject_class_balanced_loader(
                    train_samples,
                    batch_size,
                    use_global_features=config.use_global_features,
                    subjects_per_batch=subjects_per_batch,
                    num_workers=num_workers,
                    seed=fold_seed,
                )
            )
            validation_loader = make_evaluation_loader(
                validation_samples,
                batch_size,
                feature_mean=feature_mean,
                feature_std=feature_std,
                use_global_features=config.use_global_features,
                num_workers=num_workers,
            )
            calibration_loader = None
            if tune_threshold:
                calibration_loader = make_evaluation_loader(
                    calibration_samples,
                    batch_size,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    use_global_features=config.use_global_features,
                    num_workers=num_workers,
                )

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            history = []
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
                    f"contrast={stats['contrast']:.4f} | "
                    f"noise={stats['noise']:.4f} | "
                    f"repulsion={stats['repulsion']:.4f}"
                )

            threshold = default_energy_threshold
            calibration_macro_f1 = None
            if calibration_loader is not None:
                cal_probabilities, cal_labels, _, _, cal_energies = evaluate(
                    model, calibration_loader, device
                )
                threshold, calibration_macro_f1 = tune_energy_threshold(
                    cal_probabilities,
                    cal_labels,
                    cal_energies,
                    default_energy_threshold,
                )

            probabilities, labels, subjects, _, energies = evaluate(
                model, validation_loader, device
            )
            predictions = predict_classes(probabilities, energies, threshold)
            metrics = calculate_metrics(labels, predictions)
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
                "energy_threshold": threshold,
                "calibration_macro_f1": calibration_macro_f1,
                "training_history": history,
                **metrics,
            }
            fold_results.append(fold_result)
            print(
                f"threshold={threshold:.4f} | "
                f"accuracy={metrics['accuracy']:.4f} | "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )
            del model, optimizer, train_loader, validation_loader, calibration_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[variant] = fold_results

    summary = {}
    for variant, fold_results in results.items():
        summary[variant] = {}
        for metric in SUMMARY_METRICS:
            values = [float(result[metric]) for result in fold_results]
            summary[variant][metric] = {
                "mean": mean(values),
                "standard_deviation": pstdev(values),
            }

    finished_at = datetime.now(timezone.utc)
    dataset_stat = dataset_path.stat()
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
            "subject_count": len({sample["animal_id"] for sample in samples}),
        },
        "parameters": {
            "variants": variants,
            "folds": folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "subjects_per_batch": subjects_per_batch,
            "default_energy_threshold": default_energy_threshold,
            "tune_threshold": tune_threshold,
            "calibration_fraction": calibration_fraction,
            "contrast_weight": contrast_weight,
            "noise_weight": noise_weight,
            "repulsion_weight": repulsion_weight,
            "noise_margin": noise_margin,
            "num_workers": num_workers,
            "seed": seed,
            "device_name": device_name,
        },
        "ablation_configs": {
            name: get_ablation_config(name).to_dict() for name in variants
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
    print(f"Saved ablation results to {saved_path}")
    return payload


def parse_variants(values: list[str]) -> list[str]:
    """Expand the special ``all`` selection and reject mixed use."""
    if values == ["all"]:
        return list(ABLATION_CONFIGS)
    if "all" in values:
        raise ValueError("Use --variants all by itself")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset_arcsinh.pkl")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline"],
        choices=(*ABLATION_CONFIGS, "all"),
        help="Named variants to run; defaults to baseline only",
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
        help="Use --energy-threshold instead of a held-out subject calibration split",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument("--noise-weight", type=float, default=0.5)
    parser.add_argument("--repulsion-weight", type=float, default=0.5)
    parser.add_argument("--noise-margin", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_ablations(
        args.dataset,
        parse_variants(args.variants),
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        subjects_per_batch=args.subjects_per_batch,
        default_energy_threshold=args.energy_threshold,
        tune_threshold=not args.fixed_threshold,
        calibration_fraction=args.calibration_fraction,
        contrast_weight=args.contrast_weight,
        noise_weight=args.noise_weight,
        repulsion_weight=args.repulsion_weight,
        noise_margin=args.noise_margin,
        num_workers=args.workers,
        seed=args.seed,
        device_name=args.device,
        output_path=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
