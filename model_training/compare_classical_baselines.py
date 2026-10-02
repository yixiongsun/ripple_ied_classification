"""Run nested, subject-grouped SVM baselines against the frozen protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Sequence

import numpy as np
import sklearn
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .ablation_test import (
    CLASS_NAMES,
    SUMMARY_METRICS,
    apply_confidence_rejection,
    calculate_metrics,
    calculate_per_subject_metrics,
    save_json,
    split_calibration_subjects,
    tune_rejection_threshold,
)
from .classical_features import assemble_feature_matrix, build_preprocessor
from .loaders import load_samples
from .train import subject_kfold


MODEL_NAMES = ("binary_rejection_svm", "three_class_svm")


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3)
    return {name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)}


def _subject_names(samples: Sequence[dict]) -> list[str]:
    return sorted({str(sample["animal_id"]) for sample in samples})


def _event_identifier(sample: dict, fallback_index: int) -> str:
    """Create a stable human-readable ID from persistent event metadata."""
    parts = [
        str(sample.get("animal_id", "")),
        str(sample.get("session", "")),
        str(sample.get("event_num", "")),
        str(sample.get("file", "")),
        np.asarray(sample.get("window", ())).astype(str).tolist(),
    ]
    encoded = json.dumps(parts, separators=(",", ":"), sort_keys=False).encode()
    return f"event-{hashlib.sha256(encoded).hexdigest()[:20]}-{fallback_index}"


def decision_scores_to_binary(
    decision_scores: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """Map an SVC signed decision score to the actual 0/1 class labels."""
    classes = np.asarray(classes)
    if classes.tolist() != [0, 1]:
        raise ValueError(f"Binary SVM classes must be [0, 1], received {classes.tolist()}")
    scores = np.asarray(decision_scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("Binary SVM decision scores must be one-dimensional")
    return np.where(scores >= 0, classes[1], classes[0]).astype(np.int64)


def _indices(samples: Sequence[dict], index_by_identity: dict[int, int]) -> np.ndarray:
    return np.asarray([index_by_identity[id(sample)] for sample in samples], dtype=int)


def _inner_splits(
    fitting_samples: list[dict],
    index_by_identity: dict[int, int],
    fitting_indices: np.ndarray,
    *,
    folds: int,
    seed: int,
    binary: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return local row indices for deterministic grouped inner validation."""
    global_to_local = {int(value): index for index, value in enumerate(fitting_indices)}
    splits = []
    for inner_training, inner_validation in subject_kfold(
        fitting_samples, k=folds, seed=seed
    ):
        train_global = _indices(inner_training, index_by_identity)
        validation_global = _indices(inner_validation, index_by_identity)
        train_local = np.asarray([global_to_local[int(value)] for value in train_global])
        validation_local = np.asarray(
            [global_to_local[int(value)] for value in validation_global]
        )
        if binary:
            train_local = train_local[
                np.asarray([fitting_samples[index]["label"] for index in train_local]) != 2
            ]
            validation_local = validation_local[
                np.asarray(
                    [fitting_samples[index]["label"] for index in validation_local]
                )
                != 2
            ]
        if not len(train_local) or not len(validation_local):
            raise ValueError("An inner fold has no usable samples")
        splits.append((train_local, validation_local))
    return splits


def _build_pipeline(
    waveform_feature_count: int,
    pca_components: int,
    *,
    seed: int,
    cache_size_mb: float,
    memory: str | None = None,
) -> Pipeline:
    return Pipeline(
        (
            (
                "preprocess",
                build_preprocessor(
                    waveform_feature_count, pca_components, random_state=seed
                ),
            ),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    cache_size=cache_size_mb,
                ),
            ),
        ),
        memory=memory,
    )


def _select_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    waveform_feature_count: int,
    pca_components: list[int],
    c_values: list[float],
    gamma_values: list[str | float],
    seed: int,
    binary: bool,
    cache_size_mb: float,
    jobs: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    max_components = min(
        waveform_feature_count,
        min(len(train_indices) for train_indices, _ in splits),
    )
    valid_components = [value for value in pca_components if value <= max_components]
    if not valid_components:
        raise ValueError(
            f"No PCA candidate fits the inner training data (maximum {max_components})"
        )
    labels = (0, 1) if binary else (0, 1, 2)
    scorer = make_scorer(
        f1_score, labels=labels, average="macro", zero_division=0
    )
    with tempfile.TemporaryDirectory(prefix="phase10-sklearn-cache-") as cache:
        estimator = _build_pipeline(
            waveform_feature_count,
            valid_components[0],
            seed=seed,
            cache_size_mb=cache_size_mb,
            memory=cache,
        )
        search = GridSearchCV(
            estimator,
            param_grid={
                "preprocess__waveform__pca__n_components": valid_components,
                "svm__C": c_values,
                "svm__gamma": gamma_values,
            },
            scoring=scorer,
            cv=splits,
            refit=False,
            n_jobs=jobs,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X, y)
    ranked = sorted(
        range(len(search.cv_results_["params"])),
        key=lambda index: (
            int(search.cv_results_["rank_test_score"][index]),
            index,
        ),
    )
    best_index = ranked[0]
    selected = {
        "pca_components": int(
            search.cv_results_["params"][best_index][
                "preprocess__waveform__pca__n_components"
            ]
        ),
        "C": float(search.cv_results_["params"][best_index]["svm__C"]),
        "gamma": search.cv_results_["params"][best_index]["svm__gamma"],
        "mean_macro_f1": float(search.cv_results_["mean_test_score"][best_index]),
        "standard_deviation": float(
            search.cv_results_["std_test_score"][best_index]
        ),
    }
    scores = []
    for index in ranked:
        params = search.cv_results_["params"][index]
        scores.append(
            {
                "pca_components": int(
                    params["preprocess__waveform__pca__n_components"]
                ),
                "C": float(params["svm__C"]),
                "gamma": params["svm__gamma"],
                "mean_macro_f1": float(search.cv_results_["mean_test_score"][index]),
                "standard_deviation": float(
                    search.cv_results_["std_test_score"][index]
                ),
                "rank": int(search.cv_results_["rank_test_score"][index]),
            }
        )
    return selected, scores


def _preprocessing_summary(pipeline: Pipeline, fitting_subjects: list[str]) -> dict:
    preprocessor = pipeline.named_steps["preprocess"]
    waveform = preprocessor.named_transformers_["waveform"]
    wave_scaler = waveform.named_steps["scale"]
    pca = waveform.named_steps["pca"]
    global_scaler = preprocessor.named_transformers_["global"]
    return {
        "fitted_subjects": fitting_subjects,
        "waveform_input_features": int(wave_scaler.n_features_in_),
        "waveform_scaler_samples_seen": int(wave_scaler.n_samples_seen_),
        "pca_components": int(pca.n_components_),
        "pca_explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "global_feature_mean": global_scaler.mean_.astype(float).tolist(),
        "global_feature_scale": global_scaler.scale_.astype(float).tolist(),
        "global_scaler_samples_seen": int(global_scaler.n_samples_seen_),
    }


def _direct_confidence(decision_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(decision_scores, dtype=float)
    if scores.ndim == 1:
        return np.abs(scores)
    ordered = np.sort(scores, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _fit_and_evaluate(
    model_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    fitting_indices: np.ndarray,
    calibration_indices: np.ndarray,
    validation_indices: np.ndarray,
    fitting_samples: list[dict],
    validation_samples: list[dict],
    index_by_identity: dict[int, int],
    *,
    waveform_feature_count: int,
    inner_folds: int,
    pca_components: list[int],
    c_values: list[float],
    gamma_values: list[str | float],
    training_seed: int,
    threshold_candidates: int,
    cache_size_mb: float,
    jobs: int,
) -> dict[str, object]:
    started = perf_counter()
    binary = model_name == "binary_rejection_svm"
    inner = _inner_splits(
        fitting_samples,
        index_by_identity,
        fitting_indices,
        folds=inner_folds,
        seed=training_seed,
        binary=binary,
    )
    search_indices = fitting_indices
    if binary:
        search_indices = fitting_indices[labels[fitting_indices] != 2]
        # Inner split indices are relative to all fitting rows. Rebase after removing noise.
        retained = np.flatnonzero(labels[fitting_indices] != 2)
        old_to_new = {int(old): new for new, old in enumerate(retained)}
        inner = [
            (
                np.asarray([old_to_new[int(value)] for value in train]),
                np.asarray([old_to_new[int(value)] for value in validation]),
            )
            for train, validation in inner
        ]
    selected, inner_scores = _select_hyperparameters(
        features[search_indices],
        labels[search_indices],
        inner,
        waveform_feature_count=waveform_feature_count,
        pca_components=pca_components,
        c_values=c_values,
        gamma_values=gamma_values,
        seed=training_seed,
        binary=binary,
        cache_size_mb=cache_size_mb,
        jobs=jobs,
    )
    pipeline = _build_pipeline(
        waveform_feature_count,
        int(selected["pca_components"]),
        seed=training_seed,
        cache_size_mb=cache_size_mb,
    )
    pipeline.set_params(svm__C=selected["C"], svm__gamma=selected["gamma"])
    fit_started = perf_counter()
    pipeline.fit(features[search_indices], labels[search_indices])
    fit_seconds = perf_counter() - fit_started

    calibration_threshold = None
    calibration_macro_f1 = None
    if binary:
        calibration_scores = pipeline.decision_function(features[calibration_indices])
        calibration_binary = decision_scores_to_binary(
            calibration_scores, pipeline.named_steps["svm"].classes_
        )
        calibration_threshold, calibration_macro_f1 = tune_rejection_threshold(
            calibration_binary,
            labels[calibration_indices],
            np.abs(calibration_scores),
            default=0.0,
            candidate_count=threshold_candidates,
        )

    inference_started = perf_counter()
    validation_scores = pipeline.decision_function(features[validation_indices])
    if binary:
        binary_predictions = decision_scores_to_binary(
            validation_scores, pipeline.named_steps["svm"].classes_
        )
        confidence = np.abs(validation_scores)
        predictions = apply_confidence_rejection(
            binary_predictions, confidence, float(calibration_threshold)
        )
    else:
        predictions = pipeline.predict(features[validation_indices]).astype(np.int64)
        confidence = _direct_confidence(validation_scores)
    inference_seconds = perf_counter() - inference_started

    validation_labels = labels[validation_indices]
    validation_subjects = _subject_names(validation_samples)
    subject_to_index = {subject: index for index, subject in enumerate(validation_subjects)}
    subject_indices = np.asarray(
        [subject_to_index[str(sample["animal_id"])] for sample in validation_samples]
    )
    svm = pipeline.named_steps["svm"]
    model_blob = pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL)
    metrics = calculate_metrics(validation_labels, predictions)
    return {
        "selected_hyperparameters": selected,
        "inner_cv_scores": inner_scores,
        "preprocessing": _preprocessing_summary(
            pipeline, _subject_names(fitting_samples)
        ),
        "rejection_threshold": (
            None if calibration_threshold is None else float(calibration_threshold)
        ),
        "calibration_macro_f1": (
            None if calibration_macro_f1 is None else float(calibration_macro_f1)
        ),
        "fit_seconds": float(fit_seconds),
        "inference_seconds": float(inference_seconds),
        "duration_seconds": float(perf_counter() - started),
        "support_vectors": int(svm.support_vectors_.shape[0]),
        "support_vectors_per_class": [int(value) for value in svm.n_support_],
        "serialized_model_size_bytes": len(model_blob),
        "per_subject_metrics": calculate_per_subject_metrics(
            validation_labels, predictions, subject_indices, validation_subjects
        ),
        "events": {
            "identifiers": [
                _event_identifier(sample, int(validation_indices[index]))
                for index, sample in enumerate(validation_samples)
            ],
            "labels": validation_labels.astype(int).tolist(),
            "predictions": predictions.astype(int).tolist(),
            "confidence_scores": np.asarray(confidence, dtype=float).tolist(),
        },
        **metrics,
    }


def _partial_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + ".partial" + output_path.suffix)


def _load_partial(path: Path, parameters: dict) -> dict[str, list[dict]]:
    if not path.exists():
        return {name: [] for name in MODEL_NAMES}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("parameters") != parameters:
        raise ValueError(
            f"Partial result parameters do not match this run: {path}. "
            "Use a different output path or --overwrite."
        )
    results = payload.get("fold_results", {})
    return {name: list(results.get(name, [])) for name in MODEL_NAMES}


def run_comparison(
    dataset_path: str | Path,
    *,
    folds: int = 5,
    seeds: list[int] | None = None,
    inner_folds: int = 3,
    calibration_fraction: float = 0.2,
    pca_components: list[int] | None = None,
    c_values: list[float] | None = None,
    gamma_values: list[str | float] | None = None,
    threshold_candidates: int = 257,
    cache_size_mb: float = 2_048,
    jobs: int = 1,
    output_path: str | Path = "baseline_results/phase10_svm_comparison.json",
    overwrite: bool = False,
) -> dict[str, object]:
    """Run both SVM baselines and atomically checkpoint every completed model-fold."""
    seeds = [0, 1, 2] if seeds is None else list(seeds)
    pca_components = [64, 128, 256] if pca_components is None else pca_components
    c_values = [0.1, 1.0, 10.0] if c_values is None else c_values
    gamma_values = ["scale", 0.01, 0.1] if gamma_values is None else gamma_values
    output_path = Path(output_path)
    partial_path = _partial_path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Results already exist: {output_path}")
    parameters = {
        "folds": folds,
        "seeds": seeds,
        "inner_folds": inner_folds,
        "calibration_fraction": calibration_fraction,
        "pca_components": pca_components,
        "C": c_values,
        "gamma": gamma_values,
        "class_weight": "balanced",
        "kernel": "rbf",
        "threshold_candidates": threshold_candidates,
    }
    results = (
        {name: [] for name in MODEL_NAMES}
        if overwrite
        else _load_partial(partial_path, parameters)
    )
    completed = {
        (name, int(item["repeat_seed"]), int(item["fold"]))
        for name, items in results.items()
        for item in items
    }

    started_at = datetime.now(timezone.utc)
    dataset_path = Path(dataset_path)
    print(f"Loading {dataset_path}...")
    samples = load_samples(dataset_path)
    labels = np.asarray([sample["label"] for sample in samples], dtype=np.int64)
    if not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("Dataset labels must use 0=ripple, 1=IED, 2=noise")
    print("Precomputing waveform and global features once...")
    feature_matrix = assemble_feature_matrix(samples)
    features = feature_matrix.values
    index_by_identity = {id(sample): index for index, sample in enumerate(samples)}

    for repeat_seed in seeds:
        outer_splits = subject_kfold(samples, k=folds, seed=repeat_seed)
        for fold_index, (outer_training, validation_samples) in enumerate(
            outer_splits, start=1
        ):
            training_seed = repeat_seed * 10_000 + fold_index
            fitting_samples, calibration_samples = split_calibration_subjects(
                outer_training, calibration_fraction, training_seed
            )
            fitting_indices = _indices(fitting_samples, index_by_identity)
            calibration_indices = _indices(calibration_samples, index_by_identity)
            validation_indices = _indices(validation_samples, index_by_identity)
            subject_sets = [
                set(_subject_names(group))
                for group in (fitting_samples, calibration_samples, validation_samples)
            ]
            if any(
                subject_sets[first] & subject_sets[second]
                for first, second in ((0, 1), (0, 2), (1, 2))
            ):
                raise RuntimeError("Subject leakage detected in an outer fold")
            common = {
                "repeat_seed": repeat_seed,
                "fold": fold_index,
                "training_seed": training_seed,
                "model_fitting_subjects": _subject_names(fitting_samples),
                # Compatibility with the locked Phase 8 schema.
                "train_subjects": _subject_names(fitting_samples),
                "calibration_subjects": _subject_names(calibration_samples),
                "validation_subjects": _subject_names(validation_samples),
                "model_fitting_samples": len(fitting_samples),
                "train_samples": len(fitting_samples),
                "calibration_samples": len(calibration_samples),
                "validation_samples": len(validation_samples),
                "model_fitting_class_counts": _class_counts(labels[fitting_indices]),
                "calibration_class_counts": _class_counts(labels[calibration_indices]),
                "validation_class_counts": _class_counts(labels[validation_indices]),
            }
            for model_name in MODEL_NAMES:
                key = (model_name, repeat_seed, fold_index)
                if key in completed:
                    print(f"Skipping completed {model_name}, seed {repeat_seed}, fold {fold_index}")
                    continue
                print(f"Running {model_name}, seed {repeat_seed}, fold {fold_index}/{folds}")
                result = _fit_and_evaluate(
                    model_name,
                    features,
                    labels,
                    fitting_indices,
                    calibration_indices,
                    validation_indices,
                    fitting_samples,
                    validation_samples,
                    index_by_identity,
                    waveform_feature_count=feature_matrix.waveform_feature_count,
                    inner_folds=inner_folds,
                    pca_components=pca_components,
                    c_values=c_values,
                    gamma_values=gamma_values,
                    training_seed=training_seed,
                    threshold_candidates=threshold_candidates,
                    cache_size_mb=cache_size_mb,
                    jobs=jobs,
                )
                results[model_name].append({**common, **result})
                completed.add(key)
                save_json(
                    {
                        "schema_version": 1,
                        "experiment_type": "phase10_classical_baseline_partial",
                        "started_at_utc": started_at.isoformat(),
                        "parameters": parameters,
                        "completed_runs": len(completed),
                        "expected_runs": len(seeds) * folds * len(MODEL_NAMES),
                        "fold_results": results,
                    },
                    partial_path,
                    overwrite=True,
                )
                print(f"macro_f1={result['macro_f1']:.4f}")

    summary = {}
    for model_name, model_results in results.items():
        summary[model_name] = {}
        for metric in SUMMARY_METRICS:
            values = [float(item[metric]) for item in model_results]
            summary[model_name][metric] = {
                "mean": mean(values),
                "standard_deviation": pstdev(values),
            }
    finished_at = datetime.now(timezone.utc)
    stat = dataset_path.stat()
    payload = {
        "schema_version": 1,
        "experiment_type": "phase10_classical_baseline_comparison",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "dataset": {
            "path": str(dataset_path.resolve()),
            "size_bytes": stat.st_size,
            "modified_at_utc": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat(),
            "sample_count": len(samples),
            "subject_count": len({sample["animal_id"] for sample in samples}),
            "waveform_shape": list(feature_matrix.waveform_shape),
        },
        "parameters": parameters,
        "runtime": {
            "command_line": [sys.executable, *sys.argv],
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "class_order": list(CLASS_NAMES),
        "fold_results": results,
        "summary": summary,
    }
    saved = save_json(payload, output_path, overwrite=overwrite)
    print(f"Saved comparison to {saved}")
    return payload


def _gamma(value: str) -> str | float:
    return value if value in {"scale", "auto"} else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset_arcsinh.pkl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--pca-components", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.1, 1, 10])
    parser.add_argument(
        "--gamma-values", type=_gamma, nargs="+", default=["scale", 0.01, 0.1]
    )
    parser.add_argument("--threshold-candidates", type=int, default=257)
    parser.add_argument("--cache-size-mb", type=float, default=2_048)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output", default="baseline_results/phase10_svm_comparison.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_comparison(
        args.dataset,
        folds=args.folds,
        seeds=args.seeds,
        inner_folds=args.inner_folds,
        calibration_fraction=args.calibration_fraction,
        pca_components=args.pca_components,
        c_values=args.c_values,
        gamma_values=args.gamma_values,
        threshold_candidates=args.threshold_candidates,
        cache_size_mb=args.cache_size_mb,
        jobs=args.jobs,
        output_path=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

