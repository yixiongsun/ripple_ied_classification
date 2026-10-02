from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model_training.ablation_test import (
    apply_confidence_rejection,
    calculate_metrics,
    split_calibration_subjects,
    tune_rejection_threshold,
)
from model_training.classical_features import (
    assemble_feature_matrix,
    build_preprocessor,
    flatten_waveforms,
)
from model_training.compare_classical_baselines import (
    _inner_splits,
    _load_partial,
    decision_scores_to_binary,
    run_comparison,
)
from model_training.train import subject_kfold


def _samples(subject_count: int = 9, events_per_class: int = 2) -> list[dict]:
    samples = []
    time = np.linspace(0, 1, 64, endpoint=False, dtype=np.float32)
    for subject_index in range(subject_count):
        for label in range(3):
            for event_index in range(events_per_class):
                frequency = 4 + label * 3
                wave = np.stack(
                    (
                        np.sin(2 * np.pi * frequency * time),
                        np.cos(2 * np.pi * frequency * time),
                    )
                ).astype(np.float32)
                wave += np.float32(subject_index * 0.01 + event_index * 0.001)
                samples.append(
                    {
                        "animal_id": f"subject-{subject_index:02d}",
                        "session": "synthetic",
                        "event_num": label * events_per_class + event_index,
                        "window": np.asarray([event_index, event_index + 64]),
                        "label": label,
                        "wave": wave,
                        "entropy": np.asarray([label + 0.1, label + 0.2]),
                        "ratio": np.asarray([frequency, frequency + 0.5]),
                    }
                )
    return samples


class ClassicalBaselineTests(unittest.TestCase):
    def test_waveform_flattening_is_deterministic_and_validates_shape(self):
        samples = _samples(1, 1)
        first = flatten_waveforms(samples)
        second = flatten_waveforms(samples)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (3, 128))
        samples[1]["wave"] = samples[1]["wave"][:, :-1]
        with self.assertRaisesRegex(ValueError, "shape"):
            flatten_waveforms(samples)

    def test_subject_partitions_and_inner_folds_never_overlap(self):
        samples = _samples()
        index_by_identity = {id(sample): index for index, sample in enumerate(samples)}
        outer_training, outer_validation = subject_kfold(samples, k=2, seed=0)[0]
        fitting, calibration = split_calibration_subjects(outer_training, 0.2, seed=1)
        groups = [
            {sample["animal_id"] for sample in part}
            for part in (fitting, calibration, outer_validation)
        ]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        fitting_indices = np.asarray([index_by_identity[id(sample)] for sample in fitting])
        splits = _inner_splits(
            fitting,
            index_by_identity,
            fitting_indices,
            folds=2,
            seed=1,
            binary=False,
        )
        fitting_subjects = np.asarray([sample["animal_id"] for sample in fitting])
        for train, validation in splits:
            self.assertFalse(
                set(fitting_subjects[train]) & set(fitting_subjects[validation])
            )

    def test_scaler_and_pca_are_fitted_on_training_rows_only(self):
        samples = _samples(3, 1)
        matrix = assemble_feature_matrix(samples)
        training = matrix.values[:6]
        held_out = matrix.values[6:] + 10_000
        preprocessor = build_preprocessor(
            matrix.waveform_feature_count, 2, random_state=7
        )
        preprocessor.fit(training)
        preprocessor.transform(held_out)
        waveform = preprocessor.named_transformers_["waveform"]
        self.assertEqual(int(waveform.named_steps["scale"].n_samples_seen_), len(training))
        self.assertEqual(
            int(preprocessor.named_transformers_["global"].n_samples_seen_), len(training)
        )
        np.testing.assert_allclose(
            preprocessor.named_transformers_["global"].mean_,
            training[:, matrix.waveform_feature_count :].mean(axis=0),
        )

    def test_binary_decision_sign_and_low_confidence_rejection(self):
        scores = np.asarray([-2.0, -0.1, 0.0, 3.0])
        binary = decision_scores_to_binary(scores, np.asarray([0, 1]))
        np.testing.assert_array_equal(binary, [0, 0, 1, 1])
        rejected = apply_confidence_rejection(binary, np.abs(scores), threshold=0.5)
        np.testing.assert_array_equal(rejected, [0, 2, 2, 1])

    def test_threshold_selection_uses_only_supplied_calibration_arrays(self):
        calibration_binary = np.asarray([0, 0, 1, 1, 0, 1])
        calibration_labels = np.asarray([0, 2, 1, 2, 0, 1])
        confidence = np.asarray([2.0, 0.1, 2.0, 0.2, 1.5, 1.2])
        first = tune_rejection_threshold(
            calibration_binary, calibration_labels, confidence, candidate_count=9
        )
        unrelated_validation_labels = np.asarray([2, 2, 2, 0, 1])
        self.assertEqual(unrelated_validation_labels.shape, (5,))
        second = tune_rejection_threshold(
            calibration_binary, calibration_labels, confidence, candidate_count=9
        )
        self.assertEqual(first, second)
        predictions = apply_confidence_rejection(calibration_binary, confidence, first[0])
        self.assertAlmostEqual(
            first[1], calculate_metrics(calibration_labels, predictions)["macro_f1"]
        )

    def test_two_fold_run_is_deterministic_complete_and_restartable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.pkl"
            output = root / "comparison.json"
            with dataset.open("wb") as destination:
                pickle.dump(_samples(), destination)
            options = dict(
                folds=2,
                seeds=[0],
                inner_folds=2,
                pca_components=[2],
                c_values=[1.0],
                gamma_values=["scale"],
                threshold_candidates=9,
                cache_size_mb=64,
                output_path=output,
            )
            first = run_comparison(dataset, **options)
            self.assertTrue(output.exists())
            with self.assertRaises(FileExistsError):
                run_comparison(dataset, **options)
            partial = output.with_name("comparison.partial.json")
            self.assertTrue(partial.exists())
            self.assertTrue(
                all(len(first["fold_results"][name]) == 2 for name in first["fold_results"])
            )
            expected_metrics = {
                name: [item["macro_f1"] for item in results]
                for name, results in first["fold_results"].items()
            }
            output.unlink()
            resumed = run_comparison(dataset, **options)
            self.assertEqual(
                {
                    name: [item["macro_f1"] for item in results]
                    for name, results in resumed["fold_results"].items()
                },
                expected_metrics,
            )
            parameters = resumed["parameters"]
            loaded = _load_partial(partial, parameters)
            self.assertTrue(all(len(loaded[name]) == 2 for name in loaded))
            with self.assertRaisesRegex(ValueError, "do not match"):
                _load_partial(partial, {**parameters, "folds": 3})

            deterministic_output = root / "deterministic.json"
            repeated = run_comparison(
                dataset, **{**options, "output_path": deterministic_output}
            )
            self.assertEqual(
                {
                    name: [item["selected_hyperparameters"] for item in results]
                    for name, results in repeated["fold_results"].items()
                },
                {
                    name: [item["selected_hyperparameters"] for item in results]
                    for name, results in first["fold_results"].items()
                },
            )

    def test_metric_calculation_matches_three_class_definition(self):
        labels = np.asarray([0, 0, 1, 1, 2, 2])
        predictions = np.asarray([0, 2, 1, 1, 2, 0])
        metrics = calculate_metrics(labels, predictions)
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 1], [0, 2, 0], [1, 0, 1]])


if __name__ == "__main__":
    unittest.main()

