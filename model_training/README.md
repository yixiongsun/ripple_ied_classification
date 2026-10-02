# Model training and evaluation

This package contains the neural and classical models used in the reported
subject-grouped evaluation.

## Selected method

The selected neural model is `compact_wave_no_temporal`: a shared per-channel
waveform encoder, morphology features, cross-channel attention, and five global
features fused into a 64-dimensional embedding. A binary ripple-versus-IED
score is trained with uncertainty and representation losses. Events whose
absolute score falls below a threshold fitted on training-side calibration
subjects are labeled as noise.

The locked settings are defined in `experiment_config.py`:

| Setting | Value |
| --- | --- |
| Epochs | 40 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Batch size | 32 |
| Subjects per batch | 8 |
| Embedding size | 64 |
| Dropout | 0 |
| Gradient clipping | Disabled |
| Calibration subjects | 20% of each outer-training partition |
| Threshold selection | Maximize calibration macro F1 |

## Main commands

Run commands from the repository root.

```powershell
# Locked compact-model versus full-model comparison
python -m model_training.final_confirmation --dataset dataset_arcsinh.pkl

# Nested subject-grouped classical baselines
python -m model_training.compare_classical_baselines --dataset dataset_arcsinh.pkl --folds 5 --seeds 0 1 2 --output experiment_results/baselines/svm_comparison.json

# Summarize a completed baseline comparison
python -m model_training.summarize_baseline_comparison --baseline experiment_results/baselines/svm_comparison.json

# General cross-validation and inference utilities
python -m model_training.cross_validate --help
python -m model_training.predict_events --help
```

`final_confirmation.py` accepts only data, execution, and output options so the
reported model settings cannot be changed accidentally. It evaluates the
selected compact model and the original full baseline on identical folds and
calibration subjects.

`compare_classical_baselines.py` fits preprocessing and hyperparameters inside
the training partition. It writes a resumable `.partial.json` during execution;
only the completed JSON and its summary belong in version control.

## Code map

| File | Purpose |
| --- | --- |
| `ablation_model.py` | Configurable full and compact neural architectures |
| `loaders.py` | Dataset loading, global features, normalization, and data loaders |
| `train.py` | Losses, subject folds, training loops, evaluation, and checkpoint helpers |
| `final_confirmation.py` | Locked repeated grouped comparison |
| `experiment_config.py` | Selected architecture and training constants |
| `classical_features.py` | Leakage-safe waveform/global-feature preprocessing |
| `compare_classical_baselines.py` | Nested grouped RBF-SVM evaluation |
| `summarize_final_subjects.py` | Per-subject summaries from the final comparison |
| `summarize_baseline_comparison.py` | Neural-versus-SVM summary statistics |
| `predict_events.py` | Checkpoint inference and event exports |

The `tune_*`, `ablation_test.py`, and loss-comparison utilities remain available
for rerunning individual model comparisons. They share the same subject-level
split and result-recording helpers; use each command’s `--help` for its options.

## Evaluation safeguards

- All samples from a subject remain in one outer fold.
- Rejection thresholds are fitted without the outer validation subjects.
- Compared models reuse identical folds, seeds, and calibration subjects.
- Scaling, PCA, and SVM tuning are fitted inside training data only.
- Result JSONs retain fold assignments, class metrics, confusion matrices,
  thresholds, runtime, parameter counts, and dataset metadata.

The repeated folds reuse the same cohort and are not independent replications.
Confidence intervals and tests in the report are therefore descriptive.

## Full-dataset checkpoints

`train_full_dataset.py` preserves the original full waveform-plus-spectrogram
checkpoint workflow and its historical defaults. It does not train the selected
compact architecture and should not be used to reproduce the reported compact
model comparison. A deployment checkpoint also needs a calibration strategy;
the mean cross-validation threshold is not a universal deployment threshold.
