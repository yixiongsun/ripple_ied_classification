# Ripple / IED classification

This repository contains the signal-processing, labeling, modeling, and
evaluation code used to classify candidate hippocampal events as sharp-wave
ripples, interictal epileptiform discharges (IEDs), or noise.

The selected model uses multichannel waveforms, cross-channel attention, five
global signal features, and a calibrated confidence threshold that rejects
uncertain events as noise. The spectrogram branch is used during data review
but is not part of the selected classifier.

## Results

The dataset contains 7,701 labeled events from 30 subjects: 4,178 ripples,
1,212 IEDs, and 2,311 noise events. Five subject-grouped folds were repeated
with three random seeds. Threshold calibration used subjects drawn only from
the corresponding training partition.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Selected waveform model | 0.861 | 0.849 |
| Full waveform + spectrogram model | 0.862 | 0.848 |
| Direct three-class RBF-SVM | 0.771 | 0.753 |
| Binary RBF-SVM with rejection | 0.694 | 0.630 |
| Threshold-derived candidate rule | 0.627 | 0.435 |

The selected model has 73,985 trainable parameters, 35.1% fewer than the full
model, and took about half as long to train in the final comparison. Noise was
the least reliable class (mean F1 0.763). These are internal cross-validation
results, not estimates from an independent cohort.

The full analysis is in
[`experiment_results/final_evaluation/FINAL_REPORT.md`](experiment_results/final_evaluation/FINAL_REPORT.md).
The public presentation is available at
[Neural event classification](https://yixiongsun.github.io/work/neural-event-classification/).

## Repository layout

| Path | Contents |
| --- | --- |
| `dataset_pipeline/` | Event detection, preprocessing, sample selection, labeling, and dataset assembly |
| `model_training/` | Model definitions, grouped validation, frozen evaluation, inference, and classical baselines |
| `experiment_results/` | Machine-readable experiment records and concise summaries |
| `tests/` | Leakage, determinism, metric, and restartability checks |

## Installation

The project targets Python 3.13.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for the target CUDA version first when a
specific GPU build is required.

## Data required for model runs

Model evaluation requires `dataset_arcsinh.pkl` at the repository root. The
file is intentionally not versioned because it is approximately 2.4 GB and is
derived from study recordings. It must contain event dictionaries with a
subject identifier, class label, waveform, spectrogram, and the scalar inputs
used by `model_training/loaders.py`.

The extraction pipeline also requires per-subject cleaned LFP and sleep-stage
arrays. See [`dataset_pipeline/README.md`](dataset_pipeline/README.md) for the
input contract and dataset-building commands.

## Reproduce the reported evaluation

Run commands from the repository root.

```powershell
python -m unittest discover -s tests
python -m model_training.final_confirmation --dataset dataset_arcsinh.pkl
python -m model_training.compare_classical_baselines --dataset dataset_arcsinh.pkl --folds 5 --seeds 0 1 2 --output experiment_results/baselines/svm_comparison.json
```

The locked comparison uses five subject-grouped folds, seeds 0–2, 40 epochs,
Adam at a learning rate of 0.001, batch size 32, eight subjects per batch, and a
20% training-side calibration split. The configuration is defined in
`model_training/experiment_config.py`.

`model_training/train_full_dataset.py` preserves the earlier full-model
checkpoint workflow; it is not the command that produced the selected compact
model results above.

## Data and reporting boundaries

Raw recordings, labels, dataset pickles, subject metadata, trained checkpoints,
and portfolio figure assets are excluded from version control. Commit aggregate
result JSONs and concise scientific summaries; do not commit training logs,
resumable partial files, smoke-test outputs, or local planning notes.

Because the same 30 subjects informed model development and the final grouped
comparison, the results may be optimistic. External subjects and previously
unseen artifact types are needed before treating the reported threshold or
performance as deployment estimates.
