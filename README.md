# Ripple / IED classification

This folder contains the data preparation, labeling, training, inference, and
analysis code for classifying LFP events as ripple, IED, or noise. The model
code was developed with Python 3.13.

## Repository layout

- `dataset_pipeline/` — candidate extraction, preview/JSON generation,
  K-means selection, manual labeling, curation, transformation, and final PKL
  creation.
- `model_training/` — CNN and autoencoder definitions, cross-validation,
  full-dataset training, inference, and embedding analysis.
- `dataset/` (when generated), `*.csv`, and `*.pkl` — shared dataset artifacts
  at the boundary between the two workflows.
- `*.pt` — trained checkpoints.
- `archive/experiments/` — superseded or incomplete model experiments.

Launch Python and Jupyter from this repository root. The notebooks deliberately
keep shared artifact paths relative to the root.

## Recommended entry points

### Current CNN path

- `model_training/model.py` — multimodal waveform/spectrogram CNN architecture only.
- `model_training/loaders.py` — dataset/event loading, global features, samplers,
  and data loaders.
- `model_training/train.py` — shared losses, epoch/evaluation, folds, and checkpoint helpers.
- `model_training/train_full_dataset.py` — final complete-dataset training entry point;
  writes `final_ripple_model.pt` by default.
- `model_training/cross_validate.py` — subject-wise CV and parameter testing converted
  from `archive/model_training/train_dataset_updated_loss.ipynb`.
- `model_training/predict_events.py` — checkpoint loading, event inference, embedding
  export, and final event-window cleanup.

### Retained autoencoder path

- `model_training/autoencoder.py` — waveform autoencoder and dataset class.
- `model_training/train_autoencoder.ipynb` — autoencoder training and evaluation.

The autoencoder was not the successful model, but it is intentionally retained
as a fallback experiment.

### Data preparation and labeling

- `dataset_pipeline/event_classification.py` — subject-agnostic ripple/IED
  detection and labeling-file generation, converted from the notebook.
- `dataset_pipeline/baseline_spectrograms.py` — subject-agnostic event-free
  background CWT used to normalize event spectrograms.
- `dataset_pipeline/lfp_statistics.py` — subject-agnostic stage-2 waveform
  statistics using the historical 1-300 Hz filter and percentile clipping.
- `dataset_pipeline/ripple_detection.py` and `dataset_pipeline/ied_detection.py`
  — compatibility imports for older code.
- `archive/event_classification.ipynb` — historical exploratory source.
- `archive/baseline_spectrograms.ipynb` — historical exploratory source.
- `dataset_pipeline/label_app.py` — PyQt labeling application.
- `dataset_pipeline/kmeans_selection.py` — current K-means candidate selector;
  writes labeling CSVs with the same columns as `updated_labels.csv`.
- `dataset_pipeline/create_dataset.py` — final labeled-JSON to training-pickle
  conversion. Canonical JSON files are already transformed and background
  corrected, so this step does not process them twice.
- `archive/kmeans_selection.ipynb`, `archive/dataset_curation.ipynb`, and
  `archive/create_dataset.ipynb` — historical exploratory sources.

### Analysis

- `model_training/analyze_embeddings.ipynb` — analysis of exported CNN embeddings.

## Data and checkpoints

- `dataset_arcsinh.pkl` — serialized training samples used by the current CNN
  and autoencoder notebooks.
- `dataset/` — currently empty. It can be regenerated later from the external
  subject/session data; the previous local JSON and JPG copies were redundant.
- `labels.csv` — paths rewritten to the local `dataset/` folder.
- `updated_labels.csv` — original source-system image paths. Replacing each
  `.jpg` suffix with `.json` locates the canonical event record in the external
  subject/session tree.
- `final_ripple_model.pt` — default checkpoint used by
  `model_training/predict_events.py`.
- `new_loss_model.pt` — checkpoint from the archived updated-loss experiment.

## Project-local dependencies

`subjects.py` maps the subject JSON files configured in `settings.py` to the
external recording tree used by data preparation and inference.

The following import is not supplied by PyPI and is not present here:

- `core.py` — expected to provide `power_spectral_density`, `detect_ripples`,
  and `detect_ieds` for three data-preparation notebooks.

The upstream data tree resolved by `subjects.load(...)` is also outside this
folder. Depending on the step, it is expected to contain per-subject files such
as `sleep_stages.npy`, `*_cleaned_hpc.npy`, `*_lfp_stats.npy`, and
`*_baseline_spec.npy`. The current Python pipeline is fixed at 2 kHz and does
not fall back to MATLAB sleep-stage files. `dataset_pipeline/lfp_statistics.py`,
`dataset_pipeline/ripple_envelope.py`, and `dataset_pipeline/ied_envelope.py`
generate the derived statistics and envelope files through the subject-specific
adapter in `dataset_pipeline/subject_data.py`.
Those inputs are not needed to train from the included `dataset_arcsinh.pkl`,
but they are needed to regenerate events from the raw recordings.

Two additional missing modules are referenced only by the archived, incomplete
two-stage experiment:

- `model_projection.py`
- `model_multi_band.py` (expected to provide `SupConLoss`)

Several generated/source data files referenced by historical notebook cells are
also absent: `dataset_updated.pkl`, `label_collapsed.csv`, `labeled.csv`, and
`label_v2.csv`. The current training script uses the included
`dataset_arcsinh.pkl`; `dataset_updated.pkl` is only historical. The existing
`final_ripple_model.pt` can still be used for inference.

Inference and analysis workflows also refer to generated outputs including
`ripple_pred.csv`, `ied_pred.csv`, `ripple_embeddings.npy`, and
`ied_embeddings.npy` under the external subject/session tree. Their absence in
this folder is expected; the inference script creates them.

## Known issues that require a scientific decision

- `labels.csv` has 7,701 rows but only 7,327 unique image paths. There are 374
  duplicate rows, and 143 image paths have conflicting labels. No rows were
  discarded during this cleanup.
- `updated_labels.csv` references 85 external events for which neither the JSON
  nor JPG currently exists. One example is
  `F:\AlzheimerData\106-2\12-08-2023\Sleep1\ripples\ripple1136.json`.
  These may need to be restored or replaced when the dataset is regenerated.
- Separately, the historical K-means notebook reconstructed selected event
  filenames from list positions. That could produce incorrect paths whenever
  event numbering had gaps. The Python replacement uses each selected file's
  actual name.
- The archived updated-loss notebook trained each CV fold on `samples` rather
  than `train_samples` and omitted the subject adversary from the optimizer.
  Both issues are corrected in `model_training/cross_validate.py`; historical
  notebook metrics should therefore not be treated as leakage-free CV results.
- `model_training/predict_events.py` defaults to `final_ripple_model.pt`.
  Choosing `new_loss_model.pt` should be based on newly verified CV results.
- `archive/create_dataset.ipynb` and `archive/kmeans_selection.ipynb` each contain one
  syntactically incomplete exploratory cell. Their main executed pipelines are
  retained, but those cells should not be run as-is.

## Setup

Create and activate a Python 3.13 virtual environment, then install:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

For a CUDA-enabled PyTorch build, use the platform-specific installation command
from the PyTorch installer before installing the remaining requirements.
