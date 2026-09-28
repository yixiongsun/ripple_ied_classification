# Ripple / IED Classification

Python tools for detecting and classifying hippocampal local field potential
(LFP) events as ripples, interictal epileptiform discharges (IEDs), or uncertain
noise events.

The repository covers the complete modeling workflow: candidate-event
extraction, spectrogram and waveform preparation, manual labeling, subject-wise
validation, full-dataset training, inference, ablation studies, and loss-function
comparison.

## Highlights

- Multimodal CNN using event waveforms, spectrograms, and global features
- Subject-wise cross-validation to keep recordings from the same subject within
  a single fold
- Explicit uncertainty handling for low-confidence events
- Reproducible architecture ablations and loss-function comparisons
- PyQt-based labeling interface and representative-sample selection
- Committed aggregate experiment results, tables, and figures

## Repository structure

| Path | Purpose |
| --- | --- |
| `dataset_pipeline/` | Event extraction, preprocessing, candidate selection, labeling, and dataset assembly |
| `model_training/` | Model definitions, training, cross-validation, inference, and analysis |
| `ablation_results/` | Saved architecture-ablation metrics and comparisons |
| `cross_validation_results/` | Subject-wise validation results and run metadata |
| `loss_comparison_results/` | Loss-study metrics, rankings, training histories, and plots |

See the detailed workflow guides in
[`dataset_pipeline/README.md`](dataset_pipeline/README.md) and
[`model_training/README.md`](model_training/README.md).

## Installation

The project targets Python 3.13.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build appropriate for the target CUDA
version before installing the remaining dependencies.

## Usage

Run commands from the repository root. Every command-line workflow provides
additional options through `--help`.

### Build a labeled dataset

The data pipeline extracts and transforms candidate events, supports manual
labeling, and packages labeled event records into the training dataset.

```bash
python -m dataset_pipeline.subject_data --help
python -m dataset_pipeline.label_app
python -m dataset_pipeline.create_dataset --help
```

### Run subject-wise cross-validation

```bash
python -m model_training.cross_validate
```

Each run writes a timestamped JSON record to `cross_validation_results/`,
including parameters, subject splits, fold metrics, confusion matrices, and
aggregate metrics.

### Train the final model

```bash
python -m model_training.train_full_dataset
```

The default output is `final_ripple_model.pt`. Existing checkpoints are not
overwritten unless `--overwrite` is supplied.

### Run inference

```bash
python -m model_training.predict_events all
```

Inference exports predicted event tables, embeddings, and final ripple/IED
windows to the configured subject data locations.

### Reproduce experiment comparisons

```bash
python -m model_training.ablation_test --variants all
python -m model_training.compare_ablations
python -m model_training.compare_losses
python -m model_training.plot_loss_comparison
```

The comparison utilities read saved run records and generate ranked tables,
fold-level CSV files, and publication-ready summary plots.

## Data and model files

Raw recordings, derived datasets, labels, and trained checkpoints are not
included in this public repository. They may contain study-specific information
and can be large. The committed experiment-result directories contain only
aggregate outputs used to compare model configurations.

Expected local artifacts include:

- A labeled training pickle, such as `dataset_arcsinh.pkl`
- Labeling CSV files used by the dataset pipeline
- Per-subject LFP and sleep-stage arrays required for event extraction
- Model checkpoints used for inference

These files are excluded through `.gitignore` and should be supplied through an
appropriate controlled data-access process.

## Method notes

The classifier has a binary ripple/IED output. Noise is treated as uncertainty:
noise examples are trained toward an event probability of 0.5 and repelled from
known-event embeddings, while events with low absolute logits are labeled as
noise during inference.

Cross-validation uses subject-level folds. Calibration subjects are drawn from
the training partition, so rejection-threshold selection does not use the outer
validation fold.

## Reproducibility

Saved result JSON files include run parameters, runtime and dataset metadata,
subject splits, fold-level metrics, and aggregate statistics. Use the comparison
scripts to regenerate summary tables and plots from these records.

Exact results can depend on the dataset version, subject composition, hardware,
random seed, and PyTorch/CUDA configuration. Record these details when running
new experiments.

## Citation

If you use this code in research, please cite the associated publication or
project record when one becomes available.
