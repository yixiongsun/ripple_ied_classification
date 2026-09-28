# Model and training

This folder contains the model-side workflow only.

- `model.py` contains only the current CNN architecture.
- `loaders.py` loads training PKLs and event JSON, computes the five global
  features, and creates training/evaluation data loaders.
- `train.py` contains shared loss, epoch, evaluation, subject-fold, and
  checkpoint helpers.
- `train_full_dataset.py` is the runnable replacement for the final full-dataset
  notebook and writes `final_ripple_model.pt` by default.
- `cross_validate.py` is the cleaned subject-wise CV/parameter-testing workflow.
- `ablation_model.py` defines the baseline-compatible configurable model and
  the eight first-pass architecture variants.
- `ablation_test.py` runs those variants on identical subject-wise folds and
  writes a single JSON record with fold, class, threshold, and summary metrics.
- `compare_ablations.py` ranks one or more saved ablation runs and writes CSV,
  text, and PNG comparison outputs.
- `compare_losses.py` compares the seven binary-rejection and three-class loss
  formulations with paired subject folds and threshold calibration.
- `plot_loss_comparison.py` reads saved loss-comparison JSON files and writes
  ranked tables, fold/history CSVs, and performance, training, and threshold
  plots.
- `predict_events.py` loads extracted JSON events, predicts them, saves CSVs and
  embeddings, and creates final ripple/IED window arrays.
- `analyze_embeddings.ipynb` analyzes embeddings produced during inference.

Run from the repository root:

```powershell
python -m model_training.train_full_dataset
python -m model_training.cross_validate
python -m model_training.predict_events all
```

Run a quick baseline smoke experiment or the complete first-pass ablation set:

```powershell
python -m model_training.ablation_test --variants baseline --epochs 1
python -m model_training.ablation_test --variants all
python -m model_training.compare_ablations
python -m model_training.compare_losses
python -m model_training.plot_loss_comparison
```

The ablation runner reserves complete training subjects to calibrate the noise
energy threshold without using the outer validation fold. Pass
`--fixed-threshold` to use `--energy-threshold` directly. Results are saved to
timestamped files under `ablation_results/` unless `--output` is supplied.
The comparison command discovers those files automatically. It prints a ranked
table and creates a timestamped directory containing the aggregate CSV, raw
fold CSV, text ranking, and a fold-aware performance plot. Paths, directories,
and glob patterns can also be passed explicitly.

The loss comparison defaults to all seven formulations and the baseline
architecture. It reserves the same calibration subjects for every formulation,
tunes the rejection threshold only for binary objectives, and saves a
timestamped JSON record under `loss_comparison_results/`. Use
`--architecture wave_only` or another named ablation to repeat the loss study
on a simplified architecture.
The loss plotting command automatically discovers `losses_*.json` files under
`loss_comparison_results/`. Explicit JSON paths, directories, and glob patterns
are also accepted, and `--metric` selects the ranking metric.

All scripts accept `--help`. Full training defaults to `dataset_arcsinh.pkl`,
40 epochs, batch size 32, learning rate 3e-4, and an energy threshold of 3.13.
It will not replace an existing checkpoint unless `--overwrite` is passed. The
shared dataset PKL and checkpoints remain at the repository root.

The CV script exposes the contrast, noise, repulsion, margin, subject-adversary,
and energy-threshold settings as command-line options so parameter tests do not
require editing source code. Its default training parameters match the full-dataset
training script; CV-specific subject-aware batching and folds remain separate.
Each run also writes a timestamped JSON file under
`cross_validation_results/`. The file records all run parameters, dataset and
runtime metadata, subject splits, per-fold metrics and confusion matrices, and
aggregate metrics. Use `--output PATH` to choose a specific filename; an existing
file is only replaced when `--overwrite` is also supplied.

To cap the CUDA memory available to cross-validation, pass a fraction greater
than zero and at most one, for example `--gpu-memory-fraction 0.5`. This limits
the process to approximately half of the GPU's memory. Reduce `--batch-size` if
the selected limit causes an out-of-memory error.

Cross-validation defaults to `--workers 0`. On Windows, data-loader subprocesses
use process spawning and can duplicate the large in-memory dataset, causing a
system `MemoryError`. A positive worker count can still be selected explicitly
on systems with sufficient RAM.

The classifier has a binary ripple/IED output. Noise remains an uncertainty
class: noise samples are trained toward probability 0.5 and repelled from known
event embeddings, then low-absolute-logit events are labeled as noise during
inference.
