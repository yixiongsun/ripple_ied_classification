# Dataset pipeline

This folder contains the data-side workflow only:

1. Compute per-channel ripple and IED envelopes from cleaned LFP recordings,
   using sleep-stage samples as the normalization baseline.
2. Build the consensus envelopes (top-three channels for ripples; moving RMS
   across channels for IEDs).
3. Compute the stage-2 LFP statistics used to normalize exported waveforms.
4. Build event-free stage-2 baseline spectrograms for every LFP channel.
5. Detect candidates and optionally retain only the strongest top-k events.
6. Save each candidate as a JPG preview and a JSON record containing the
   transformed waveform and background-corrected spectrogram.
7. Use clustering/K-means selection and `label_app.py` for manual labeling.
8. Package the labeled JSON records into `dataset_arcsinh.pkl`.

Noise samples and labels are retained in the dataset artifacts, but they are not
yet treated as a finalized model class because their inclusion reduced model
accuracy.

Generated `dataset/`, CSV files, and PKL files belong at the repository root
because they are shared artifacts between this pipeline and model training.
Canonical JSON and JPG files remain in the external paths recorded by the local
labeling CSV files.

Run scripts and launch Jupyter from the repository root so existing artifact
paths continue to resolve. For example:

```powershell
python -m dataset_pipeline.subject_data ripple-envelope --subjects 236-1
python -m dataset_pipeline.subject_data ied-envelope --subjects 236-1
python -m dataset_pipeline.subject_data lfp-statistics --subjects 236-1
python -m dataset_pipeline.subject_data baseline-spectrogram --subjects 236-1
python -m dataset_pipeline.subject_data ripple-events --subjects 236-1
python -m dataset_pipeline.subject_data ied-events --subjects 236-1
python -m dataset_pipeline.subject_data label-candidates --output-csv extracted_samples.csv
python -m dataset_pipeline.label_app
python -m dataset_pipeline.create_dataset --labels updated_labels.csv --output dataset_arcsinh.pkl
jupyter lab
```

Delete the generated ripple JSON/JPG directories for one subject after they
are no longer needed:

```powershell
python -m dataset_pipeline.delete_ripple_directories 236-1
```

This permanently removes the complete `ripples` directory from `Sleep1` and
`Sleep2`. Use `--sessions` to restrict the deletion to selected sessions, or
replace the subject ID with `--all` to process every configured subject.

The equivalent command for deleting generated IED JSON/JPG directories is:

```powershell
python -m dataset_pipeline.delete_ied_directories 236-1
```

It supports the same `--sessions` and `--all` options.

The reusable `ripple_envelope.py`, `ied_envelope.py`, and
`event_classification.py` modules accept NumPy arrays and have no knowledge of subjects
or directory layout. `subject_data.py` is the only project-specific adapter; it
loads Sleep1/Sleep2 data, writes envelopes, and calls event export.

The subject runner is intentionally fixed at 2 kHz because event windows and
trained-model inputs use fixed sample counts. It accepts `--sessions` and
`--overwrite`. Event extraction also accepts `--workers` and optional
`--event-top-k K`. Progress bars track subjects, sessions, channels, and event
exports; pass `--no-progress` to suppress them in logs or automated runs.
Ripple consensus uses the strongest three channels by default, configurable
with `--channel-top-k K`.

`kmeans_selection.py` selects up to 100 representative ripples and 100 IEDs
per subject/session by default. Its CSV columns match `updated_labels.csv`:
`subject_id,session_id,image_path,label`. New rows have an empty label for
`label_app.py` to fill.

`label_app.py` retains CSV rows whose images are temporarily missing but hides
them from the labeling view, so opening and re-saving a label file does not
silently discard references that may be regenerated later.

`create_dataset.py` is the final pipeline step. It maps labels to model classes
(`ripple=0`, `ied/ripple_ied=1`, `noise=2`) and packages the already-normalized
JSON arrays into the training pickle. It fails before writing when referenced
JSON files are missing; pass `--skip-missing` only when an intentionally partial
dataset is acceptable.

## Extraction prerequisites

`subject_data.py` requires a project-local `subjects.py` module (or `subjects`
package) providing:

- `get_subjects()` — all available subject IDs.
- `load(subject_id)` — metadata containing `olm.directory` and
  `olm.lfp_channels`.

The project-level `subjects.py` provides this interface using the paths in
`settings.py` and each subject's JSON metadata.

For every subject, session, and LFP channel, extraction expects the following
files under the directory returned by `subjects.load()`:

- `{channel}_cleaned_hpc.npy`
- `sleep_stages.npy`
- `{channel}_lfp_stats.npy`
- `{channel}_baseline_spec.npy`
- `{channel}_ripple_zenv.npy`

Ripple extraction additionally requires `consensus_ripple_env.npy`. IED
extraction additionally requires `{channel}_ied_zenv.npy` and
`consensus_ied_env.npy`.

`ripple_envelope.py` and `ied_envelope.py` create the per-channel z-envelope and
consensus-envelope files through `subject_data.py`. `lfp_statistics.py` creates
the waveform-normalization statistics from 1-300 Hz filtered, percentile-clipped
stage-2 LFP. If a requested session has no stage-2 samples, the subject adapter
copies the statistics from another requested session for the same channel, as
the historical notebook did. `baseline_spectrograms.py` then creates the
baseline-spectrogram files directly from the consensus event windows, avoiding
the notebook's dependency on already-exported JSON files. When one requested
session has too few background samples for a complete CWT window, the subject
adapter copies that channel's baseline spectrogram from another requested
session. It raises an error if no requested session can provide one. No code in
this repository creates `{channel}_cleaned_hpc.npy`; that earlier cleaning step
is still external or missing.

Only `sleep_stages.npy` is supported by the current Python pipeline. Historical
`.mat` sleep-stage files must be converted before running it.

## Deferred improvements

These are plans only; the current numerical pipeline is unchanged.

### Continuous waveform filtering

Event waveforms are currently cropped to 100 ms and then filtered at 1-300 Hz.
The planned change is to filter each complete LFP channel once per session and
crop the already-filtered signal for each event. This avoids the strong boundary
effects of applying a 1 Hz high-pass filter to a very short segment and also
removes repeated filtering work. The preview and CWT branches can continue to
use the original LFP. Before adopting this change, compare transformed waves and
model validation results against the current dataset because regenerated JSON
values will differ.

### Memory-bounded K-means selection

The planned selector will avoid holding every flattened spectrogram and the full
combined feature matrix in memory. It will write or stream float32 features in
batches, fit dimensionality reduction incrementally, train a mini-batch K-means
model, and make a final streamed pass to find the real event nearest each center.
Waveform, spectrogram, and scalar features will remain separately standardized
before their configured weighting. Validate deterministic behavior and inspect
the selected representatives before replacing the current selector.

The baseline calculation will continue concatenating all retained stage-2
samples before forming CWT windows. This is an intentional choice: its purpose
is a robust aggregate background estimate, and the large sample count is
expected to average boundary artifacts out.
