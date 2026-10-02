# Dataset pipeline

This package converts cleaned hippocampal LFP recordings and sleep-stage labels
into the event records consumed by the modeling code.

## Workflow

1. Compute channel-level ripple and IED envelopes using stage-2 sleep as the
   normalization baseline.
2. Form consensus envelopes across channels.
3. Compute waveform-normalization statistics and event-free baseline
   spectrograms.
4. Detect candidate events and export a preview image plus a JSON record.
5. Select representative candidates for manual labeling.
6. Label candidates with the PyQt application.
7. Package labeled records into `dataset_arcsinh.pkl`.

Run commands from the repository root:

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
```

Use `--help` on any command for optional subject, session, overwrite, worker,
and candidate-count arguments.

## Input contract

`subject_data.py` expects a project-local `subjects.py` module that provides:

- `get_subjects()` returning the available subject identifiers;
- `load(subject_id)` returning metadata with `olm.directory` and
  `olm.lfp_channels`.

For each subject, session, and LFP channel, the pipeline reads or creates:

| File | Purpose |
| --- | --- |
| `{channel}_cleaned_hpc.npy` | Cleaned LFP input; its creation is outside this repository |
| `sleep_stages.npy` | Per-sample sleep-stage labels |
| `{channel}_lfp_stats.npy` | Stage-2 waveform-normalization statistics |
| `{channel}_baseline_spec.npy` | Event-free baseline spectrogram |
| `{channel}_ripple_zenv.npy` | Standardized ripple envelope |
| `{channel}_ied_zenv.npy` | Standardized IED envelope |
| `consensus_ripple_env.npy` | Cross-channel ripple envelope |
| `consensus_ied_env.npy` | Cross-channel IED envelope |

The subject adapter is fixed at 2 kHz because the event windows and model
inputs use fixed sample counts. Only `sleep_stages.npy` is supported; historical
MATLAB sleep-stage files must be converted before use.

If a requested session lacks stage-2 samples, normalization statistics may be
copied from another requested session for the same subject and channel. The
baseline spectrogram follows the same fallback when there are too few complete
background windows. The command fails if no requested session can provide the
required baseline.

## Label and dataset formats

`kmeans_selection.py` selects representative candidates and writes
`subject_id,session_id,image_path,label`. `label_app.py` fills the label column
without dropping rows whose preview image is temporarily unavailable.

`create_dataset.py` maps `ripple` to 0, `ied` and `ripple_ied` to 1, and `noise`
to 2. It fails before writing when a referenced event JSON is missing; use
`--skip-missing` only for an intentionally incomplete dataset.

Generated datasets, labels, previews, and source recordings are local study
artifacts and are not committed. Aggregate metrics derived from them may be
committed after subject identifiers have been removed or replaced.

## Baseline comparison

The threshold-derived candidate rule reported on the project page can be
recomputed from the labeling CSV with:

```powershell
python -m dataset_pipeline.naive_baseline --labels labels.csv
```

The filename prefix records which detector proposed a candidate; this baseline
measures that detector-derived rule against the final three-class labels.
