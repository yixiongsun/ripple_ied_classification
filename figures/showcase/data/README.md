# Public plotting data

This directory contains the de-identified tables and compact JSON files used by
the public project page and its result graphics. Numeric precision is retained
in the files and rounded only for display.

| File | Contents |
| --- | --- |
| `dataset_summary.csv` | Event counts and class order |
| `final_model_runs.csv` | Matched compact/full results for every seed and fold |
| `final_class_metrics.csv` | Per-class precision, recall, F1, and support |
| `final_subject_runs.csv` | Results by de-identified subject, seed, and model |
| `threshold_runs.csv` | Calibration-fraction sensitivity |
| `ablation_runs.csv` | Architecture comparisons with evidence level retained |
| `cross_validation_web.json` | Compact payload for the validation graphic |
| `robustness_web.json` | Efficiency and calibration payload |
| `scientific_comparison_web.json` | Candidate rule, SVM, and neural-model comparison |
| `representative_events.json` | De-identified signal examples used by the event figure |
| `manifest.json` | Source hashes, schemas, and row counts |
| `VALIDATION.md` | Human-readable integrity checks |

Subject labels are deterministic aliases based on sorted source identifiers;
the original identifiers are not exported. Screening results and repeated
confirmation results remain explicitly distinguished in `ablation_runs.csv`.

From the repository root, regenerate the tables from their source result files
with:

```powershell
python figures/showcase/scripts/extract_showcase_data.py
```

This command verifies paired folds, locked headline values, class order, row
counts, and source hashes. It also reads the local `labels.csv` for the
threshold-derived candidate-rule comparison. That label file is intentionally
excluded from version control, so the committed derived tables are the public,
reproducible presentation inputs when the private study data are unavailable.
