# Experiment results

This directory contains the versioned outputs used to support the reported
model selection and internal-validation findings.

| Directory | Contents |
| --- | --- |
| `architecture/` | Neural architecture comparisons and ablations |
| `baselines/` | Classical SVM and detector-derived baselines |
| `batch_composition/` | Batch-size and subject-composition comparisons |
| `cross_validation/` | General subject-grouped validation runs |
| `final_evaluation/` | Final report, locked model comparison, and per-subject metrics |
| `loss_comparison/` | Loss-function comparisons and generated summaries |
| `loss_weights/` | Contrastive and uncertainty-loss weight comparisons |
| `optimization/` | Learning-rate and weight-decay experiments |
| `regularization/` | Dropout and gradient-clipping experiments |
| `representation_capacity/` | Embedding-size comparisons |
| `threshold_calibration/` | Calibration-fraction sensitivity analysis |
| `training_schedule/` | Schedule and stopping-policy comparisons |

Completed JSON records are the source of truth. Console logs, resumable partial
files, and smoke-test outputs are local artifacts and are excluded from version
control.
