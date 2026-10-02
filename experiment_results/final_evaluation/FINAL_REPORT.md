# Model development and internal validation report

## Summary

This study developed a model to classify candidate hippocampal events as
ripples, interictal epileptiform discharges (IEDs), or noise. The final model
uses a binary ripple-versus-IED classifier and rejects low-confidence events as
noise. This approach was retained because it provides an explicit rejection
rule for uncertain events. However, the dataset did not include a separate set
of unseen noise types, so robustness to new artifacts was not directly tested.

The final configuration was evaluated using five subject-level folds and three
random seeds. Mean macro F1 was 0.849, and mean accuracy was 0.861. Macro F1
gives equal weight to ripple, IED, and noise performance, even though the
classes contain different numbers of samples. Performance was strongest for
ripples and IEDs and lower for noise.

The compact model performed similarly to the original full model while using
35% fewer parameters and approximately half the training time. There was no
evidence of a meaningful difference in overall classification performance
between the two models. The compact model was therefore selected based on
efficiency rather than higher predictive accuracy.

These results are internal cross-validation estimates. They should not be
interpreted as performance on an independent test cohort because the same
dataset was used for model development and final comparison.

## Dataset and evaluation

The dataset contained 7,701 labeled events from 30 subjects:

| Class | Events | Percentage |
| --- | ---: | ---: |
| Ripple | 4,178 | 54.3% |
| IED | 1,212 | 15.7% |
| Noise | 2,311 | 30.0% |

All events from one subject were kept in the same fold. This prevents events
from the same subject from appearing in both training and validation data.
Final comparisons used five folds and three random seeds, giving 15 paired
training runs for each model. Each event was evaluated while its subject was
held out from training.

Noise was handled through confidence-based rejection. The model first produced
a ripple-versus-IED score. Events with insufficient evidence for either class
were labeled as noise. For each fold, the rejection threshold was selected
using calibration subjects taken only from the training partition. The held-out
validation subjects were not used to set the threshold.

The main outcome was macro F1. Accuracy and class-specific precision, recall,
and F1 were also examined. Because the folds and repeated seeds reuse the same
subjects, statistical intervals and tests were treated as descriptive rather
than as independent replication.

## Model development

Model development was performed through sequential experiments. Initial screens used
three folds and one seed. Important choices were then checked using five folds
and three seeds where indicated.

### Loss and noise rejection

A preliminary comparison found similar macro F1 for three-class
cross-entropy (0.855) and the binary rejection loss (0.854). The binary method
was retained because it allows uncertain events to be rejected instead of
forcing every event into one of three learned classes.

Within the binary loss, reducing the contrast weight from 0.25 to 0.10 improved
mean macro F1 from 0.849 to 0.855 in the five-fold, three-seed comparison. The
mean difference was 0.0066, but it was not statistically conclusive. Noise
weight 0.5, repulsion weight 0.1, noise margin 0.5, and contrast temperature
0.1 were retained.

### Architecture

The compact architecture used the waveform input and global features but
removed the spectrogram branch and temporal convolution. It retained channel
attention and the spike-oscillation features. During confirmation, macro F1 was
0.853 for the compact model and 0.854 for the full model. The difference was
small relative to variation across folds and seeds. The compact architecture
was retained because it reduced the parameter count from 113,985 to 73,985 and
reduced mean training time from 226 to 116 seconds per run.

### Optimization and regularization

The confirmed optimizer setting was Adam with a learning rate of 0.001 and no
weight decay. A higher learning rate of 0.002 reduced macro F1. Exploratory
tests did not show a consistent benefit from dropout or gradient clipping, so
both were omitted.

A cosine learning-rate schedule had a higher mean than the constant schedule in
one three-fold screen, but it also had greater variation and was not confirmed
across multiple seeds. The simpler constant schedule was retained for 40
epochs. Early stopping performed worse in the exploratory screen.

### Batch composition and representation size

A batch size of 32 with eight subjects represented per batch outperformed a
batch size of 64 in the five-fold, three-seed confirmation. This setting was
retained because the contrastive component of the loss depends on which
subjects and classes are present together in a batch.

An embedding size of 64 slightly outperformed 128 during confirmation and used
less model capacity. An embedding size of 256 performed worse in the initial
screen. The final representation size was therefore set to 64.

### Threshold calibration

Calibration fractions of 15%, 20%, and 25% of the training subjects were
compared. A 20% calibration fraction produced the highest mean macro F1
(0.848) and was retained. The selected thresholds varied considerably between
runs, showing that threshold calibration is an important part of the method
rather than a single fixed value.

## Frozen configuration

| Component | Selected setting |
| --- | --- |
| Architecture | Waveform-only compact model without temporal convolution |
| Global features | Included |
| Channel attention | Included |
| Embedding size | 64 |
| Loss | Binary rejection |
| Contrast weight | 0.10 |
| Noise weight | 0.50 |
| Repulsion weight | 0.10 |
| Noise margin | 0.50 |
| Contrast temperature | 0.10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Schedule | Constant, 40 epochs |
| Dropout | 0 |
| Gradient clipping | Disabled |
| Batch size | 32 |
| Subjects per batch | 8 |
| Calibration fraction | 20% of training subjects |
| Threshold rule | Maximize macro F1 in the calibration subset |

## Final repeated cross-validation results

| Measure | Compact model | Full baseline | Compact minus baseline |
| --- | ---: | ---: | ---: |
| Macro F1 | 0.8489 ± 0.0211 | 0.8484 ± 0.0298 | +0.0004 |
| Accuracy | 0.8611 ± 0.0156 | 0.8619 ± 0.0235 | -0.0008 |
| Weighted F1 | 0.8608 ± 0.0144 | 0.8618 ± 0.0223 | -0.0010 |
| Parameters | 73,985 | 113,985 | -35.1% |
| Mean training time | 112 seconds | 222 seconds | -49.5% |

The compact model had higher macro F1 in 8 of the 15 paired runs and lower
macro F1 in 7. The mean paired macro-F1 difference was 0.0004. The seed-block
95% interval ranged from -0.0048 to 0.0044, and the exact seed-level sign-flip
test gave p = 1.00. These results do not show a meaningful performance
difference between the models.

Class-specific results were:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Ripple | 0.901 | 0.891 | 0.896 |
| IED | 0.889 | 0.898 | 0.888 |
| Noise | 0.774 | 0.757 | 0.763 |

Noise remained the most difficult class. The compact model increased IED F1 by
0.007 and ripple F1 by 0.001 relative to the full baseline, but reduced noise
F1 by 0.006. These differences were small and varied across subjects.

The compact model had higher subject-level macro F1 for 14 subjects and lower
macro F1 for 16 subjects. This subject-level variation indicates that the
aggregate result does not describe every subject equally well. Some subjects
also contained few or no examples of one class, which makes subject-level macro
F1 unstable.

The mean calibrated threshold was 2.77 with a standard deviation of 1.40 and a
range of 0.83 to 5.28. This variation supports recalibrating the threshold when
the model is trained on a new dataset rather than treating 2.77 as a universal
deployment threshold.

## Interpretation

The main finding is that the spectrogram branch and temporal convolution could
be removed without a measurable reduction in repeated cross-validation
performance. The compact model is therefore preferable when computational
efficiency and model simplicity are considered. The final model separated
ripples and IEDs well, but rejection of noise was less reliable.

The binary rejection approach is suitable for the current objective because it
does not require every uncertain event to be assigned to a learned noise class.
It may also be more appropriate when future noise differs from the labeled
training noise. This potential advantage remains a hypothesis because no
unseen-noise dataset was available.

## Limitations

The principal limitation is the absence of an untouched test set. Although all
validation was subject-grouped, the same 30 subjects were used during model
selection and final comparison. The reported values may therefore be
optimistic and should be described as internal validation results.

The dataset was imbalanced, with fewer IEDs than ripples or noise events.
Performance also varied across subjects. In addition, noise labels were treated
as one group, and no separate noise categories were available for testing
generalization to new artifact types.

Several development screens used three folds and one seed. These
screens were intended to remove poor configurations efficiently and should not
be interpreted as definitive comparisons. Sequential tuning can also adapt the
model to this dataset even when each individual evaluation avoids direct
validation leakage.

## Conclusion

The selected model is a compact waveform-based classifier with explicit noise
rejection. In repeated subject-grouped cross-validation, it achieved a macro F1
of 0.849 and accuracy of 0.861. Its performance was indistinguishable from the
larger full model, while it used fewer parameters and approximately half the
training time. The compact model is therefore the preferred configuration for
subsequent full-dataset training.

These results support internal model selection but do not establish external
generalization. Future work should evaluate the frozen model on new subjects
and, when possible, on noise types not represented during training.

## Result files

- [Final repeated cross-validation](final_confirmation_20260930_233532/final_confirmation.json)
- [Per-subject results](final_confirmation_20260930_233532/per_subject_metrics.md)
- [Threshold robustness](../threshold_calibration/robustness_20260930_214618/threshold_robustness.json)
- [Representation-size confirmation](../representation_capacity/embedding_dimension_confirmation_20260930_193604/embedding_dimension_confirmation.json)
- [Batch-composition confirmation](../batch_composition/confirmation_20260930_155359/batch_composition_confirmation.json)
- [Optimizer confirmation](../optimization/confirmation_20260929T210958.192537Z/confirmation.json)
- [Architecture confirmation directory](../architecture/confirmation_20260929T024757.286631Z/)
- [Loss-weight confirmation](../loss_weights/contrast_confirmation_5fold_3seed_20260928.json)

<!-- CLASSICAL_BASELINE_START -->
## Post-hoc classical baseline comparison

This analysis preserves the locked outer folds and calibration subjects. It is an internal post-hoc benchmark, and there is still no untouched test cohort.

| Model | Accuracy | Weighted F1 | Macro F1 |
| --- | ---: | ---: | ---: |
| Frozen compact model | 0.8611 ± 0.0156 | 0.8608 ± 0.0144 | 0.8489 ± 0.0211 |
| Binary RBF-SVM + rejection | 0.6937 ± 0.0401 | 0.6599 ± 0.0538 | 0.6304 ± 0.0547 |
| Direct three-class RBF-SVM | 0.7712 ± 0.0330 | 0.7750 ± 0.0332 | 0.7535 ± 0.0538 |

Paired macro-F1 differences (compact model minus baseline):

- Binary RBF-SVM + rejection: mean 0.2185, median 0.2229, 15 wins / 0 ties / 0 losses; seed-block 95% interval [0.2118, 0.2312], exact seed-level sign-flip p=0.2500.
- Direct three-class RBF-SVM: mean 0.0954, median 0.0725, 15 wins / 0 ties / 0 losses; seed-block 95% interval [0.0790, 0.1125], exact seed-level sign-flip p=0.2500.

The rejection threshold averaged 0.590 (range 0.182–1.017).

Mean class F1 values were:

| Model | Ripple | IED | Noise |
| --- | ---: | ---: | ---: |
| Frozen compact model | 0.8957 | 0.8882 | 0.7627 |
| Binary RBF-SVM + rejection | 0.7869 | 0.7823 | 0.3219 |
| Direct three-class RBF-SVM | 0.8271 | 0.7929 | 0.6404 |

Cross-validation folds and repeated seeds reuse subjects and are not independent experimental units. With only three seed blocks, intervals and sign-flip results are descriptive and have very low power.
<!-- CLASSICAL_BASELINE_END -->
