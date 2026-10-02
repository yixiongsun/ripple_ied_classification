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
