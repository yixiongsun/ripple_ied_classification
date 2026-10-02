# Showcase captions and alt text

## 00 — Shared visual system

**Caption:** The website visual system uses the portfolio's dark-gray background, white and gray structure, and restrained editorial typography. Blue, orange, and muted purple are reserved for ripple, IED, and uncertain/noise events. It establishes direct labeling and deliberate reflow from the site's 760-pixel desktop column to its 339-pixel mobile content width.

**Alt text:** Dark-gray style tile showing blue ripple, orange IED, muted-purple noise, white selected-model, and gray baseline treatments; Arial and Helvetica typography; directly labeled example marks; and a vertical flow for desktop and mobile.

## 00a — Why the problem is difficult

Hippocampal ripples and interictal epileptiform discharges are brief, variable events whose waveforms and frequency content can overlap, especially in noisy multichannel recordings. Traditional detectors threshold ripple- and IED-band envelopes, but high-frequency transients, large ripple complexes, and artifacts can satisfy the same amplitude or power rules. Those thresholds can locate candidate events without fully describing their shape across time and channels. The classification problem therefore requires learning multichannel morphology, transferring that distinction to entirely held-out subjects, and abstaining when an event does not clearly resemble either class. Noise performance and unseen artifact types remain the principal limitations.

## 01 — End-to-end pipeline

**Caption:** Cleaned multichannel hippocampal recordings are screened for ripple- and IED-like candidates, converted into normalized event records, and manually labeled. In each outer fold, complete subjects are held out for validation; a separate subset drawn only from the remaining training subjects calibrates the confidence threshold. The selected compact model produces a binary ripple-versus-IED score and assigns low-confidence events to noise. The spectrogram is shown to explain event morphology and model development, but it is not an input to the selected compact model. Results shown elsewhere are internal subject-grouped cross-validation, not external validation.

**Alt text:** Five-stage pipeline from multichannel hippocampal recordings through candidate detection and manually labeled 100-millisecond event records to subject-grouped training, calibration, and prediction. The training panel separates model-fitting subjects, calibration subjects, and held-out validation subjects. The final binary ripple-versus-IED score is compared with a calibrated confidence threshold; low-confidence events become noise. A guardrail states that validation subjects never overlap training or set the threshold.

**Source references:** `dataset_pipeline/README.md`, `dataset_pipeline/_event_extraction.py`, `model_training/train.py`, and `model_training/README.md`.

## 02 — Representative events

**Caption:** Three labeled events from different subjects illustrate the morphological overlap behind the classification problem. Each panel shows 300 milliseconds of recording context around the detected event window; the model itself receives the centered 100-millisecond crop. The ripple was selected from the central portion of its class after visual artifact and centering review, the IED is nearest its class centroid, and the ambiguous/noise example is nearest the midpoint between those class centroids. Waveforms are normalized within each event, and baseline-normalized 20–300 Hz spectrograms use one shared color scale. This reproducible selection is a new descriptive analysis, not part of the Phase 9 cross-validation. The spectrogram is included for interpretation; the selected compact model does not use a spectrogram branch.

**Alt text:** Three columns compare a ripple, an interictal epileptiform discharge, and an ambiguous noise event from different subjects. Each column contains three aligned channel waveforms over 300 milliseconds and a colored 20-to-300-hertz spectrogram. A shaded band marks the detected event window. The IED has a sharp central transient, while the ambiguous event shares features with both class centers.

**Selection manifest:** `figures/showcase/data/representative_events.json` records the deterministic dataset indices, private source identifiers, de-identified display labels, feature-space definition, and source-data hash.

## 03 — Final model architecture

**Caption:** The selected `compact_wave_no_temporal` model encodes each waveform channel with a shared one-dimensional CNN, summarizes spike and oscillatory morphology, relates channels with attention, and pools them into a 64-dimensional waveform summary. A separate encoder processes five standardized global features. The paths join in a normalized 64-dimensional embedding and produce one binary ripple-versus-IED logit. Scores with magnitude below a threshold calibrated from training subjects are rejected as noise. The frozen model has 73,985 trainable parameters and no dropout; the spectrogram encoder and temporal convolution were removed after ablation.

**Alt text:** Architecture diagram with a main waveform path and a smaller five-feature path. The waveform path uses a shared per-channel CNN, waveform adapter, morphology summary, cross-channel attention, and max pooling. The paths merge into a normalized 64-dimensional embedding and binary score. A calibrated confidence threshold maps low-confidence scores to noise and confident scores to ripple or IED.

**Source references:** `model_training/ablation_model.py`, `model_training/model.py`, `model_training/loaders.py`, and `model_training/experiment_config.py`.

## 04 — Cross-validation results

**Caption:** Across five subject-level folds and three seeds, the selected compact model and full baseline had nearly identical mean macro F1: 0.8489 and 0.8484, respectively. Each line joins results from an identical seed/fold pairing; the compact model was higher in eight runs and lower in seven, with a mean paired difference of +0.0004. For the compact model, class F1 was 0.896 for ripple, 0.888 for IED, and 0.763 for noise. The repeated runs reuse subjects and represent descriptive internal cross-validation, not 15 independent cohorts or performance on an external test set.

**Alt text:** Two-panel cross-validation summary. The first panel connects full-baseline and compact-model macro F1 for each of 15 matched seed-and-fold runs; the means are nearly identical at 0.848 and 0.849. The second panel shows compact-model class F1 of 0.896 for ripple, 0.888 for IED, and 0.763 for noise, making noise visibly weakest.

**Data source:** `figures/showcase/data/cross_validation_web.json`, generated by `figures/showcase/scripts/extract_showcase_data.py` from the locked Phase 9 result files.

## 05 — Robustness and ablation

**Caption:** The compact model retained essentially the same final-confirmation macro F1 as the full baseline (0.8489 versus 0.8484) while reducing trainable parameters from 113,985 to 73,985 (−35.1%) and mean training time from 222 to 112 seconds (−49.5%). A separate matched robustness experiment compared calibration subsets containing 15%, 20%, and 25% of the training subjects. The 20% fraction had the highest mean macro F1 and was retained, but fitted rejection thresholds remained variable across subject splits and seeds. The threshold is therefore a calibration result, not a universal deployment constant.

**Alt text:** Two-panel model-selection summary. The first panel compares the full and compact models on macro F1, parameter count, and mean training time, showing nearly identical performance with substantially lower compact-model cost. The second panel shows the distribution of 15 fitted rejection thresholds for each of three calibration fractions; 20% is marked as selected, while threshold points remain broadly dispersed.

**Data source:** `figures/showcase/data/robustness_web.json`, generated by `figures/showcase/scripts/extract_showcase_data.py` from final confirmation and threshold-robustness results.

## 06 — Conclusion and further work

**Conclusion:** The waveform model achieved 86.1% accuracy and 0.849 macro F1 in repeated subject-grouped internal cross-validation. Using the same three-class label definition, the threshold-derived candidate rule achieved 62.7% accuracy and 0.435 macro F1, while a direct three-class SVM reached 77.1% accuracy and 0.753 macro F1. Together, these comparisons indicate that learning multichannel event morphology provides information beyond fixed detection thresholds and a conventional classifier. The result remains an internal finding rather than evidence from an independent cohort.

**Further work:** Support a variable number of recording channels so predictions can use all available electrodes without requiring a fixed three-channel input; then move from classifying detected events to predicting ripple or IED occurrence from the neural activity in preceding time windows.

**Comparison data:** `figures/showcase/data/scientific_comparison_web.json`, generated from `labels.csv`, the final repeated-CV results, and `baseline_results/phase10_svm_comparison_summary.json`. Historical `ripple_ied` labels are merged into IED for the threshold-rule comparison so all three approaches use the same ripple/IED/noise label definition.

