# Beyond Validation Scores: Patient-Independent Generalization, Robustness and Uncertainty in Respiratory-Sound Classification

**Rishabh Mishra**
Independent Researcher, Vaughan, Ontario, Canada

Independent research manuscript (preprint), version 1.0. Not peer reviewed.
DOI: [Zenodo DOI to be added after publication]
Code, frozen protocol and locked results: https://github.com/rish-mishra/acoustic-biomarker-classifier (software release v1.0.1; model checkpoint attached to release v1.0.0)

> This is a research and educational study. The model described here is not a medical device or a diagnostic system, it has not been clinically validated, and nothing in this paper supports its use for diagnosis or medical decisions.

---

## Abstract

Automatic classification of lung sounds can look accurate on paper. That accuracy can be misleading, though, when the same patients appear in training and evaluation data, or when models are evaluated only on clean recordings. This paper asks how well a respiratory-sound classifier generalizes to completely unseen patients and to degraded audio, and whether its own uncertainty can identify predictions that should not be trusted.

Using the ICBHI 2017 Respiratory Sound Database (126 patients, 6,898 annotated respiratory cycles), I classified each cycle as normal, crackle, wheeze or both under a fixed patient-level split of 88/19/19 patients. Three architectures were compared on validation data only: a small convolutional network (CNN), a convolutional-recurrent network and an ImageNet-pretrained MobileNetV3. A predeclared three-seed study found no architecture convincingly better than the others, so the most stable and simplest model, the baseline CNN, was kept as the reference. Every model choice, calibration temperature, abstention threshold and degradation condition was then frozen and hash-verified before the test patients were evaluated, exactly once.

Macro F1 fell from 0.4271 on validation to 0.3175 on the untouched test patients (patient-level bootstrap 95% CI 0.218–0.413). The mixed "both" class essentially failed (F1 0.041), and more than half of the test normal cycles were predicted as crackle. Under 15 predeclared degradations, several failure patterns reproduced from validation to test, including the collapse of predictions onto crackle under heavy noise. Temperature scaling fitted on validation reduced the average overconfidence gap on test by about 40% (from +0.208 to +0.127). However, uncertainty remained a weak detector of individual errors (AUROC 0.542–0.554), and abstention reduced the error rate only modestly. The results suggest that clean validation performance alone substantially overstates how such classifiers behave on new patients and recording conditions.

**Keywords:** respiratory sounds; lung auscultation; ICBHI 2017; patient-independent evaluation; domain shift; robustness; calibration; uncertainty estimation; selective prediction

---

## 1. Introduction

When a clinician listens to a patient's chest through a stethoscope, part of what they listen for are *adventitious* sounds layered on top of normal breathing. Two of the most commonly described are crackles, which are short, discontinuous, popping sounds, and wheezes, which are longer, continuous, musical tones [1]. Respiratory-sound classification tries to recognize these patterns automatically from recordings. In the benchmark used here [2, 3], each annotated breath, or *respiratory cycle*, is labelled as normal, crackle, wheeze, or both when crackles and wheezes occur in the same cycle.

Machine-learning models for this task can look strong on paper without necessarily learning what we hope they learn. Respiratory recordings carry a great deal of information unrelated to the breath sounds themselves: the stethoscope or recorder that was used, its sample rate and frequency response, background noise, how hard the sensor was pressed, and the individual patient's anatomy. If recordings from the same patient appear in both the training and evaluation data, a model can do well partly by recognizing the patient or their recording setup, and the evaluation will not reveal it. In clinical machine learning more broadly, evaluations that let records from the same subject fall on both sides of the split have been shown to overestimate accuracy, often substantially [4]. Splitting data by patient removes that shortcut for patient identity, but it does not remove shortcuts based on the recording device. Differences in recording-device properties are a recognized issue in the ICBHI 2017 dataset [5], and in that dataset device is also entangled with the patients' diagnoses (Section 2).

A second issue is trust. A classifier outputs class probabilities, and it is tempting to treat a high probability as a sign that the prediction is reliable. For that to be useful, the probabilities should be *calibrated*, meaning that predictions made with 70% confidence should be right about 70% of the time, and more importantly, low confidence should actually point to the predictions that are wrong. Neither property is guaranteed. Modern neural networks are often poorly calibrated [6], and the quality of their uncertainty estimates tends to degrade when inputs differ from the training data [7].

This paper is built around one question:

> How well can respiratory-sound classifiers generalize to completely unseen patients and degraded real-world audio, and can uncertainty estimation identify predictions that should not be trusted?

The goal was not to build a clinically useful classifier, and the results below should not be read that way. Instead, I set up an evaluation that is deliberately harder to fool than a single random split. Patients never cross between training, validation and test data. All development decisions were made on validation data. The test patients stayed untouched until a protocol fixing every remaining choice had been frozen and hash-verified, and were then evaluated exactly once. On top of the clean evaluation, the same frozen model was tested under 15 synthetic audio degradations, and its calibration, uncertainty and abstention behaviour were assessed with settings fixed in advance.

The main finding is a negative but, I think, useful one. A model that looked reasonable on validation patients performed substantially worse on new patients. It failed in systematic ways under degraded audio, and its confidence provided little help in recognizing its own errors. These problems are easy to miss under a less strict evaluation, which is the main reason they are worth reporting.

## 2. Dataset

The study uses the ICBHI 2017 Respiratory Sound Database [2, 3], which contains 920 annotated recordings from 126 patients, segmented into 6,898 respiratory cycles. Each cycle is annotated for the presence of crackles and wheezes, which gives the four classes used throughout: normal (3,642 cycles, 52.8%), crackle (1,864, 27.0%), wheeze (886, 12.8%) and both (506, 7.3%). The imbalance matters for evaluation, because a model can reach respectable accuracy by favouring the normal class while doing poorly on the rarer adventitious classes.

Several properties of the dataset matter as much as its size.

**Recording devices and sample rates.** The recordings were made with four devices, identified in the dataset as AKGC417L (4,346 cycles), Meditron (1,456), LittC2SE (594) and Litt3200 (502), at three sample rates: 44.1 kHz (5,821 cycles), 4 kHz (1,016) and 10 kHz (61). Sample rate limits the frequencies a recording can contain: a 4 kHz recording carries nothing above 2 kHz, whatever was present in the original sound. Differences in frequency content between devices are therefore a potential shortcut, since a model could learn to recognize a recording setup rather than a breath sound.

**Device and diagnosis are confounded.** The dataset's metadata shows that device choice is strongly tied to diagnosis. Every recording from patients diagnosed as healthy, or with upper or lower respiratory tract infection, bronchiectasis or bronchiolitis, was made with the Meditron. The AKGC417L and Litt3200 were used only with patients diagnosed with chronic obstructive pulmonary disease (COPD). Because adventitious sounds are themselves more common in some diagnoses than others, device, diagnosis and label distribution are entangled, and no split of this dataset can fully separate them.

**Clipping.** Many recordings are clipped: the signal reached the recorder's maximum level and was flattened. Rather than discard or repair these cycles, I measured clipping on each annotated cycle at its native sample rate. Two measures were used: the fraction of samples with absolute amplitude of at least 0.999, and the number of clipped runs lasting at least 32 ms. Three quality groups were defined on 2026-10-03, before any model was trained: *all* cycles; *moderately clean* cycles (clipped fraction ≤ 0.10); and *clean* cycles (clipped fraction ≤ 0.01 and no clipped run of 32 ms or longer). These groups filter evaluation cycles only. They never change the split.

**Leading digital silence.** Most AKGC417L recordings begin with roughly 90 ms of samples that are exactly zero, a pattern not shown by the other devices. Converted into a spectrogram, these become frames at the floor value, which would act as a device fingerprint. To remove this shortcut, leading exact-zero samples are trimmed from every cycle on every device (Section 4).

## 3. Experimental Design

### 3.1 Patient-level split

The patients were split once into training (88 patients, 613 recordings, 4,741 cycles), validation (19 patients, 138 recordings, 1,077 cycles) and test (19 patients, 169 recordings, 1,080 cycles) sets. No patient contributes data to more than one split. Within that constraint, the split was chosen from 5,000 seeded random candidates as the one whose label, device and sample-rate distributions best matched those of the full dataset. Diagnosis was not used in building the split and is not a prediction target. Matching the device mix keeps the test focused on new patients rather than mixing patient generalization with a change in recording devices. It reduces, but does not remove, the device confound described above.

### 3.2 Roles of the splits and the frozen protocol

The training set was used to fit model parameters and the per-band normalization statistics. Every development decision was based on the validation set: architecture and hyperparameters, the number of training epochs, the seed study, the calibration temperature and the abstention thresholds. Each validation analysis followed a plan written before its results were computed.

The test set was kept completely untouched during development. I was trying to avoid a common and often unintentional form of overfitting, in which test scores are checked repeatedly and the model or analysis is adjusted until the number looks good. Before any test data were loaded, a final evaluation protocol fixed the following:
- the model checkpoint, identified by its SHA-256 hash;
- the data pipeline;
- the primary and secondary metrics;
- the calibration temperature and abstention thresholds;
- the 15 degradation conditions;
- the bootstrap procedure.

The protocol also recorded an audit confirming that no earlier artifact had touched the test split, together with content hashes of every frozen file. The evaluation software would only read test data with an explicit unlock step from a clean, committed code state. Before unlocking, it had to reproduce the reference validation predictions bit for bit. It then ran the clean test evaluation once, wrote and hash-locked the results, and only afterwards ran the degraded test conditions. Section 13 describes these safeguards in more detail.

### 3.3 Metrics and confidence intervals

The primary metric is **macro F1** over the four classes. F1 combines precision and recall for a single class, and macro F1 averages it across classes with equal weight. A model therefore cannot score well by doing well only on the dominant normal class. Classes with no correct predictions contribute an F1 of zero. Balanced accuracy (mean per-class recall), overall accuracy, per-class precision, recall and F1, and the confusion matrix are reported as secondary metrics.

Respiratory cycles from the same patient are not independent observations, so confidence intervals are computed with a **patient-level bootstrap**. Each of 2,000 resamples (seed 0) draws patients with replacement and includes all of their cycles; the 95% interval is the 2.5th to 97.5th percentile of the resampled statistic. With only 19 patients per evaluation split, these intervals are wide, and I treat them as a measure of uncertainty rather than as a basis for significance claims.

## 4. Preprocessing and Input Representation

Each respiratory cycle is processed in the following steps.
1. **Slicing.** The cycle is cut from its recording at the recording's native sample rate, using the annotated start and end times. The audio is read as 32-bit floating-point values in the range −1 to 1, and multi-channel files are averaged to mono.
2. **Trimming.** Any leading samples that are exactly zero are removed, up to 150 ms, always keeping at least one sample. This is done for every cycle, not just AKGC417L ones, so the operation itself carries no device information.
3. **Resampling.** The cycle is resampled to 16 kHz with a windowed-sinc resampler.
4. **Centring.** Its mean is subtracted, so a constant offset in the signal cannot show up as low-frequency energy.

The waveform is then converted into a **log-Mel spectrogram**. A short-time Fourier transform with a 32 ms Hann window (512 samples) and a 10 ms hop measures how much energy the sound contains at each frequency over time. Those frequencies are grouped into 64 bands spaced on the mel scale, which is roughly linear at low frequencies and logarithmic at higher ones, much like human pitch perception. Band energies are converted to decibels with a floor of −100 dB. The bands cover 50–2000 Hz. The 2 kHz upper limit is deliberate: it is the highest frequency a 4 kHz recording can contain, so every cycle fills the whole representation, and band-limited devices cannot be identified from empty upper bands. The 50 Hz lower limit excludes the lowest band, which mostly holds offset, handling noise and heart sounds.

Finally, each mel band is standardized to zero mean and unit variance using statistics computed on the training patients only. Cycles keep their natural duration rather than being cropped or stretched to a fixed length. Within a mini-batch, shorter cycles are padded, and a mask records which frames are real. All three models use this mask so that a cycle's prediction does not depend on how much padding its batch happened to need.

## 5. Models and Training

Three architectures were compared. All of them take the same standardized log-Mel input and output four class scores.

The **baseline CNN** (294,740 parameters) has four convolutional blocks with 16, 32, 64 and 128 channels. Each block applies two 3×3 convolutions, each followed by group normalization and a ReLU, and then 2×2 max pooling. The normalization statistics are computed over real frames only, so padding cannot leak into them. After the last block, the network averages over frequency, takes both the masked mean and the masked maximum over time, and classifies the result with a dropout layer and a linear layer.

The **convolutional-recurrent network (CRNN)** (353,365 parameters) tests whether explicitly modelling temporal structure helps. Its convolutional front end mirrors the first three blocks of the baseline but pools time less aggressively, which keeps 20 ms resolution for short events such as crackles. A bidirectional GRU then reads the sequence, followed by masked attention pooling over time. Its size is close to the baseline's, so the comparison is mostly about temporal modelling rather than capacity.

The **pretrained MobileNetV3** (1,521,668 parameters) tests transfer learning. It is the small variant of MobileNetV3 [8] initialized with ImageNet weights. Its first convolution was adapted to a single input channel by summing the pretrained filters over their colour channels, and its final layer was replaced with a four-class output. As with the other models, padded positions are masked throughout.

All models were trained identically:
- **Loss:** class-weighted cross-entropy, with weights inversely proportional to class frequency in the training set.
- **Optimizer:** AdamW [9] with weight decay 0.01, batch size 32 and gradient clipping at 5.0.
- **Batching:** training batches were drawn with a length-bucketing sampler, which groups cycles of similar duration to reduce padding while still shuffling every epoch.
- **Stopping:** training ran for at most 60 epochs with early stopping on validation macro F1, with a patience of 10 epochs. The checkpoint from the best epoch was kept.
- **Determinism:** random seeds, deterministic cuDNN settings and deterministic algorithms were fixed, so reruns reproduce identical predictions on the same hardware and software.

## 6. Model Selection and Multi-Seed Study

Model selection proceeded in two stages, both on validation data only.

The first stage tuned the baseline CNN in a small learning-rate and dropout study (four runs, seed 42). All four runs were declared before any of them was evaluated, and the plan ruled out adding runs on the basis of their results. The best configuration, `baseline_cnn_v3` (learning rate 1×10⁻⁴, dropout 0.3), reached a validation macro F1 of 0.4271. The CRNN and MobileNetV3 were then trained with the same settings, reaching 0.4153 and 0.4381 respectively.

These single-run differences were small, and a single training run says little about how much a result depends on random initialization and data order. Before choosing an architecture, I therefore retrained each one with two further seeds. The decision rule was written down first. An architecture would be preferred only if it beat every other architecture on all three seeds, and by more than the pooled within-architecture standard deviation across seeds. Otherwise the simplest architecture whose mean was within one pooled standard deviation of the best mean would be kept. The results are shown in Table 1 and Figure 1.

**Table 1.** Validation macro F1 by architecture and seed (best epoch).

| Architecture | Parameters | Seed 42 | Seed 1337 | Seed 2026 | Mean ± SD |
|---|---|---|---|---|---|
| Baseline CNN | 294,740 | 0.4271 | 0.4226 | 0.4225 | 0.424 ± 0.003 |
| CRNN | 353,365 | 0.4153 | 0.4367 | 0.3869 | 0.413 ± 0.025 |
| MobileNetV3 (pretrained) | 1,521,668 | 0.4381 | 0.4372 | 0.4194 | 0.432 ± 0.011 |

![Figure 1](../reports/final_report/figures/seed_comparison.svg)

**Figure 1.** Validation macro F1 of the three architectures for seeds 42, 1337 and 2026. Dots show individual seeds; the vertical tick marks the mean, labelled with mean ± standard deviation across seeds. Validation data only.

The pooled within-architecture seed standard deviation was 0.0157, while the three architecture means spanned only 0.0186. MobileNetV3 had the highest mean, but it beat the baseline CNN on only two of the three seeds, by an average margin smaller than the seed noise. No architecture met the predeclared criterion, so the rule kept the simplest architecture within the noise range, the baseline CNN. It was also by far the most stable across seeds. Given how small the differences were, a roughly five-fold larger pretrained model was not justified by its slightly higher point estimate.

The frozen reference model is **`baseline_cnn_v3`, seed 42, best epoch 13**: the configuration and run chosen in the first stage, before the seed study took place. The seed study was used to check that this choice was stable, not to search for a more favourable seed. The seed-1337 and seed-2026 runs were never evaluated on test data.

## 7. Final Untouched-Test Evaluation

The clean test evaluation was run once, under the frozen protocol, on 1,080 cycles from the 19 test patients. Table 2 compares it with the same model's validation result.

**Table 2.** Clean performance of the frozen reference model (`baseline_cnn_v3`). Test interval: patient-level bootstrap 95% CI.

| Metric | Validation (19 patients) | Test (19 unseen patients) |
|---|---|---|
| Macro F1 (primary) | 0.4271 | **0.3175** (0.218–0.413) |
| Balanced accuracy | 0.4568 | 0.4030 (0.325–0.522) |
| Accuracy | 0.5097 | 0.3935 (0.293–0.527) |

**Table 3.** Per-class test results.

| Class | Precision | Recall | F1 | Test cycles |
|---|---|---|---|---|
| Normal | 0.732 | 0.278 | 0.403 | 647 |
| Crackle | 0.333 | 0.725 | 0.456 | 255 |
| Wheeze | 0.274 | 0.570 | 0.370 | 100 |
| Both | 0.043 | 0.038 | 0.041 | 78 |

![Figure 2](../reports/final_report/figures/test_confusion_matrix.svg)

**Figure 2.** Confusion matrix of the frozen reference model on the clean test set (1,080 cycles, 19 patients). Rows are true classes and columns are predicted classes. Each cell shows the number of cycles, and its shading and percentage give the share of that row (the class's recall).

Test macro F1 was 0.110 lower than validation macro F1, and the validation value lies just above the upper end of the test interval. Development-set selection may contribute to this gap, alongside patient and recording-domain differences between the splits. The reference configuration was chosen because it did best on validation, which can make its validation score optimistic. The study was not designed to separate these contributions. With only 19 test patients, the size of the gap is itself uncertain, but the drop together with the changed pattern of errors (below) is difficult to attribute to random variation alone.

The pattern of errors is more informative than the headline number. The "both" class essentially failed: only 3 of 78 test cycles were correctly identified, and 46 were predicted as wheeze. The largest single source of error, however, was normal cycles: 342 of the 647 normal test cycles (53%) were predicted as crackle, and normal recall fell from 0.50 on validation to 0.28 on test. Crackle and wheeze recall held up comparatively well (0.725 and 0.570). As a result, the model predicted crackle for 556 test cycles, more than twice the 255 that actually were crackles.

Recording quality does not explain the gap. Restricting the test set to the predeclared quality groups barely changed the result: macro F1 was 0.3175 for all 1,080 cycles, 0.3176 for the 1,062 moderately clean cycles and 0.3317 for the 949 clean cycles. The intervals of all three groups overlap almost completely.

## 8. Robustness to Audio Degradation

### 8.1 Degradation conditions

To probe how the frozen model responds to degraded audio, I defined 15 conditions in advance and applied each one to the 16 kHz waveform before feature extraction. The definitions were fixed in a validation plan before any degraded result was computed, and were not changed afterwards:
- **Gaussian noise** at 20, 10, 5 and 0 dB signal-to-noise ratio. The noise is white across the full band and rescaled so that the requested SNR is met exactly.
- **Gain changes** by factors of 0.5, 0.25, 1.5 and 2, without clipping. Gain changes the loudness of a recording but not its spectral shape or timing.
- **Low-pass filtering** at 2000, 1000 and 500 Hz, using a zero-phase windowed-sinc FIR filter.
- **Temporal dropout** of 10%, 25% and 50%, in which randomly chosen 20 ms blocks are set to silence.
- The **unmodified clean** condition.

The randomness in the noise and dropout conditions is seeded for each cycle, so the same cycle always receives the same degradation. The conditions were first evaluated on validation data during development. On test data they were run only after the clean test results had been written and hash-locked, and the robustness run first had to reproduce the locked clean predictions exactly.

### 8.2 Results

**Table 4.** Macro F1 under the predeclared degradations. Test changes are relative to the clean test result, with patient-level bootstrap 95% CIs.

| Condition | Validation | Test | Test change vs clean (95% CI) |
|---|---|---|---|
| Clean | 0.427 | 0.3175 | — |
| Noise 20 dB | 0.358 | 0.243 | −0.074 (−0.120, −0.021) |
| Noise 10 dB | 0.309 | 0.196 | −0.122 (−0.192, −0.038) |
| Noise 5 dB | 0.240 | 0.161 | −0.156 (−0.221, −0.072) |
| Noise 0 dB | 0.208 | 0.137 | −0.180 (−0.253, −0.090) |
| Gain 0.5× | 0.405 | 0.327 | +0.009 (−0.041, +0.052) |
| Gain 0.25× | 0.341 | 0.314 | −0.004 (−0.070, +0.051) |
| Gain 1.5× | 0.411 | 0.310 | −0.007 (−0.033, +0.010) |
| Gain 2× | 0.410 | 0.297 | −0.021 (−0.050, +0.003) |
| Low-pass 2000 Hz | 0.428 | 0.318 | +0.001 (+0.000, +0.002) |
| Low-pass 1000 Hz | 0.173 | 0.157 | −0.161 (−0.210, −0.102) |
| Low-pass 500 Hz | 0.240 | 0.246 | −0.072 (−0.160, +0.002) |
| Dropout 10% | 0.384 | 0.308 | −0.009 (−0.061, +0.037) |
| Dropout 25% | 0.256 | 0.277 | −0.040 (−0.129, +0.039) |
| Dropout 50% | 0.166 | 0.200 | −0.118 (−0.234, +0.002) |

![Figure 3](../reports/final_report/figures/robustness_validation_vs_test.svg)

**Figure 3.** Macro F1 of the frozen reference model under the 15 predeclared degradations, shown separately for each degradation family. Blue: validation (development); orange: the untouched test patients. The leftmost point in each panel is the clean condition.

Gaussian noise was the most damaging family on test, with macro F1 falling steadily to 0.137 at 0 dB. The 1000 Hz low-pass filter (0.157) and 50% temporal dropout (0.200) were also severe.

The way the model failed is as informative as how much it failed. Under heavy degradation, predictions tended to collapse onto a single class, and the same collapses appeared on both validation and test:
- **Heavy noise → crackle.** At 0 dB, 94% of test predictions were crackle (85% on validation).
- **Severe dropout → normal.** At 50% dropout, 97% of test predictions were normal (99% on validation).
- **1000 Hz low-pass → wheeze.** 67% of test predictions were wheeze, and crackle predictions almost disappeared.
- **500 Hz low-pass → normal.** 71% of test predictions were normal.

These collapses also explain why some accuracy figures look less alarming than macro F1. Normal makes up 60% of the test set, so predicting normal for almost everything keeps accuracy deceptively high: it was 0.585 under 50% dropout, while macro F1 was 0.200.

One result was unexpected: the 1000 Hz low-pass filter was more damaging than the 500 Hz filter, on validation (0.173 against 0.240) and again on test (0.157 against 0.246). Removing more of the signal therefore did not produce a larger drop. I do not have an explanation for this non-monotonic behaviour. I report it as an observed pattern that reproduced across splits, together with the different classes the predictions collapsed onto at the two cut-offs.

The 2000 Hz low-pass filter served as a control. Because the log-Mel representation stops at 2 kHz, this filter removes essentially nothing the model sees, and macro F1 changed by less than 0.001 on both splits, as expected.

Gain changes are informative for a different reason. Loudness carries no information about whether a crackle or wheeze is present, yet changing it moved the model's predictions. On test, attenuating to 0.25× nearly doubled the number of normal predictions (246 to 500), and amplifying to 2× doubled the number of "both" predictions (70 to 142). On test the net effect on macro F1 was small. On validation, though, 0.25× gain cost 0.086, and the shift in predictions is itself a sign that the model is sensitive to absolute recording level.

## 9. Calibration, Uncertainty and Selective Prediction

### 9.1 Definitions

For each cycle, the model produces a probability for each class. Its **confidence** is the largest of these probabilities. The **predictive entropy** measures how spread out the four probabilities are: it is low when one class dominates and high when the model hedges. Both can serve as uncertainty scores, where low confidence or high entropy suggests that a prediction is less trustworthy.

A model is well **calibrated** if its confidence matches how often it is right. **Expected calibration error (ECE)** summarizes this. Predictions are grouped into 15 equal-width confidence bins, the gap between average confidence and accuracy is computed within each bin, and the gaps are averaged with weights proportional to bin size. I also report the **overconfidence gap**, the mean confidence minus the accuracy, which captures whether the model is too sure of itself on average.

**Temperature scaling** [6] adjusts calibration with a single parameter T. The model's output scores are divided by T before conversion to probabilities, so T > 1 makes the model less confident. Because the same T rescales every class score equally, it never changes which class is predicted, and accuracy and macro F1 are unaffected. On validation, T was estimated with leave-one-patient-out cross-validation: each patient's temperature was fitted on the other 18 patients, so the calibrated validation results are out-of-fold estimates. The 19 fold temperatures were stable, ranging from 1.23 to 1.60 (SD 0.076). A single temperature fitted on all 19 validation patients, **T = 1.3856**, was then frozen and applied unchanged to the test data.

**Error detection** asks whether uncertainty ranks wrong predictions above right ones. It is measured by the area under the ROC curve (AUROC) for separating errors from correct predictions. Using the maximum softmax probability in this way is a standard baseline for detecting misclassified examples [10]. An AUROC of 0.5 means the uncertainty score is no better than chance at telling errors from correct predictions.

**Selective prediction**, or abstention, lets the model decline to answer when it is unsure. Cycles are ranked by confidence and only the most confident fraction (the *coverage*) is kept; the error rate among kept cycles is the *selective risk*, and the trade-off between the two is summarized by a risk–coverage curve [11]. I evaluated coverages of 100%, 90%, 80%, 70%, 60% and 50%. I also froze two confidence thresholds from clean validation data, τ80 = 0.4617 and τ50 = 0.5771, which kept 80% and 50% of validation cycles, and applied them unchanged to test data.

### 9.2 Results

**Table 5.** Calibration, error detection and abstention for the frozen reference model. Validation calibrated values are leave-one-patient-out out-of-fold estimates; test calibrated values use the frozen validation temperature.

| | Validation | Test |
|---|---|---|
| ECE, raw → temperature-scaled | 0.088 → 0.050 | 0.208 → 0.127 |
| Overconfidence gap, raw → temperature-scaled | +0.074 → +0.001 | +0.208 → +0.127 |
| Error-detection AUROC, 1 − confidence | 0.599 (0.496, 0.709) | 0.542 (0.467, 0.612) |
| Error-detection AUROC, entropy | 0.620 (0.527, 0.715) | 0.554 (0.490, 0.608) |
| Error rate at 100% → 50% coverage | 0.490 → 0.416 | 0.606 → 0.561 |

On validation, temperature scaling removed the model's average overconfidence. The overconfidence gap changed by −0.074, with a 95% interval of −0.091 to −0.061. The ECE improvement was less certain, with an interval that included no change, and negative log-likelihood and Brier score barely moved.

On the test patients, the raw model was clearly more overconfident than on validation, with ECE 0.208 (0.120, 0.302) and an overconfidence gap of +0.208 (0.080, 0.301). The frozen validation temperature transferred partially. It brought ECE down to 0.127 and the overconfidence gap to +0.127 (−0.001, 0.220), and negative log-likelihood fell from 1.392 to 1.296. Figure 4 shows the corresponding reliability diagrams.

![Figure 4](../reports/final_evaluation/clean_test/reliability_diagram.svg)

**Figure 4.** Reliability diagrams for the clean test set: raw probabilities (left) and probabilities rescaled with the frozen validation temperature (right). Predictions are grouped into 15 equal-width confidence bins. Bars show the accuracy within each bin and are labelled with the share of cycles in that bin, horizontal ticks mark the bin's mean confidence, and the dashed diagonal indicates perfect calibration.

Better average calibration did not translate into recognizing individual errors. On test, the AUROC of 1 − confidence for detecting errors was 0.542, and that of entropy was 0.554, with intervals reaching 0.467 and 0.490 at their lower ends. In practice the model's uncertainty separated its correct and incorrect test predictions only slightly better than chance. Calibration describes average agreement between confidence and accuracy; error detection requires that individual low-confidence predictions are the wrong ones, and the second does not follow from the first.

Abstention gave limited protection (Figure 5). Keeping only the most confident half of the test cycles lowered the error rate from 0.606 to 0.561, a modest gain relative to the 0.606 error rate of random abstention. The frozen validation thresholds kept close to their nominal coverage on clean test data (80.4% and 53.1%). However, the error rate among retained cycles was 0.584 and 0.557, and 29.5% of all test cycles were wrong despite clearing τ50, against 20.8% on validation. Abstention also rejected minority classes faster than common ones: on validation, 36% of true "both" cycles survived at 50% coverage, against 53% of normal cycles.

![Figure 5](../reports/final_report/figures/risk_coverage_validation_vs_test.svg)

**Figure 5.** Risk–coverage curves for the frozen reference model on the clean validation (blue) and test (orange) sets. Cycles are ranked by raw confidence, and each curve shows the error rate among the retained cycles as the retained share decreases. Dashed horizontal lines mark the overall error rate, which is the expected risk of abstaining at random. The first 20 retained cycles are omitted from each curve for readability.

Under degradation, confidence was an even weaker guide. With Gaussian noise, mean confidence barely changed (0.602 on clean test, 0.573 at 0 dB) while macro F1 fell by more than half, so overconfidence grew. Even after temperature scaling, the gap at 0 dB was +0.207 (0.092, 0.309). Under 50% dropout, mean confidence actually rose, to 0.635. Under noise, the frozen τ80 threshold kept *more* test cycles, 88–94% depending on noise level, while the error among those kept rose to 0.63–0.73. Between 36% and 42% of all noisy test cycles were confidently wrong, compared with 19–25% on validation. The 1000 Hz and 500 Hz low-pass conditions were the main exception. There, mean confidence fell (to 0.450 and 0.458), and τ80 kept only 36% and 41% of cycles.

## 10. Post-Hoc Exploratory Analysis

**Post-hoc exploratory analysis — not predeclared.** The analysis in this section was designed after the locked test results had been seen. It uses only the saved test predictions, no model was re-run, and it does not change any of the confirmatory results above. It is reported because it may help explain the normal→crackle errors, but it should be treated as hypothesis-generating only.

Breaking the 342 normal→crackle test errors down by recording device showed a strong concentration: **324 of the 342 errors (95%) came from the six test patients recorded with the AKGC417L**. These six patients together hold 68% of the test set's normal cycles, so some concentration would be expected, but all six were affected, with 32% to 97% of each patient's normal cycles predicted as crackle. Accuracy on AKGC417L test cycles was 0.315, compared with 0.662 on Meditron cycles. No single patient explains the overall result: recomputing test macro F1 with each patient left out in turn gives values between 0.293 and 0.344.

This association is **confounded**. In ICBHI 2017, AKGC417L recordings come from different patients than other devices, from a narrower range of diagnoses (COPD only), under different recording conditions and at a single sample rate (44.1 kHz). With six patients, the analysis cannot separate an effect of the device from an effect of the particular patients recorded with it. I therefore do not claim that the device caused these errors. The finding is consistent with, but does not demonstrate, the broader concern that recording characteristics influence the model's predictions. Testing it would require a separate, predeclared experiment, for example holding out an entire device.

Two further exploratory observations put other results in context. First, 73 of the 78 test "both" cycles come from just two patients, so the test result for that class effectively reflects two people. Second, the collapse onto crackle under noise was broad rather than driven by a few patients: 11 of the 14 patients with at least 10 test cycles had at least 80% of their cycles predicted as crackle by 5 dB, as did all four devices by 0 dB.

## 11. Discussion

The clearest lesson from this study is that validation performance alone overstated how well the model would do on new patients. The drop from 0.4271 to 0.3175 macro F1 happened even though patients were split, selection was done on validation only, and the test set was untouched until a frozen protocol had been fixed. I did not compare against less strict designs, such as a random split by cycle or repeated checking of a test set. Such evaluations would, however, let information leak into the estimate. A random split lets patient-specific information cross into the test data, which is known to overstate performance [4]. Repeated test-set checking lets feedback from the test results shape the model. Development-set selection may contribute to the observed gap, alongside patient and recording-domain differences; the study did not experimentally separate these factors. The class-level changes, especially the collapse in normal recall and the near-total failure on "both", are consistent with the model meeting patients and recordings unlike those it was developed on.

Several results point towards the recording domain as an important factor. The model's predictions moved under pure gain changes, which alter loudness but not breath sounds. Its behaviour under band-limiting was sharply non-monotonic. In the post-hoc analysis, the dominant test error concentrated in recordings from one device. None of these observations proves that the model relies on recording characteristics, and the device association is confounded. Taken together, though, they are consistent with a model that has learned some properties of how recordings were made alongside properties of the sounds themselves. Deep networks are known to exploit such cues when they happen to be predictive in the training data, and these "shortcuts" then fail to transfer to new conditions [12]. Datasets in which acquisition conditions and patient groups are entangled make such shortcuts available.

The degradation results were systematic rather than noisy. The same collapse patterns appeared on validation and on the untouched test patients: crackle under noise, normal under dropout, wheeze and then normal under band-limiting. The 2000 Hz control behaved exactly as the feature design predicts. This consistency matters because it suggests that the robustness failures reflect the model and its representation rather than a particular split. It would be risky to assume that average robustness on one split implies robustness in general.

The calibration results show that "calibrated" and "trustworthy" are different properties. Temperature scaling did what it is designed to do: it reduced the model's average overconfidence, and a temperature fitted on validation patients carried over partially to test patients, closing about 40% of the test overconfidence gap. But calibration is a statement about averages. Error detection asks whether individual low-confidence predictions are the wrong ones, and on test the answer was barely better than chance. Selective prediction inherits this weakness. Abstaining on the least confident half of the test set removed 54% of the errors, barely more than the 50% expected from abstaining at random. Under noise, thresholds learned on clean data kept *more* inputs as those inputs became less reliable. Abstention based on in-distribution confidence should therefore not be treated as an automatic safety mechanism, especially under distribution shift.

The "both" class deserves particular mention. It is the rarest class, it requires recognizing two kinds of adventitious sound at once, and in this split its test examples come almost entirely from two patients. A near-zero F1 for "both" is a meaningful failure of the model. It is also a reminder that per-class results on small patient samples can hinge on very few individuals.

Finally, I think the negative results are the most useful part of this work. Aggregate scores such as accuracy can look acceptable while hiding class collapse, as the 50% dropout condition shows. Average calibration can improve while error detection remains poor, and validation scores can look stable across seeds while generalizing poorly to new patients. A stricter evaluation design did not make the model better, but it made these failure modes visible, which is a necessary step before any claim of reliability could be considered.

## 12. Limitations

This study has important limitations.
- **One dataset, one split.** It uses a single public dataset with one fixed patient split and one frozen model. There is no external validation on independently collected recordings.
- **Few patients.** The validation and test sets contain only 19 patients each, and a few patients contribute a large share of the cycles: the four largest hold 60% of validation and 48% of test cycles. Confidence intervals are therefore wide, subgroup estimates are fragile, and the test "both" class rests mainly on two patients.
- **Confounding.** Device, diagnosis, sample rate and recording conditions are confounded in ICBHI 2017. Matching the device mix across splits reduces this but cannot remove it, and no held-out-device experiment was performed.
- **Synthetic degradations.** The degradations are synthetic. They probe the model's sensitivities in a controlled way but are not equivalent to real acoustic corruption such as friction, ambient speech or stethoscope placement.
- **Accuracy-based calibration measures.** Top-label ECE and the overconfidence gap compare confidence with accuracy, so they can look acceptable when predictions collapse onto the majority class.
- **Limited model search.** The search was deliberately small, and no data augmentation, ensembling or domain-adaptation methods were explored.
- **No clinical validation.** No clinical validation of any kind was performed.

## 13. Conclusion

The frozen classifier learned patterns in respiratory sounds that carried over to some degree to unseen patients. However, its performance dropped substantially on those patients relative to validation, and it failed in consistent, predictable ways under several kinds of degraded audio. Temperature scaling corrected part of its average overconfidence, yet its uncertainty barely distinguished correct predictions from wrong ones, and abstention offered only limited protection. Respiratory-sound models should therefore be evaluated on new patients, under changes in recording conditions, and for whether their uncertainty is actually informative, rather than judged on clean validation performance alone.

## 14. Reproducibility and Code Availability

All code, analysis plans, the frozen evaluation protocol and the locked results are available at https://github.com/rish-mishra/acoustic-biomarker-classifier. This paper corresponds to software release **v1.0.1**, a documentation and test-maintenance release that changed no result. The trained checkpoint was published once, with release **v1.0.0**, and was not re-uploaded with v1.0.1.
- **Frozen protocol.** Version 3 of the evaluation protocol (`reports/final_evaluation/protocol.md`) records the frozen model, data pipeline, metrics, temperature, thresholds and degradation conditions, a test-access audit, and content hashes of every frozen artifact.
- **Locked test outputs.** The clean and robustness test outputs are stored with hash locks (`clean_test/lock.json` and `robustness_test/lock.json`), and markers prevent either stage from being re-run.
- **Checkpoint.** The reference checkpoint, `baseline_cnn_v3`, seed 42, epoch 13, is available only as an asset of release v1.0.0 (`baseline_cnn_v3_seed42_best.pt`, 1,188,469 bytes, SHA-256 `2c2c7c03ed00b6562c63ff88be12d43bd8f9ff890074f9dac4d4ce5874e3b567`). It is unchanged in v1.0.1.
- **Determinism.** Training and inference used fixed seeds and deterministic GPU settings, and reruns reproduced the reference validation predictions bit for bit.
- **Statistics.** All intervals use the patient-level bootstrap described in Section 3.3.
- **Tests.** The repository's test suite, which runs on synthetic data and also checks reported numbers against the stored results, passed 338 of 338 tests at release v1.0.1.
- **Raw audio.** The ICBHI 2017 recordings are not redistributed with the code and must be obtained from the original source.

A small number of repository files were changed after the final evaluation, all of them documentation and test maintenance. These changes are recorded in `reports/final_evaluation/post_evaluation_note.md`, and no result, protocol, lock or checkpoint was altered.

DOI: [Zenodo DOI to be added after publication]

## 15. Ethics and Medical Disclaimer

This work is a research and educational project that uses a public, de-identified dataset. The model has not been clinically validated, is not a medical device, and is not intended for screening, diagnosis or any medical decision. The results describe the behaviour of one model under controlled conditions and should not be interpreted as evidence about the health of any person or about the clinical value of automated auscultation.

## 16. Generative AI Use Disclosure

Generative AI tools were used during software development and manuscript preparation, to assist with code, documentation, language editing, structuring and literature searches. The research question, experimental design and evaluation protocol are the author's, as is the interpretation of the locked results, and the author takes final responsibility for the work. All reported numerical results come from the project's stored experimental artifacts, not from AI-generated content. All references were checked against their primary sources before publication.

---

## References

[1] Pasterkamp H, Brand PLP, Everard M, Garcia-Marcos L, Melbye H, Priftis KN. Towards the standardisation of lung sound nomenclature. *European Respiratory Journal*. 2016;47(3):724–732. doi:10.1183/13993003.01132-2015

[2] Rocha BM, Filos D, Mendes L, Vogiatzis I, Perantoni E, Kaimakamis E, et al. A respiratory sound database for the development of automated classification. In: *Precision Medicine Powered by pHealth and Connected Health* (IFMBE Proceedings). Singapore: Springer; 2018. p. 33–37. doi:10.1007/978-981-10-7419-6_6

[3] Rocha BM, Filos D, Mendes L, Serbes G, Ulukaya S, Kahya YP, et al. An open access database for the evaluation of respiratory sound classification algorithms. *Physiological Measurement*. 2019;40(3):035001. doi:10.1088/1361-6579/ab03ea

[4] Saeb S, Lonini L, Jayaraman A, Mohr DC, Kording KP. The need to approximate the use-case in clinical machine learning. *GigaScience*. 2017;6(5):gix019. doi:10.1093/gigascience/gix019

[5] Nguyen T, Pernkopf F. Lung sound classification using co-tuning and stochastic normalization. *IEEE Transactions on Biomedical Engineering*. 2022;69(9):2872–2882. doi:10.1109/TBME.2022.3156293

[6] Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. In: *Proceedings of the 34th International Conference on Machine Learning*. Proceedings of Machine Learning Research, vol. 70; 2017. p. 1321–1330.

[7] Ovadia Y, Fertig E, Ren J, Nado Z, Sculley D, Nowozin S, et al. Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. In: *Advances in Neural Information Processing Systems 32* (NeurIPS 2019); 2019.

[8] Howard A, Sandler M, Chu G, Chen LC, Chen B, Tan M, et al. Searching for MobileNetV3. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*; 2019. p. 1314–1324. doi:10.1109/ICCV.2019.00140

[9] Loshchilov I, Hutter F. Decoupled weight decay regularization. In: *International Conference on Learning Representations (ICLR)*; 2019.

[10] Hendrycks D, Gimpel K. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In: *International Conference on Learning Representations (ICLR)*; 2017.

[11] Geifman Y, El-Yaniv R. Selective classification for deep neural networks. In: *Advances in Neural Information Processing Systems 30* (NIPS 2017); 2017.

[12] Geirhos R, Jacobsen JH, Michaelis C, Zemel R, Brendel W, Bethge M, et al. Shortcut learning in deep neural networks. *Nature Machine Intelligence*. 2020;2(11):665–673. doi:10.1038/s42256-020-00257-z
