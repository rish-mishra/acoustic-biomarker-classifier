# Calibration, uncertainty and selective prediction — frozen baseline CNN (validation only)

Model `baseline_cnn_v3` (frozen, not retrained or re-run); inputs are the saved validation softmax probabilities (1077 cycles, 19 patients). The test split was not read, loaded or evaluated. All analyses below were predeclared in `plan.json` unless they appear under *Post-hoc*.

Calibrated ("LOPO") results are **leave-one-patient-out out-of-fold validation/development-set estimates**, not final held-out generalization evidence. Intervals are patient-level bootstrap 95% intervals (2000 resamples of 19 patients, paired across methods and conditions; temperatures are not refitted inside resamples). These results describe model behaviour and support no clinical or diagnostic claim.

## Part 1–2 — Calibration before and after temperature scaling

![Reliability diagrams](reliability_diagram.svg)

| Metric | Raw softmax | LOPO temperature-scaled (out-of-fold) | Change (95% CI) |
|---|---|---|---|
| NLL | 1.1489 (0.904, 1.353) | 1.1427 (0.961, 1.301) | -0.0062 (-0.058, +0.062) |
| Brier (0–2) | 0.6378 (0.498, 0.738) | 0.6346 (0.528, 0.711) | -0.0032 (-0.029, +0.030) |
| ECE, 15 equal-width bins (primary) | 0.0876 (0.048, 0.179) | 0.0497 (0.036, 0.154) | -0.0379 (-0.069, +0.079) |
| ECE, 15 equal-mass bins | 0.0861 (0.058, 0.180) | 0.0705 (0.054, 0.162) | -0.0156 (-0.057, +0.068) |
| Classwise ECE | 0.0888 (0.065, 0.167) | 0.0854 (0.067, 0.173) | -0.0034 (-0.020, +0.025) |
| MCE | 0.4755 (0.243, 0.736) | 0.5447 (0.260, 0.856) | +0.0692 (-0.330, +0.263) |
| Overconfidence gap (mean MSP − accuracy) | 0.0744 (-0.053, 0.174) | 0.0007 (-0.140, 0.111) | -0.0737 (-0.091, -0.061) |
| Mean MSP, correct predictions | 0.6059 (0.576, 0.637) | 0.5227 (0.500, 0.546) | -0.0831 (-0.102, -0.066) |
| Mean MSP, incorrect predictions | 0.5615 (0.526, 0.609) | 0.4976 (0.465, 0.541) | -0.0639 (-0.077, -0.055) |
| Accuracy / macro F1 | 0.5097 / 0.4271 | 0.5097 / 0.4271 | unchanged (temperature never changes a prediction) |

Temperature scaling changes probabilities only; every predicted class, the accuracy, macro F1 and the confusion matrix are identical, so no discrimination change is claimed.

### Temperatures

LOPO fold temperatures (each fitted on the other 18 patients): min 1.226, median 1.393, max 1.595, SD 0.076; any at the search bound: False.

| Held-out patient | Cycles held out | T |
|---|---|---|
| 154 | 208 | 1.3019 |
| 172 | 176 | 1.2741 |
| 176 | 133 | 1.5950 |
| 156 | 130 | 1.2256 |
| 135 | 112 | 1.4804 |
| 112 | 57 | 1.3932 |
| 110 | 52 | 1.3889 |
| 139 | 46 | 1.3529 |
| 201 | 41 | 1.3986 |
| 167 | 28 | 1.4244 |
| 101 | 23 | 1.4130 |
| 161 | 19 | 1.3578 |
| 127 | 10 | 1.3933 |
| 116 | 8 | 1.3986 |
| 123 | 8 | 1.3936 |
| 197 | 8 | 1.3829 |
| 152 | 7 | 1.3960 |
| 212 | 7 | 1.3846 |
| 190 | 4 | 1.3914 |

**final_validation_temperature = 1.3856** (fitted on all 19 validation patients). It is the frozen calibration parameter for a single later application to the untouched test set. Descriptive only, in-sample and NOT calibration evidence: NLL 1.1241, ECE 0.0395.

## Part 3 — Does uncertainty identify errors?

Errors are the positive class. Error rate 0.490 (= AUPRC of a random score); AUROC 0.5 = no separation.

| Score (higher = more uncertain) | AUROC raw | AUPRC raw | AUROC LOPO-scaled | AUPRC LOPO-scaled |
|---|---|---|---|---|
| 1 − MSP (confidence) | 0.599 (0.496, 0.709) | 0.587 (0.434, 0.723) | 0.579 (0.488, 0.684) | 0.571 (0.426, 0.716) |
| Predictive entropy | 0.620 (0.527, 0.715) | 0.608 (0.459, 0.745) | 0.590 (0.505, 0.683) | 0.584 (0.444, 0.727) |
| 1 − top-2 margin (secondary) | 0.573 (0.474, 0.683) | 0.551 (0.426, 0.673) | 0.553 (0.462, 0.660) | 0.538 (0.419, 0.662) |

Uncertainty by true class (raw):

| Class | n | Error rate | Mean MSP | Mean entropy | MSP correct | MSP incorrect | AUROC (1 − MSP) |
|---|---|---|---|---|---|---|---|
| normal | 508 | 0.498 | 0.592 | 0.962 | 0.580 | 0.604 | 0.453 |
| crackle | 337 | 0.377 | 0.593 | 0.978 | 0.645 | 0.507 | 0.774 |
| wheeze | 122 | 0.434 | 0.559 | 1.017 | 0.582 | 0.528 | 0.636 |
| both | 110 | 0.864 | 0.549 | 1.049 | 0.615 | 0.538 | 0.708 |

Uncertainty by predicted class (raw):

| Class | n | Error rate | Mean MSP | Mean entropy | MSP correct | MSP incorrect | AUROC (1 − MSP) |
|---|---|---|---|---|---|---|---|
| normal | 387 | 0.341 | 0.558 | 1.011 | 0.580 | 0.516 | 0.655 |
| crackle | 400 | 0.475 | 0.636 | 0.899 | 0.645 | 0.627 | 0.541 |
| wheeze | 240 | 0.713 | 0.556 | 1.053 | 0.582 | 0.545 | 0.579 |
| both | 50 | 0.700 | 0.502 | 1.076 | 0.615 | 0.454 | 0.882 |

## Part 4 — Selective prediction / abstention

![Risk-coverage curves](risk_coverage.svg)

**Primary:** selective risk (error rate among retained cycles) and the risk-coverage curve. Random abstention keeps the risk at 0.490 at every coverage.

| Ranking | AURC | E-AURC |
|---|---|---|
| MSP (raw) | 0.4271 (0.231, 0.569) | 0.2801 (0.149, 0.374) |
| Entropy (raw) (secondary) | 0.4176 (0.240, 0.550) | 0.2706 (0.157, 0.353) |
| MSP (LOPO-scaled) (secondary) | 0.4527 (0.257, 0.570) | 0.3057 (0.177, 0.382) |

Primary ranking (raw MSP), predeclared coverages. Macro F1 / balanced accuracy / accuracy are secondary; a class with no retained cycles counts with F1 = 0 and recall = 0 (zero-division convention) and is never dropped.

| Coverage | Retained | Selective risk (95% CI) | Risk − full coverage (95% CI) | Macro F1 | Bal. acc. | Accuracy |
|---|---|---|---|---|---|---|
| 100% | 1077 | 0.4903 (0.355, 0.593) | — | 0.4271 | 0.4568 | 0.5097 |
| 90% | 969 | 0.4685 (0.340, 0.573) | -0.0217 (-0.032, -0.001) | 0.4490 | 0.4740 | 0.5315 |
| 80% | 862 | 0.4606 (0.323, 0.566) | -0.0297 (-0.054, +0.003) | 0.4665 | 0.4925 | 0.5394 |
| 70% | 754 | 0.4549 (0.297, 0.562) | -0.0353 (-0.086, +0.008) | 0.4746 | 0.5070 | 0.5451 |
| 60% | 646 | 0.4381 (0.259, 0.560) | -0.0522 (-0.122, +0.011) | 0.4860 | 0.5305 | 0.5619 |
| 50% | 539 | 0.4156 (0.233, 0.544) | -0.0747 (-0.158, +0.000) | 0.5021 | 0.5522 | 0.5844 |

Retained / rejected counts by class (raw MSP ranking; true-class and predicted-class counts as normal / crackle / wheeze / both):

| Coverage | Retained true | Rejected true | Retained predicted | Rejected predicted | Share of each true class retained |
|---|---|---|---|---|---|
| 100% | 508 / 337 / 122 / 110 | 0 / 0 / 0 / 0 | 387 / 400 / 240 / 50 | 0 / 0 / 0 / 0 | 100% / 100% / 100% / 100% |
| 90% | 462 / 301 / 110 / 96 | 46 / 36 / 12 / 14 | 358 / 369 / 207 / 35 | 29 / 31 / 33 / 15 | 91% / 89% / 90% / 87% |
| 80% | 420 / 257 / 101 / 84 | 88 / 80 / 21 / 26 | 308 / 343 / 183 / 28 | 79 / 57 / 57 / 22 | 83% / 76% / 83% / 76% |
| 70% | 369 / 236 / 76 / 73 | 139 / 101 / 46 / 37 | 255 / 318 / 156 / 25 | 132 / 82 / 84 / 25 | 73% / 70% / 62% / 66% |
| 60% | 316 / 209 / 60 / 61 | 192 / 128 / 62 / 49 | 198 / 296 / 134 / 18 | 189 / 104 / 106 / 32 | 62% / 62% / 49% / 55% |
| 50% | 270 / 180 / 49 / 40 | 238 / 157 / 73 / 70 | 160 / 258 / 108 / 13 | 227 / 142 / 132 / 37 | 53% / 53% / 40% / 36% |

Secondary rankings, selective risk at each coverage:

| Ranking | 100% | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|---|
| MSP (raw) | 0.4903 | 0.4685 | 0.4606 | 0.4549 | 0.4381 | 0.4156 |
| Entropy (raw) | 0.4903 | 0.4634 | 0.4501 | 0.4403 | 0.4164 | 0.3970 |
| MSP (LOPO-scaled) | 0.4903 | 0.4727 | 0.4629 | 0.4589 | 0.4427 | 0.4193 |

## Part 5 — Uncertainty under synthetic degradation

![Confidence vs accuracy under degradation](degradation_confidence.svg)

Raw probabilities, and LOPO-scaled probabilities using each patient's CLEAN-fitted temperature (no temperature is fitted on any degraded condition). Overconfidence gap = mean MSP − accuracy.

| Condition | Accuracy | Mean MSP raw | Gap raw (95% CI) | ECE raw | NLL raw | Gap LOPO (95% CI) | ECE LOPO | NLL LOPO | AUROC 1 − MSP | AUROC entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| Clean | 0.510 | 0.584 | +0.074 (-0.053, 0.174) | 0.088 | 1.149 | +0.001 (-0.140, 0.111) | 0.050 | 1.143 | 0.599 | 0.620 |
| Noise 20 dB | 0.452 | 0.582 | +0.129 (-0.019, 0.261) | 0.129 | 1.176 | +0.053 (-0.109, 0.189) | 0.053 | 1.166 | 0.612 | 0.634 |
| Noise 10 dB | 0.428 | 0.561 | +0.133 (-0.030, 0.286) | 0.133 | 1.231 | +0.058 (-0.115, 0.214) | 0.070 | 1.218 | 0.560 | 0.568 |
| Noise 5 dB | 0.400 | 0.552 | +0.152 (-0.015, 0.314) | 0.152 | 1.259 | +0.079 (-0.099, 0.243) | 0.079 | 1.243 | 0.573 | 0.573 |
| Noise 0 dB | 0.386 | 0.546 | +0.160 (-0.014, 0.332) | 0.160 | 1.278 | +0.088 (-0.093, 0.263) | 0.088 | 1.258 | 0.560 | 0.569 |
| Gain 0.5x | 0.501 | 0.578 | +0.077 (-0.049, 0.179) | 0.077 | 1.138 | +0.005 (-0.135, 0.118) | 0.042 | 1.134 | 0.615 | 0.613 |
| Gain 0.25x | 0.484 | 0.582 | +0.098 (-0.043, 0.206) | 0.104 | 1.159 | +0.025 (-0.127, 0.141) | 0.047 | 1.146 | 0.596 | 0.593 |
| Gain 1.5x | 0.483 | 0.583 | +0.100 (-0.022, 0.192) | 0.100 | 1.175 | +0.027 (-0.113, 0.130) | 0.042 | 1.161 | 0.615 | 0.612 |
| Gain 2x | 0.473 | 0.588 | +0.115 (0.001, 0.210) | 0.119 | 1.213 | +0.040 (-0.087, 0.142) | 0.048 | 1.186 | 0.587 | 0.579 |
| Low-pass 2000 Hz | 0.511 | 0.584 | +0.074 (-0.053, 0.172) | 0.087 | 1.149 | -0.000 (-0.141, 0.110) | 0.049 | 1.142 | 0.597 | 0.619 |
| Low-pass 1000 Hz | 0.292 | 0.439 | +0.148 (-0.042, 0.279) | 0.148 | 1.504 | +0.099 (-0.094, 0.229) | 0.099 | 1.444 | 0.514 | 0.489 |
| Low-pass 500 Hz | 0.433 | 0.479 | +0.046 (-0.110, 0.202) | 0.063 | 1.283 | -0.015 (-0.169, 0.138) | 0.023 | 1.277 | 0.566 | 0.580 |
| Dropout 10% | 0.491 | 0.571 | +0.080 (-0.028, 0.184) | 0.080 | 1.157 | +0.013 (-0.104, 0.126) | 0.079 | 1.145 | 0.556 | 0.561 |
| Dropout 25% | 0.457 | 0.619 | +0.162 (0.009, 0.309) | 0.168 | 1.324 | +0.088 (-0.070, 0.239) | 0.118 | 1.249 | 0.477 | 0.478 |
| Dropout 50% | 0.472 | 0.664 | +0.192 (-0.015, 0.370) | 0.194 | 1.377 | +0.108 (-0.106, 0.287) | 0.119 | 1.277 | 0.451 | 0.451 |

Fixed clean-data abstention thresholds applied unchanged to every condition (raw MSP): τ80 = 0.4617, τ50 = 0.5771. *Confidently wrong* = wrong and MSP ≥ τ50.

| Condition | Coverage at τ80 | Risk at τ80 | Coverage at τ50 | Risk at τ50 | Confidently wrong (share of all cycles) | Share of errors that are confident |
|---|---|---|---|---|---|---|
| Clean | 80.0% | 0.461 | 50.0% | 0.416 | 20.8% (0.117, 0.273) | 42.4% |
| Noise 20 dB | 85.1% | 0.527 | 53.2% | 0.471 | 25.1% (0.120, 0.373) | 45.8% |
| Noise 10 dB | 84.1% | 0.556 | 46.1% | 0.545 | 25.2% (0.092, 0.391) | 44.0% |
| Noise 5 dB | 86.4% | 0.587 | 41.2% | 0.543 | 22.4% (0.067, 0.382) | 37.3% |
| Noise 0 dB | 88.8% | 0.609 | 34.3% | 0.542 | 18.6% (0.045, 0.367) | 30.3% |
| Gain 0.5x | 82.2% | 0.468 | 46.6% | 0.420 | 19.6% (0.140, 0.251) | 39.3% |
| Gain 0.25x | 84.2% | 0.499 | 49.5% | 0.454 | 22.5% (0.161, 0.310) | 43.5% |
| Gain 1.5x | 78.6% | 0.478 | 47.8% | 0.423 | 20.2% (0.104, 0.287) | 39.1% |
| Gain 2x | 78.7% | 0.496 | 48.3% | 0.456 | 22.0% (0.128, 0.315) | 41.7% |
| Low-pass 2000 Hz | 80.1% | 0.460 | 50.0% | 0.414 | 20.7% (0.117, 0.273) | 42.3% |
| Low-pass 1000 Hz | 30.6% | 0.739 | 7.1% | 0.818 | 5.8% (0.010, 0.124) | 8.3% |
| Low-pass 500 Hz | 57.3% | 0.541 | 7.2% | 0.269 | 1.9% (0.004, 0.058) | 3.4% |
| Dropout 10% | 86.4% | 0.485 | 43.6% | 0.457 | 20.0% (0.123, 0.282) | 39.2% |
| Dropout 25% | 94.1% | 0.546 | 61.0% | 0.554 | 33.8% (0.205, 0.483) | 62.2% |
| Dropout 50% | 99.9% | 0.529 | 86.8% | 0.538 | 46.7% (0.277, 0.636) | 88.4% |

Predeclared family rule (gap at every level inside the clean gap's 95% interval → *tracks difficulty*; above → *overconfident*; below → *underconfident*; both → *mixed*):

| Family | Raw rule | LOPO rule | Mean MSP moves with accuracy (per level, raw) |
|---|---|---|---|
| Gaussian noise | confidence tracks difficulty | confidence tracks difficulty | Noise 20 dB: yes, Noise 10 dB: yes, Noise 5 dB: yes, Noise 0 dB: yes |
| Amplitude attenuation | confidence tracks difficulty | confidence tracks difficulty | Gain 0.5x: yes, Gain 0.25x: yes |
| Amplitude amplification | confidence tracks difficulty | confidence tracks difficulty | Gain 1.5x: yes, Gain 2x: no |
| Low-pass filtering | confidence tracks difficulty | confidence tracks difficulty | Low-pass 2000 Hz: yes, Low-pass 1000 Hz: yes, Low-pass 500 Hz: yes |
| Temporal dropout | becomes overconfident | confidence tracks difficulty | Dropout 10%: yes, Dropout 25%: no, Dropout 50%: no |

## Post-hoc exploratory observations (not predeclared)

Added after inspecting the predeclared results. Exploratory; not used to change any analysis.

**1. Accuracy-based calibration hides class collapse.** The overconfidence gap and top-label ECE compare confidence with accuracy, and accuracy stays high when predictions collapse onto the majority class (normal is 47% of validation cycles). Macro F1 exposes what the gap does not:

| Condition | Accuracy | Macro F1 | Mean MSP raw | Gap raw | Most-predicted class (share) |
|---|---|---|---|---|---|
| Clean | 0.510 | 0.427 | 0.584 | +0.074 | crackle (37%) |
| Noise 20 dB | 0.452 | 0.358 | 0.582 | +0.129 | crackle (64%) |
| Noise 10 dB | 0.428 | 0.309 | 0.561 | +0.133 | crackle (74%) |
| Noise 5 dB | 0.400 | 0.240 | 0.552 | +0.152 | crackle (80%) |
| Noise 0 dB | 0.386 | 0.208 | 0.546 | +0.160 | crackle (85%) |
| Gain 0.5x | 0.501 | 0.405 | 0.578 | +0.077 | normal (50%) |
| Gain 0.25x | 0.484 | 0.341 | 0.582 | +0.098 | normal (64%) |
| Gain 1.5x | 0.483 | 0.411 | 0.583 | +0.100 | crackle (46%) |
| Gain 2x | 0.473 | 0.410 | 0.588 | +0.115 | crackle (52%) |
| Low-pass 2000 Hz | 0.511 | 0.428 | 0.584 | +0.074 | crackle (37%) |
| Low-pass 1000 Hz | 0.292 | 0.173 | 0.439 | +0.148 | wheeze (56%) |
| Low-pass 500 Hz | 0.433 | 0.240 | 0.479 | +0.046 | normal (81%) |
| Dropout 10% | 0.491 | 0.384 | 0.571 | +0.080 | normal (47%) |
| Dropout 25% | 0.457 | 0.256 | 0.619 | +0.162 | normal (78%) |
| Dropout 50% | 0.472 | 0.166 | 0.664 | +0.192 | normal (99%) |

**2. The predeclared family rule is insensitive.** The clean gap's patient-bootstrap interval is (-0.053, +0.174), 0.227 wide, because overconfidence varies strongly between patients. Gaussian noise is therefore labelled *tracks difficulty* although its point-estimate gap grows monotonically from +0.074 to +0.129, +0.133, +0.152, +0.160 while accuracy falls and mean confidence barely moves.

**3. Under shift, the clean-data threshold can retain more cycles while they become more often wrong.** At τ80 (80% coverage on clean data): Noise 20 dB keeps 85% with risk 0.527; Noise 10 dB keeps 84% with risk 0.556; Noise 5 dB keeps 86% with risk 0.587; Noise 0 dB keeps 89% with risk 0.609; Gain 0.5x keeps 82% with risk 0.468; Gain 0.25x keeps 84% with risk 0.499; Dropout 10% keeps 86% with risk 0.485; Dropout 25% keeps 94% with risk 0.546; Dropout 50% keeps 100% with risk 0.529 (clean: 80% with risk 0.461). Abstention calibrated on clean data does not protect against these degradations.

**4. Uncertainty can rank errors backwards.** Error-detection AUROC (1 − MSP) is below 0.5 for: Dropout 25% (0.477), Dropout 50% (0.451). On clean data, cycles whose true class is normal are misclassified with higher confidence than they are correctly classified (mean MSP 0.604 vs 0.580; AUROC 0.453).

**5. Abstention removes minority classes faster.** At 50% coverage the share of each true class retained is normal 53%, crackle 53%, wheeze 40%, both 36%. Retained-subset macro F1 is therefore measured on a different class mix than full coverage and is not a like-for-like improvement.
