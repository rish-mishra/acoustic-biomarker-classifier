# POST-HOC EXPLORATORY — NOT PREDECLARED

## Exploratory analysis of the locked final test outputs

**POST-HOC EXPLORATORY — NOT PREDECLARED.** Generated after the confirmatory final evaluation was complete and locked, from the saved clean-test and robustness prediction files only (no inference, retraining, fitting, tuning or selection). These are descriptive associations on 19 test patients; they do not change any confirmatory result and support no causal, clinical or diagnostic claim. Both locks verified before and after this analysis.

Macro F1 / balanced accuracy are shown only where meaningful (≥ 20 cycles and ≥ 2 true classes present) and use the classes present in that group.

## 1. Per-patient clean-test performance

| Patient | Device | Cycles | True n/c/w/b | Pred n/c/w/b | Accuracy | Macro F1 (present) | Normal→crackle (of normals) | Mean MSP |
|---|---|---|---|---|---|---|---|---|
| 170 | AKGC417L | 150 | 120/24/2/4 | 11/133/6/0 | 0.193 | 0.097 | 106 / 120 (88%) | 0.569 |
| 207 | AKGC417L | 139 | 116/22/1/0 | 23/107/8/1 | 0.302 | 0.209 | 87 / 116 (75%) | 0.625 |
| 186 | AKGC417L | 126 | 50/74/2/0 | 13/92/11/10 | 0.476 | 0.287 | 38 / 50 (76%) | 0.589 |
| 218 | Meditron | 107 | 19/33/12/43 | 27/49/27/4 | 0.402 | 0.352 | 7 / 19 (37%) | 0.618 |
| 205 | AKGC417L | 102 | 45/57/0/0 | 15/72/3/12 | 0.343 | 0.289 | 39 / 45 (87%) | 0.598 |
| 198 | AKGC417L | 98 | 29/28/41/0 | 2/62/19/15 | 0.469 | 0.407 | 28 / 29 (97%) | 0.511 |
| 162 | AKGC417L | 89 | 81/6/1/1 | 14/26/21/28 | 0.112 | 0.065 | 26 / 81 (32%) | 0.517 |
| 221 | LittC2SE | 63 | 4/0/29/30 | 4/0/59/0 | 0.492 | 0.462 | 0 / 4 (0%) | 0.787 |
| 188 | Meditron | 62 | 62/0/0/0 | 61/1/0/0 | 0.984 | — | 1 / 62 (2%) | 0.641 |
| 113 | Litt3200 | 50 | 50/0/0/0 | 1/0/49/0 | 0.020 | — | 0 / 50 (0%) | 0.658 |
| 216 | Meditron | 22 | 10/4/8/0 | 9/11/2/0 | 0.364 | 0.372 | 7 / 10 (70%) | 0.546 |
| 220 | LittC2SE | 14 | 12/2/0/0 | 14/0/0/0 | 0.857 | — | 0 / 12 (0%) | 0.683 |
| 126 | Meditron | 11 | 11/0/0/0 | 11/0/0/0 | 1.000 | — | 0 / 11 (0%) | 0.673 |
| 148 | Meditron | 11 | 11/0/0/0 | 8/2/1/0 | 0.727 | — | 2 / 11 (18%) | 0.544 |
| 129 | Meditron | 9 | 4/5/0/0 | 9/0/0/0 | 0.444 | — | 0 / 4 (0%) | 0.698 |
| 108 | Meditron | 8 | 8/0/0/0 | 7/1/0/0 | 0.875 | — | 1 / 8 (12%) | 0.598 |
| 214 | Meditron | 7 | 7/0/0/0 | 7/0/0/0 | 1.000 | — | 0 / 7 (0%) | 0.640 |
| 103 | LittC2SE | 6 | 2/0/4/0 | 4/0/2/0 | 0.667 | — | 0 / 2 (0%) | 0.664 |
| 196 | Meditron | 6 | 6/0/0/0 | 6/0/0/0 | 1.000 | — | 0 / 6 (0%) | 0.619 |

## 2. Per-device clean-test performance

| Device | Patients | Cycles | True n/c/w/b | Pred n/c/w/b | Accuracy | Bal. acc. (present) | Macro F1 (present) | Normal→crackle rate | Error rate | Mean MSP (correct / incorrect) |
|---|---|---|---|---|---|---|---|---|---|---|
| AKGC417L | 6 | 704 | 441/211/47/5 | 78/492/68/66 | 0.315 | 0.357 | 0.242 | 0.735 | 0.685 | 0.573 (0.576 / 0.572) |
| Litt3200 | 2 | 71 | 56/4/3/8 | 11/1/59/0 | 0.127 | 0.335 | 0.162 | 0.000 | 0.873 | 0.615 (0.512 / 0.629) |
| LittC2SE | 3 | 83 | 18/2/33/30 | 22/0/61/0 | 0.566 | 0.463 | 0.372 | 0.000 | 0.434 | 0.761 (0.733 / 0.797) |
| Meditron | 9 | 222 | 132/38/17/35 | 135/63/20/4 | 0.662 | 0.502 | 0.456 | 0.136 | 0.338 | 0.630 (0.640 / 0.609) |

## 3. Contribution to the normal→crackle shift

342 normal cycles were predicted as crackle. Patients: top 1 accounts for 31%, top 3 for 68%; 2 of 19 patients account for ≥ 50% and 5 for ≥ 80%. Devices: top device accounts for 95%.

| Patient | Normal→crackle | Share of all | Cumulative | Share of all normal cycles |
|---|---|---|---|---|
| 170 | 106 | 31.0% | 31.0% | 18.5% |
| 207 | 87 | 25.4% | 56.4% | 17.9% |
| 205 | 39 | 11.4% | 67.8% | 7.0% |
| 186 | 38 | 11.1% | 78.9% | 7.7% |
| 198 | 28 | 8.2% | 87.1% | 4.5% |
| 162 | 26 | 7.6% | 94.7% | 12.5% |
| 216 | 7 | 2.0% | 96.8% | 1.5% |
| 218 | 7 | 2.0% | 98.8% | 2.9% |
| 148 | 2 | 0.6% | 99.4% | 1.7% |
| 108 | 1 | 0.3% | 99.7% | 1.2% |
| 188 | 1 | 0.3% | 100.0% | 9.6% |

| Device | Normal→crackle | Share of all | Share of all normal cycles |
|---|---|---|---|
| AKGC417L | 324 | 94.7% | 68.2% |
| Meditron | 18 | 5.3% | 20.4% |
| Litt3200 | 0 | 0.0% | 8.7% |
| LittC2SE | 0 | 0.0% | 2.8% |

Leave-one-patient-out (overall clean-test values recomputed without each patient; descriptive influence only):

| Excluded patient | Cycles excluded | Macro F1 (4 classes) | Accuracy | Normal→crackle rate |
|---|---|---|---|---|
| 170 | 150 | 0.3437 | 0.426 | 0.448 |
| 113 | 50 | 0.3405 | 0.412 | 0.573 |
| 162 | 89 | 0.3339 | 0.419 | 0.558 |
| 207 | 139 | 0.3313 | 0.407 | 0.480 |
| 205 | 102 | 0.3228 | 0.399 | 0.503 |
| 216 | 22 | 0.3180 | 0.394 | 0.526 |
(all 19 rows in summary.json; full-test macro F1 is 0.3175)

## 4. Noise: is the crackle collapse concentrated?

Patients with ≥ 10 cycles: share of their cycles predicted as crackle.

| Condition | Patients | Min | Median | Max | Patients ≥ 80% crackle |
|---|---|---|---|---|---|
| noise_snr20db | 14 | 0% | 61% | 99% | 5 |
| noise_snr10db | 14 | 0% | 86% | 100% | 8 |
| noise_snr5db | 14 | 0% | 93% | 100% | 11 |
| noise_snr0db | 14 | 0% | 99% | 100% | 11 |

| Device | Patients | Cycles | True crackle share | Clean | 20 dB | 10 dB | 5 dB | 0 dB (predicted crackle share) |
|---|---|---|---|---|---|---|---|---|
| AKGC417L | 6 | 704 | 30% | 70% | 91% | 97% | 99% | 99% |
| Litt3200 | 2 | 71 | 6% | 1% | 18% | 55% | 86% | 93% |
| LittC2SE | 3 | 83 | 2% | 0% | 36% | 77% | 82% | 83% |
| Meditron | 9 | 222 | 17% | 28% | 45% | 53% | 68% | 83% |

## 5. Gain sensitivity by device

Change in the share of predictions that are normal (and share of predictions changed) vs clean test:

| Device | Patients | Cycles | Clean normal share | 0.5x | 0.25x | 1.5x | 2x |
|---|---|---|---|---|---|---|---|
| AKGC417L | 6 | 704 | 11% | +10% (22%) | +26% (44%) | -1% (8%) | -2% (14%) |
| Litt3200 | 2 | 71 | 15% | +10% (10%) | +21% (21%) | -3% (3%) | -3% (3%) |
| LittC2SE | 3 | 83 | 27% | +4% (4%) | +6% (6%) | -1% (2%) | -4% (7%) |
| Meditron | 9 | 222 | 61% | +16% (18%) | +23% (26%) | -5% (8%) | -11% (14%) |

Per-patient gain tables: gain_by_patient.csv. Devices are confounded with patients and with sample rate here (few patients per device), so device differences cannot be separated from patient differences.
