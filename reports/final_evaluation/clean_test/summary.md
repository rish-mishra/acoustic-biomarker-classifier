# Final clean test evaluation (one-time) — frozen baseline_cnn_v3

Test split: 1080 cycles, 19 patients. Protocol SHA-256 `05f7fd0048f49d9a0d018319142fc6b07397b46dff97995e1bcda07eeb524f65`, checkpoint `2c2c7c03ed00b656…`, commit `9d36c12638a4f6df5439a9777a266b9b461a3fe0`. All analyses predeclared in the frozen protocol; intervals are patient-level bootstrap 95% (2000 resamples, seed 0). No clinical or diagnostic claim.

## Primary endpoint

**Macro F1 = 0.3175** (0.218, 0.413)

## Secondary

Balanced accuracy 0.4030 (0.325, 0.522); accuracy 0.3935 (0.293, 0.527).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| normal | 0.732 | 0.278 | 0.403 | 647 |
| crackle | 0.333 | 0.725 | 0.456 | 255 |
| wheeze | 0.274 | 0.570 | 0.370 | 100 |
| both | 0.043 | 0.038 | 0.041 | 78 |

Confusion matrix (rows = true, columns = predicted; normal / crackle / wheeze / both):

      180    342     94     31
       42    185     11     17
       13     11     57     19
       11     18     46      3

## Quality sensitivity groups

| Group | Cycles / patients | Macro F1 (4 classes) | Macro F1 (classes present) | Bal. acc. | Accuracy |
|---|---|---|---|---|---|
| all | 1080 / 19 | 0.3175 (0.218, 0.413) | 0.3175 | 0.4030 | 0.3935 |
| moderately_clean | 1062 / 19 | 0.3176 (0.218, 0.413) | 0.3176 | 0.4034 | 0.3955 |
| clean | 949 / 19 | 0.3317 (0.221, 0.433) | 0.3317 | 0.4191 | 0.4204 |

## Calibration (frozen validation temperature, never fitted on test)

T = 1.3856213645470599. Temperature scaling cannot change any predicted class.

![Reliability diagrams](reliability_diagram.svg)

| Metric | Raw | Frozen T |
|---|---|---|
| NLL | 1.3919 (1.152, 1.613) | 1.2963 (1.117, 1.459) |
| Brier | 0.7656 (0.642, 0.858) | 0.7211 (0.626, 0.793) |
| ECE (15 equal-width) | 0.2083 (0.120, 0.302) | 0.1271 (0.089, 0.221) |
| Overconfidence gap | 0.2083 (0.080, 0.301) | 0.1271 (-0.001, 0.220) |
| ECE equal-mass (secondary) | 0.2083 (0.130, 0.301) | 0.1340 (0.101, 0.223) |
| Classwise ECE (secondary) | 0.1826 (0.130, 0.250) | 0.1700 (0.114, 0.246) |
| MCE (secondary) | 0.6841 (0.517, 0.963) | 0.7066 (0.581, 0.897) |

## Error detection

Test error rate 0.606 (AUPRC chance level).

| Score | AUROC raw | AUPRC raw | AUROC frozen T | AUPRC frozen T |
|---|---|---|---|---|
| 1 − MSP (confidence) | 0.542 (0.467, 0.612) | 0.662 (0.565, 0.739) | 0.548 | 0.663 |
| Predictive entropy | 0.554 (0.490, 0.608) | 0.664 (0.567, 0.740) | 0.562 | 0.668 |
| 1 − top-2 margin | 0.528 (0.447, 0.606) | 0.655 (0.556, 0.735) | 0.529 | 0.655 |

## Selective prediction (primary ranking: raw MSP)

AURC 0.6024 (0.463, 0.715); E-AURC 0.3626 (0.307, 0.410).

| Coverage | Retained | Selective risk | Macro F1 | Bal. acc. | Accuracy | Retained true (n/c/w/b) | Rejected true | Retained pred | Rejected pred |
|---|---|---|---|---|---|---|---|---|---|
| 100% | 1080 | 0.6065 (0.473, 0.707) | 0.3175 | 0.4030 | 0.3935 | 647/255/100/78 | 0/0/0/0 | 246/556/208/70 | 0/0/0/0 |
| 90% | 972 | 0.5926 (0.445, 0.700) | 0.3331 | 0.4263 | 0.4074 | 581/230/89/72 | 66/25/11/6 | 222/513/184/53 | 24/43/24/17 |
| 80% | 864 | 0.5833 (0.424, 0.696) | 0.3362 | 0.4410 | 0.4167 | 523/201/75/65 | 124/54/25/13 | 198/458/168/40 | 48/98/40/30 |
| 70% | 756 | 0.5780 (0.405, 0.699) | 0.3373 | 0.4559 | 0.4220 | 460/170/65/61 | 187/85/35/17 | 170/403/154/29 | 76/153/54/41 |
| 60% | 648 | 0.5617 (0.392, 0.697) | 0.3431 | 0.4740 | 0.4383 | 390/144/55/59 | 257/111/45/19 | 144/350/135/19 | 102/206/73/51 |
| 50% | 540 | 0.5611 (0.389, 0.701) | 0.3397 | 0.4887 | 0.4389 | 327/118/41/54 | 320/137/59/24 | 118/296/115/11 | 128/260/93/59 |

Frozen clean-validation thresholds applied unchanged:

| Threshold | τ | Test coverage | Selective risk | Confidently wrong (share of all) |
|---|---|---|---|---|
| tau_80 | 0.4616909623146057 | 80.4% | 0.5841 | 46.9% |
| tau_50 | 0.5770796537399292 | 53.1% | 0.5567 | 29.5% |
