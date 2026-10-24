# Robustness of the baseline CNN to synthetic audio degradations (validation only)

Frozen model `baseline_cnn_v3` (no retraining), validation split only (1077 cycles, 19 patients). The test split was not loaded or evaluated. Each degradation is applied to the 16 kHz cycle waveform before the frozen Log-Mel features (50-2000 Hz) and frozen normalization. Definitions and the descriptive rules below were fixed in `plan.json` before any result was computed.

These are synthetic degradations on one validation split; they describe model behaviour, not clinical or diagnostic performance.

Reference scale: the seed study's pooled within-architecture seed SD of macro F1 is **0.0157**; changes smaller than this are about as large as retraining with a different seed. Intervals are patient-level bootstrap 95% intervals (2000 resamples of 19 patients) for the paired change vs clean; with 19 patients they are wide.

![Macro and per-class F1 by condition](macro_f1_by_condition.svg)

## Results by condition

| Condition | n | Macro F1 | Δ abs (95% CI) | Δ rel | Bal. acc. | Acc. | F1 normal | F1 crackle | F1 wheeze | F1 both | Preds changed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Clean | 1077 | 0.4271 | — | — | 0.4568 | 0.5097 | 0.570 | 0.570 | 0.381 | 0.187 | — |
| Noise 20 dB SNR | 1077 | 0.3583 | -0.0688 (-0.138, +0.016) | -16.1% | 0.3887 | 0.4522 | 0.405 | 0.555 | 0.361 | 0.112 | 34.0% |
| Noise 10 dB SNR | 1077 | 0.3087 | -0.1184 (-0.213, -0.009) | -27.7% | 0.3488 | 0.4280 | 0.359 | 0.536 | 0.276 | 0.064 | 43.8% |
| Noise 5 dB SNR | 1077 | 0.2396 | -0.1875 (-0.275, -0.077) | -43.9% | 0.3048 | 0.4002 | 0.309 | 0.525 | 0.125 | 0.000 | 49.9% |
| Noise 0 dB SNR | 1077 | 0.2078 | -0.2193 (-0.312, -0.097) | -51.3% | 0.2908 | 0.3863 | 0.262 | 0.522 | 0.047 | 0.000 | 53.9% |
| Gain 0.5x (-6.0 dB) | 1077 | 0.4048 | -0.0223 (-0.058, +0.012) | -5.2% | 0.4232 | 0.5014 | 0.594 | 0.490 | 0.371 | 0.164 | 18.9% |
| Gain 0.25x (-12.0 dB) | 1077 | 0.3413 | -0.0858 (-0.141, -0.022) | -20.1% | 0.3777 | 0.4838 | 0.619 | 0.335 | 0.377 | 0.035 | 36.3% |
| Gain 1.5x (+3.5 dB) | 1077 | 0.4105 | -0.0166 (-0.049, +0.012) | -3.9% | 0.4481 | 0.4828 | 0.504 | 0.569 | 0.380 | 0.189 | 13.5% |
| Gain 2x (+6.0 dB) | 1077 | 0.4099 | -0.0172 (-0.061, +0.023) | -4.0% | 0.4567 | 0.4726 | 0.450 | 0.568 | 0.420 | 0.201 | 21.8% |
| Low-pass 2000 Hz | 1077 | 0.4276 | +0.0005 (-0.000, +0.002) | +0.1% | 0.4573 | 0.5107 | 0.572 | 0.570 | 0.382 | 0.186 | 0.3% |
| Low-pass 1000 Hz | 1077 | 0.1735 | -0.2536 (-0.316, -0.171) | -59.4% | 0.2666 | 0.2916 | 0.496 | 0.000 | 0.198 | 0.000 | 63.5% |
| Low-pass 500 Hz | 1077 | 0.2400 | -0.1871 (-0.283, -0.082) | -43.8% | 0.3228 | 0.4327 | 0.588 | 0.000 | 0.372 | 0.000 | 61.3% |
| Dropout 10% | 1077 | 0.3838 | -0.0434 (-0.089, +0.023) | -10.2% | 0.3915 | 0.4912 | 0.544 | 0.522 | 0.372 | 0.097 | 32.1% |
| Dropout 25% | 1077 | 0.2563 | -0.1708 (-0.274, -0.056) | -40.0% | 0.2774 | 0.4568 | 0.592 | 0.303 | 0.130 | 0.000 | 55.2% |
| Dropout 50% | 1077 | 0.1660 | -0.2611 (-0.358, -0.146) | -61.1% | 0.2510 | 0.4717 | 0.641 | 0.023 | 0.000 | 0.000 | 63.5% |

## Per-class F1 change vs clean

| Condition | normal | crackle | wheeze | both | Predicted class distribution (normal / crackle / wheeze / both) |
|---|---|---|---|---|---|
| Clean | +0.000 | +0.000 | +0.000 | +0.000 | 387 / 400 / 240 / 50 |
| Noise 20 dB SNR | -0.165 | -0.015 | -0.020 | -0.076 | 253 / 686 / 105 / 33 |
| Noise 10 dB SNR | -0.211 | -0.034 | -0.105 | -0.123 | 205 / 798 / 59 / 15 |
| Noise 5 dB SNR | -0.261 | -0.045 | -0.256 | -0.187 | 185 / 864 / 22 / 6 |
| Noise 0 dB SNR | -0.308 | -0.047 | -0.334 | -0.187 | 156 / 911 / 6 / 4 |
| Gain 0.5x (-6.0 dB) | +0.024 | -0.080 | -0.010 | -0.023 | 542 / 288 / 223 / 24 |
| Gain 0.25x (-12.0 dB) | +0.049 | -0.235 | -0.004 | -0.153 | 694 / 171 / 207 / 5 |
| Gain 1.5x (+3.5 dB) | -0.066 | -0.001 | -0.001 | +0.001 | 286 / 496 / 225 / 70 |
| Gain 2x (+6.0 dB) | -0.119 | -0.002 | +0.039 | +0.014 | 229 / 558 / 211 / 79 |
| Low-pass 2000 Hz | +0.002 | +0.000 | +0.001 | -0.001 | 387 / 400 / 239 / 51 |
| Low-pass 1000 Hz | -0.074 | -0.570 | -0.183 | -0.187 | 468 / 4 / 605 / 0 |
| Low-pass 500 Hz | +0.019 | -0.570 | -0.010 | -0.187 | 872 / 4 / 201 / 0 |
| Dropout 10% | -0.025 | -0.048 | -0.009 | -0.091 | 506 / 437 / 120 / 14 |
| Dropout 25% | +0.022 | -0.267 | -0.251 | -0.187 | 837 / 224 / 16 / 0 |
| Dropout 50% | +0.071 | -0.547 | -0.381 | -0.187 | 1064 / 13 / 0 / 0 |

## Degradation families (predeclared descriptive rules)

| Family | Levels (mild → severe) | Macro-F1 change per level | Step drops | Shape | First class to drop ≥ 0.05 F1 |
|---|---|---|---|---|---|
| Gaussian noise | Noise 20 dB SNR, Noise 10 dB SNR, Noise 5 dB SNR, Noise 0 dB SNR | -0.0688, -0.1184, -0.1875, -0.2193 | +0.0688, +0.0495, +0.0692, +0.0318 | gradual | normal (0.165) at Noise 20 dB SNR |
| Amplitude attenuation | Gain 0.5x (-6.0 dB), Gain 0.25x (-12.0 dB) | -0.0223, -0.0858 | +0.0223, +0.0634 | collapse at Gain 0.25x (-12.0 dB) | crackle (0.080) at Gain 0.5x (-6.0 dB) |
| Amplitude amplification | Gain 1.5x (+3.5 dB), Gain 2x (+6.0 dB) | -0.0166, -0.0172 | +0.0166, +0.0006 | gradual | normal (0.066) at Gain 1.5x (+3.5 dB) |
| Low-pass filtering | Low-pass 2000 Hz, Low-pass 1000 Hz, Low-pass 500 Hz | +0.0005, -0.2536, -0.1871 | -0.0005, +0.2541, -0.0665 | collapse at Low-pass 1000 Hz | crackle (0.570) at Low-pass 1000 Hz |
| Temporal dropout | Dropout 10%, Dropout 25%, Dropout 50% | -0.0434, -0.1708, -0.2611 | +0.0434, +0.1274, +0.0903 | gradual | both (0.091) at Dropout 10% |

## Findings

- **Most harmful condition:** Dropout 50%, macro F1 0.1660 (-0.2611, -61.1% vs clean). At its most severe level, the most harmful family is temporal dropout (-0.2611).
- **Best tolerated family:** amplitude amplification (mean |change| 0.0169). Conditions within the seed-SD reference scale: Low-pass 2000 Hz.
- **Classes that fail first:** gaussian noise: normal at Noise 20 dB SNR; amplitude attenuation: crackle at Gain 0.5x (-6.0 dB); amplitude amplification: normal at Gain 1.5x (+3.5 dB); low-pass filtering: crackle at Low-pass 1000 Hz; temporal dropout: both at Dropout 10%.
- **Gradual vs collapse:** gaussian noise: gradual; amplitude attenuation: collapse (threshold at Gain 0.25x (-12.0 dB)); amplitude amplification: gradual; low-pass filtering: collapse (threshold at Low-pass 1000 Hz); temporal dropout: gradual.
- **Prediction collapse** (≥ 80% of cycles predicted as one class): Noise 5 dB SNR → crackle (80%), Noise 0 dB SNR → crackle (85%), Low-pass 500 Hz → normal (81%), Dropout 50% → normal (99%).

## Sensitivity by degradation type

**Bandwidth loss.** The model's features cover 50-2000 Hz only, so a 2000 Hz low-pass removes almost nothing the model sees (only the filter's ~±45 Hz transition band); 1000 Hz and 500 Hz remove the upper half and three quarters of the feature band. Macro-F1 change: Low-pass 2000 Hz +0.0005, Low-pass 1000 Hz -0.2536, Low-pass 500 Hz -0.1871. By source rate (macro F1, classes present), clean → 500 Hz: 4000 Hz 0.300 → 0.221, 44100 Hz 0.435 → 0.243.

**Amplitude changes.** A gain change shifts every Log-Mel band by the same number of dB (20·log10(gain)) and leaves spectral shape and timing intact; it carries no new respiratory information. Macro-F1 change: Gain 0.5x (-6.0 dB) -0.0223 (18.9% of predictions changed), Gain 0.25x (-12.0 dB) -0.0858 (36.3% of predictions changed), Gain 1.5x (+3.5 dB) -0.0166 (13.5% of predictions changed), Gain 2x (+6.0 dB) -0.0172 (21.8% of predictions changed).

**Missing temporal audio.** Dropped 20 ms blocks become exact silence, which Log-Mel maps to floor-level frames. Mean dropped share of samples and macro-F1 change: Dropout 10%: 10.1% dropped, -0.0434, Dropout 25%: 25.1% dropped, -0.1708, Dropout 50%: 50.2% dropped, -0.2611.

**Indicators of reliance on recording characteristics.** These are observations that would be worth investigating, not conclusions:
- Pure gain changes alter up to 36.3% of predictions and change macro F1 by up to -0.0858 (Gain 0.25x (-12.0 dB)). Because gain carries no respiratory information, sensitivity here indicates dependence on absolute recording level, which varies with device, microphone placement and gain settings. This exceeds the seed-SD reference scale.
- Largest shift in predicted-class counts under gain changes: Gain 0.5x (-6.0 dB): normal 387 → 542; Gain 0.25x (-12.0 dB): normal 387 → 694; Gain 1.5x (+3.5 dB): normal 387 → 286; Gain 2x (+6.0 dB): normal 387 → 229.
- Low-pass 2000 Hz (removes essentially nothing inside the feature band): +0.0005. A change this small is consistent with the model ignoring content outside its band.
- Differences between source sample rates under the same degradation (tables below) can point to device- or bandwidth-specific cues; 4 kHz sources already contain nothing above 2 kHz.

## Post-hoc observations (added after inspecting results; not predeclared)

- **Monotonicity with severity:** gaussian noise: monotonic; amplitude attenuation: monotonic; amplitude amplification: monotonic; low-pass filtering: NOT monotonic (0.428, 0.173, 0.240); temporal dropout: monotonic. A non-monotonic family means the more severe level hurts less, which points to the model reacting to an unfamiliar input pattern rather than to gradual information loss.
- **Direction of the prediction shift** (class whose predicted count changes most vs clean):
  - Noise 20 dB SNR: crackle 400 → 686
  - Noise 10 dB SNR: crackle 400 → 798
  - Noise 5 dB SNR: crackle 400 → 864
  - Noise 0 dB SNR: crackle 400 → 911
  - Gain 0.5x (-6.0 dB): normal 387 → 542
  - Gain 0.25x (-12.0 dB): normal 387 → 694
  - Gain 1.5x (+3.5 dB): normal 387 → 286
  - Gain 2x (+6.0 dB): normal 387 → 229
  - Low-pass 2000 Hz: wheeze 240 → 239
  - Low-pass 1000 Hz: crackle 400 → 4
  - Low-pass 500 Hz: normal 387 → 872
  - Dropout 10%: wheeze 240 → 120
  - Dropout 25%: normal 387 → 837
  - Dropout 50%: normal 387 → 1064
- **Two-level families:** amplitude attenuation met the collapse rule with only two severity levels, which is weak evidence of a threshold.

## Subgroups: macro F1 (classes present) by source sample rate

| Condition | 4000 Hz (n=155) | 44100 Hz (n=922) |
|---|---|---|
| Clean | 0.300 | 0.435 |
| Noise 20 dB SNR | 0.171 | 0.396 |
| Noise 10 dB SNR | 0.132 | 0.347 |
| Noise 5 dB SNR | 0.123 | 0.260 |
| Noise 0 dB SNR | 0.110 | 0.220 |
| Gain 0.5x (-6.0 dB) | 0.243 | 0.417 |
| Gain 0.25x (-12.0 dB) | 0.246 | 0.348 |
| Gain 1.5x (+3.5 dB) | 0.280 | 0.419 |
| Gain 2x (+6.0 dB) | 0.286 | 0.419 |
| Low-pass 2000 Hz | 0.300 | 0.436 |
| Low-pass 1000 Hz | 0.142 | 0.179 |
| Low-pass 500 Hz | 0.221 | 0.243 |
| Dropout 10% | 0.290 | 0.402 |
| Dropout 25% | 0.269 | 0.255 |
| Dropout 50% | 0.187 | 0.161 |

## Subgroups: accuracy by device

| Condition | AKGC417L (n=647, 4 pts) | Litt3200 (n=74, 2 pts) | LittC2SE (n=119, 2 pts) | Meditron (n=237, 12 pts) |
|---|---|---|---|---|
| Clean | 0.451 | 0.324 | 0.672 | 0.646 |
| Noise 20 dB SNR | 0.485 | 0.243 | 0.244 | 0.532 |
| Noise 10 dB SNR | 0.499 | 0.176 | 0.118 | 0.468 |
| Noise 5 dB SNR | 0.493 | 0.176 | 0.118 | 0.359 |
| Noise 0 dB SNR | 0.501 | 0.135 | 0.101 | 0.295 |
| Gain 0.5x (-6.0 dB) | 0.417 | 0.324 | 0.798 | 0.637 |
| Gain 0.25x (-12.0 dB) | 0.359 | 0.392 | 0.874 | 0.658 |
| Gain 1.5x (+3.5 dB) | 0.474 | 0.297 | 0.437 | 0.586 |
| Gain 2x (+6.0 dB) | 0.485 | 0.284 | 0.353 | 0.557 |
| Low-pass 2000 Hz | 0.453 | 0.324 | 0.672 | 0.646 |
| Low-pass 1000 Hz | 0.151 | 0.243 | 0.639 | 0.515 |
| Low-pass 500 Hz | 0.329 | 0.311 | 0.714 | 0.612 |
| Dropout 10% | 0.451 | 0.284 | 0.597 | 0.612 |
| Dropout 25% | 0.360 | 0.527 | 0.765 | 0.544 |
| Dropout 50% | 0.329 | 0.568 | 0.933 | 0.599 |

Subgroups have few patients and are noisy; they are descriptive only.
