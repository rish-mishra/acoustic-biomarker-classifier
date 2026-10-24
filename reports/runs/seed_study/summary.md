# Seed-variability study (validation only)

Seeds [42, 1337, 2026]; best-epoch validation metrics; SD = sample SD over seeds. Test split not evaluated.

| Architecture | Params | Macro F1 s42 | Macro F1 s1337 | Macro F1 s2026 | Mean ± SD | Min–max | Bal. acc. | Acc. | Best epochs | Train time (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_cnn | 294,740 | 0.4271 | 0.4226 | 0.4225 | **0.424 ± 0.003** | 0.423–0.427 | 0.434 ± 0.020 | 0.517 ± 0.009 | [13, 20, 7] | 44.008 ± 27.416 |
| crnn | 353,365 | 0.4153 | 0.4367 | 0.3869 | **0.413 ± 0.025** | 0.387–0.437 | 0.411 ± 0.029 | 0.539 ± 0.033 | [7, 10, 23] | 117.302 ± 86.777 |
| mobilenetv3_pretrained | 1,521,668 | 0.4381 | 0.4372 | 0.4194 | **0.432 ± 0.011** | 0.419–0.438 | 0.421 ± 0.009 | 0.542 ± 0.012 | [2, 18, 4] | 385.603 ± 547.731 |

## Per-class F1 (mean ± SD over seeds)

| Architecture | normal | crackle | wheeze | both |
|---|---|---|---|---|
| baseline_cnn | 0.592 ± 0.020 | 0.548 ± 0.021 | 0.356 ± 0.025 | 0.200 ± 0.012 |
| crnn | 0.617 ± 0.021 | 0.608 ± 0.039 | 0.222 ± 0.069 | 0.204 ± 0.008 |
| mobilenetv3_pretrained | 0.630 ± 0.014 | 0.542 ± 0.019 | 0.329 ± 0.033 | 0.225 ± 0.007 |

## Seed-paired macro-F1 differences

Pooled within-architecture seed SD: **0.0157**; range of architecture means: **0.0186**.

| Comparison | s42 | s1337 | s2026 | Mean ± SD | Seeds A better |
|---|---|---|---|---|---|
| baseline_cnn − crnn | +0.0118 | -0.0141 | +0.0356 | +0.0111 ± 0.0249 | 2/3 |
| baseline_cnn − mobilenetv3_pretrained | -0.0109 | -0.0146 | +0.0031 | -0.0075 ± 0.0093 | 1/3 |
| crnn − baseline_cnn | -0.0118 | +0.0141 | -0.0356 | -0.0111 ± 0.0249 | 1/3 |
| crnn − mobilenetv3_pretrained | -0.0227 | -0.0005 | -0.0326 | -0.0186 ± 0.0164 | 0/3 |
| mobilenetv3_pretrained − baseline_cnn | +0.0109 | +0.0146 | -0.0031 | +0.0075 ± 0.0093 | 2/3 |
| mobilenetv3_pretrained − crnn | +0.0227 | +0.0005 | +0.0326 | +0.0186 ± 0.0164 | 3/3 |

## Predeclared decision

- Convincing winners: none
- Within one pooled SD of the best mean: ['baseline_cnn', 'mobilenetv3_pretrained']
- **Recommendation: baseline_cnn** (no convincing winner; simplest architecture whose mean macro F1 is within one pooled seed SD of the best mean)
