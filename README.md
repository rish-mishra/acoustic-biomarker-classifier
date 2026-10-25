# Acoustic Biomarker Classifier

**Patient-independent respiratory-sound classification on ICBHI 2017, testing generalization to unseen patients, robustness to degraded audio, and whether uncertainty can flag predictions that should not be trusted.**

> **Research and educational project.** This is **not** a medical device or diagnostic system. It makes no clinical or diagnostic claims and must not be used for medical decisions.

**Status: complete.** The model was developed on validation data only, then evaluated once on 19 untouched test patients under a frozen, hash-verified protocol. The full write-up is **[reports/final_report/final_report.md](reports/final_report/final_report.md)**.

---

## Research question

> How well can respiratory-sound classifiers generalize to completely unseen patients and degraded real-world audio, and can uncertainty estimation identify predictions that should not be trusted?

**Motivation.** Many published respiratory-sound results let one patient's recordings appear in both training and test sets, and they evaluate only on clean audio. This project measures three things under stricter conditions:
1. **patient-independent performance**, where no patient appears in more than one split;
2. **robustness** to controlled audio degradations;
3. **trustworthiness:** whether calibrated confidence and predictive uncertainty can identify predictions to defer.

## Dataset and confounds

**[ICBHI 2017 Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge):** 126 patients, 920 recordings and 6,898 respiratory cycles. Each cycle is labelled **normal / crackle / wheeze / both** (53% / 27% / 13% / 7%). Raw audio is not distributed; place it under `data/raw/` (git-ignored).

- **Device ↔ diagnosis.** Every Healthy, URTI, bronchiectasis, bronchiolitis and LRTI recording comes from the Meditron stethoscope. AKGC417L and Litt3200 were used only on COPD patients.
- **Device ↔ bandwidth.** Sample rates are 4 kHz, 10 kHz or 44.1 kHz, so a model can learn device or bandwidth artifacts instead of respiratory sounds.
- **Clipping.** Clipping is common. Three quality groups were fixed on 2026-10-03, before any model was trained:
  - *all*;
  - *moderately clean*: clipped fraction ≤ 0.10;
  - *clean*: clipped fraction ≤ 0.01 and no clipped plateau of 32 ms or more.
- **Device fingerprint.** Most AKGC417L recordings start with about 90 ms of exact digital zeros. These are trimmed from every cycle on every device.

## Method

| Stage | Choice |
|---|---|
| **Split** | Fixed patient-level split, 88 / 19 / 19 patients (4,741 / 1,077 / 1,080 cycles); no patient overlap; matched label, device and sample-rate mix; diagnosis not used ([`data/splits/`](data/splits/)) |
| **Preprocessing** | Cycle slice at native rate → trim leading exact zeros (≤ 150 ms) → resample to 16 kHz → remove mean |
| **Representation** | 64-band log-Mel, 50–2000 Hz, 32 ms window, 10 ms hop; per-band standardization fitted on train only; natural cycle length with masking |
| **Architectures** | Baseline CNN (295k parameters); CRNN (CNN + BiGRU, 353k); ImageNet-pretrained MobileNetV3 (1.52M) |
| **Training** | AdamW, class-weighted cross-entropy, length-bucketed batches, early stopping on validation macro F1, deterministic cuDNN |
| **Primary metric** | Macro F1 over the 4 classes, with patient-level bootstrap 95% intervals (2,000 resamples, seed 0) |

**Model selection (validation only):**
1. A predeclared learning-rate/dropout study chose `baseline_cnn_v3` (learning rate 1e-4, dropout 0.3; validation macro F1 0.4271).
2. CRNN (0.4153) and MobileNetV3 (0.4381) were trained with the same settings.
3. A predeclared **three-seed study** found no convincing winner relative to seed noise, so its rule recommended the simplest stable model.

| Architecture | Validation macro F1 (seeds 42 / 1337 / 2026) | Mean ± SD |
|---|---|---|
| **Baseline CNN** | 0.4271 / 0.4226 / 0.4225 | **0.424 ± 0.003** |
| CRNN | 0.4153 / 0.4367 / 0.3869 | 0.413 ± 0.025 |
| MobileNetV3 (pretrained) | 0.4381 / 0.4372 / 0.4194 | 0.432 ± 0.011 |

**Frozen reference model:** `baseline_cnn_v3` (seed 42, epoch 13), the run selected before the seed study. The seed study checked stability; it was not used to pick a best seed.

## Results

Each result is labelled by kind of evidence:
- **validation** results come from development;
- **test** results come from the single locked evaluation;
- **post-hoc** results were analysed after the test and are exploratory.

### Clean performance: validation vs untouched test

| Metric | Validation | **Test (19 unseen patients)** |
|---|---|---|
| **Macro F1 (primary)** | 0.4271 | **0.3175** (95% CI 0.218–0.413) |
| Balanced accuracy | 0.4568 | 0.4030 |
| Accuracy | 0.5097 | 0.3935 |

**Test per-class F1:** normal 0.403, crackle 0.456, wheeze 0.370, **both 0.041**.
- **"Both" essentially fails** on test: 3 of 78 correct, most predicted wheeze.
- **342 of 647 normal cycles are predicted crackle.**
- **Clipping doesn't explain the gap:** the clean-audio group scores 0.332.

### Robustness: 15 frozen degradations (macro F1)

| Condition | Validation | **Test** |
|---|---|---|
| Clean | 0.427 | **0.318** |
| Gaussian noise, 0 dB SNR | 0.208 | **0.137** |
| Low-pass 1000 Hz | 0.173 | **0.157** |
| Low-pass 500 Hz | 0.240 | 0.246 |
| Temporal dropout 50% | 0.166 | **0.200** |
| Gain 0.25× / 2× | 0.341 / 0.410 | 0.314 / 0.297 |

These behaviours carried over from validation to test:
- noise collapses predictions onto *crackle* (94% at 0 dB on test);
- the 1000 Hz low-pass is worse than 500 Hz;
- dropout collapses predictions onto *normal*;
- gain alone shifts predictions;
- the 2000 Hz control low-pass changes nothing.

### Calibration, uncertainty and abstention

| | Validation | Test |
|---|---|---|
| ECE, raw → frozen validation temperature (T = 1.386) | 0.088 → 0.050 (out-of-fold) | 0.208 → 0.127 |
| Error-detection AUROC (1 − confidence / entropy) | 0.599 / 0.620 | 0.542 / 0.554 |
| Error rate at 100% → 50% coverage | 0.490 → 0.416 | 0.606 → 0.561 |

- **Temperature scaling** fitted on validation roughly halves the average overconfidence on test. It cannot change any prediction.
- **Uncertainty is a weak error detector,** close to chance on test.
- **Abstention lowers error only modestly.** Thresholds learned on clean validation data keep *more* inputs as noise increases, and 36–42% of noisy test cycles are confidently wrong.

### Post-hoc exploratory: not predeclared
**95% of the test normal→crackle errors (324 of 342) come from the six patients recorded with the AKGC417L device.** No single patient explains the overall result.

This association is **confounded** with patient identity, recording conditions, sample rate and diagnosis mix, and it was found after the test. It does **not** show that the device causes the errors. Details: [`reports/final_evaluation/exploratory/`](reports/final_evaluation/exploratory/summary.md).

## Conclusions

- **Unseen patients:** performance was substantially weaker than validation suggested (macro F1 0.318 against 0.427).
- **Systematic sensitivities:** several degradation behaviours reproduced from validation to test, so the model's sensitivities are systematic rather than split-specific.
- **Recording domain:** sensitivity to gain and bandwidth, together with the confounded post-hoc device association, suggests recording-domain characteristics matter alongside respiratory content.
- **Calibration vs detection:** calibration can correct *average* confidence without making uncertainty an effective failure detector, and selective prediction provides only limited safety.
- **Overall:** robustness and patient-level generalization remain major obstacles for respiratory-sound classifiers.

## Limitations

- **Small evaluation sets.** Only 19 validation and 19 test patients, so intervals are wide. The test "both" class comes almost entirely from two patients.
- **Single dataset, split and model.** Device, diagnosis and sample rate are entangled, and there is no external dataset or held-out-device evaluation.
- **Synthetic degradations** probe sensitivities; they aren't real-world acoustic conditions.
- **Accuracy-based calibration measures can hide class collapse.**
- **No clinical validation.**

---

## Reproducibility

- **Validation-only development.** Every model, hyperparameter, temperature and threshold was chosen on validation. Analysis plans were written before their results:
  - [`lr_dropout_study_plan.json`](reports/runs/lr_dropout_study_plan.json)
  - [seed study](reports/runs/seed_study/plan.json)
  - [robustness](reports/robustness/noise_robustness/plan.json)
  - [calibration](reports/calibration_uncertainty/plan.json)
- **Frozen protocol.** [`reports/final_evaluation/protocol.md`](reports/final_evaluation/protocol.md) (version 3) froze the model, data pipeline, metrics, temperature, thresholds and degradations, and recorded a test-access audit. Its manifest holds content hashes: git blob IDs for text files, SHA-256 for the checkpoint, metadata and audio.
- **One-time test with locks.** The evaluator ([`src/evaluation/final_evaluation.py`](src/evaluation/final_evaluation.py)) needs an unlock token and a clean working tree. It reproduced the validation predictions bitwise before unlocking, and it ran each test stage once. A STARTED marker blocks re-runs, and the outputs are hash-locked in `clean_test/lock.json` and `robustness_test/lock.json`.
- **Determinism and statistics.** Fixed seeds and deterministic GPU settings throughout; patient-level bootstrap intervals throughout.

### Reproducing the work

The environment was Python 3.11.9, torch 2.11.0+cu130 and an NVIDIA GPU. The exact package list is in [`environment_pip_freeze.txt`](reports/final_evaluation/environment_pip_freeze.txt). Model checkpoints (`*.pt`) and the raw audio are git-ignored and not distributed. The reference checkpoint's SHA-256 is in the protocol.

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r reports/final_evaluation/environment_pip_freeze.txt --extra-index-url https://download.pytorch.org/whl/cu130

python src/data/build_metadata.py                          # data/metadata.csv from the ICBHI files
# data/splits/patient_split.csv and the train-only normalization stats are committed and frozen
python src/models/train.py --run-name baseline_cnn_v3 --lr 1e-4                    # reference model
python src/models/train.py --run-name crnn_v1 --model crnn --lr 1e-4
python src/models/train.py --run-name mobilenetv3_pretrained_v1 --model mobilenetv3_pretrained --lr 1e-4
python src/evaluation/analyze_run.py baseline_cnn_v3       # validation-only analysis

python src/models/run_seed_study.py && python src/evaluation/seed_study.py        # seed study
python src/evaluation/noise_robustness.py                  # validation robustness (resumes; never overwrites)
python src/evaluation/calibration_uncertainty.py           # calibration / uncertainty / abstention

python src/evaluation/final_protocol.py --verify           # check every frozen artifact
python src/evaluation/final_evaluation.py dry-run-validation
python src/evaluation/final_report_figures.py              # figures for the final report

python -m unittest discover -s tests                       # test suite
```

The test stages (`final_evaluation.py clean-test` / `robustness-test`) have already been run once. They refuse to run again; the locked results are under `reports/final_evaluation/`.

**Expected `--verify` note.** After the final evaluation, two files that protocol v3's manifest froze were updated as documentation and test maintenance: this README and `tests/test_final_protocol.py`. So `final_protocol.py --verify` now reports those two as changed, and nothing else. All inference code, data, checkpoint, calibration and plan artifacts still verify. The frozen pre-test state is commit `9d36c12`, where verification matches exactly. Details: [`post_evaluation_note.md`](reports/final_evaluation/post_evaluation_note.md).

`.gitattributes` pins the line endings of the hash-locked files, so a fresh clone on any platform reproduces the locked bytes. The reference checkpoint will be published as a release asset. Its filename, size and SHA-256 are in [`checkpoint_release.md`](reports/final_evaluation/checkpoint_release.md).

## Repository structure

```text
acoustic-biomarker-classifier/
├── data/
│   ├── raw/                  # ICBHI 2017 audio + annotations (git-ignored, not distributed)
│   ├── splits/               # frozen patient-level split + composition summary
│   └── normalization/        # train-only log-Mel band statistics
├── models/                   # checkpoints (git-ignored)
├── reports/
│   ├── runs/                 # training runs, model comparison, seed study
│   ├── robustness/           # validation degradation study
│   ├── calibration_uncertainty/  # validation calibration, uncertainty, abstention
│   ├── final_evaluation/     # frozen protocol, locked clean + robustness test, post-hoc exploratory
│   └── final_report/         # final report and figures
├── src/
│   ├── data/                 # metadata, patient split, dataset, normalization
│   ├── features/             # audio preprocessing, log-Mel, degradations
│   ├── models/               # CNN / CRNN / MobileNetV3, training, seed-study runner
│   └── evaluation/           # metrics, analyses, calibration, robustness, final protocol + evaluator
└── tests/                    # unit and end-to-end tests (synthetic data)
```

## Not done / future work

Not attempted, and possible next steps:
- a predeclared **held-out-device** evaluation;
- explainability (saliency maps);
- an interactive demo;
- external datasets.

## Disclaimer

This repository is a research and educational project on respiratory-sound classification, generalization and model reliability. It is not a medical device, has not been clinically validated, and must not be used for diagnosis or medical decision-making.
