# Acoustic Biomarker Classifier

I started this project with a simple question. If I train a model to recognize sounds in lung recordings, does it still work on patients it has never heard before? And when the audio gets noisy or distorted, or the model is simply wrong, can its own confidence tell me when not to trust it?

> **This is a research and educational project, not a medical device or diagnostic system.** It hasn't been clinically validated, it makes no clinical or diagnostic claims, and it shouldn't be used for any medical decision.

Short version: it generalized worse than validation suggested, it broke in systematic ways under degraded audio, and its confidence turned out to be a weak safety net. The model was developed only on validation data and then tested **once** on 19 patients it had never seen, under a protocol I froze and hash-verified beforehand. The full write-up is in **[reports/final_report/final_report.md](reports/final_report/final_report.md)**. A research manuscript based on it is in [`paper/`](paper/acoustic_biomarker_classifier_manuscript.md). It's an independent preprint and hasn't been peer reviewed; its DOI will be added once it's published.

The current software and documentation release is [v1.0.1](https://github.com/rish-mishra/acoustic-biomarker-classifier/releases/tag/v1.0.1). The frozen model checkpoint was published once, as an asset of the [v1.0.0 release](https://github.com/rish-mishra/acoustic-biomarker-classifier/releases/tag/v1.0.0), and is unchanged.

## Why this question

A lot of published respiratory-sound results look better than they should. Recordings from the same patient often end up in both the training and test sets, so a model can partly succeed by recognizing the patient or their recording setup rather than the sound itself. Evaluation is also usually done on clean audio only. I wanted to test the harder and more honest version of the problem:

> How well can respiratory-sound classifiers generalize to completely unseen patients and degraded real-world audio, and can uncertainty estimation identify predictions that should not be trusted?

## How it works

A stethoscope recording contains several breaths. Each breath is annotated as a separate **respiratory cycle**, and each cycle carries one of four labels. *Normal* means no adventitious sound. *Crackle* means short popping sounds. *Wheeze* means a sustained, musical tone. *Both* means crackles and wheezes together.

The model sees one cycle at a time. I turn the cycle's audio into a **log-Mel spectrogram**, which is basically a picture of how much energy the sound has at each frequency over time, on a scale closer to how we hear. A small CNN reads that picture and outputs a probability for each of the four classes. The highest probability is the prediction, and the probabilities themselves give a **confidence** (the top probability) and an **uncertainty** (how spread out the probabilities are, measured as entropy). Much of the project is about whether those last two numbers mean anything.

## The dataset, and why it's tricky

I used the [ICBHI 2017 Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge): 126 patients, 920 recordings and 6,898 respiratory cycles. The labels are imbalanced, at roughly 53% normal, 27% crackle, 13% wheeze and 7% both. The raw audio and annotations aren't redistributed here; if you want to rerun things, get them from the dataset's official source and put them under `data/raw/`, which is git-ignored. The dataset was created by Rocha et al.; please cite their work if you use it:

- B. M. Rocha et al., "A Respiratory Sound Database for the Development of Automated Classification," in *Precision Medicine Powered by pHealth and Connected Health* (IFMBE Proceedings), Springer, 2018, pp. 33–37. doi:[10.1007/978-981-10-7419-6_6](https://doi.org/10.1007/978-981-10-7419-6_6)
- B. M. Rocha et al., "An open access database for the evaluation of respiratory sound classification algorithms," *Physiological Measurement*, 40(3):035001, 2019. doi:[10.1088/1361-6579/ab03ea](https://doi.org/10.1088/1361-6579/ab03ea)

**Patient split.** The most important design choice was splitting by patient, 88 / 19 / 19 for train, validation and test (4,741 / 1,077 / 1,080 cycles). No patient appears in more than one split, so every validation and test score is about patients the model has never seen. Out of 5,000 random candidate splits, I kept the one whose mix of labels, recording devices and sample rates best matched the full dataset ([`data/splits/`](data/splits/)). Diagnosis was never used to build the split.

**Recording devices.** ICBHI recordings come from four stethoscopes and recorders (AKGC417L, Meditron, LittC2SE and Litt3200), and device is tangled up with almost everything else. Every Healthy, URTI, bronchiectasis, bronchiolitis and LRTI recording comes from the Meditron. The AKGC417L and Litt3200 were only used on COPD patients. The devices also record at different sample rates (4 kHz, 10 kHz or 44.1 kHz), and a 4 kHz recording physically can't contain anything above 2 kHz. So a model can get decent scores by learning what a *device* sounds like instead of what a *crackle* sounds like. Matching the device mix across splits reduces this risk, but it can't remove it.

**Clipping and silence.** Many recordings are clipped, meaning the signal hit the recorder's maximum level and got flattened. Rather than throw those cycles out, I fixed three quality groups on 2026-10-03, before training anything: *all* cycles, *moderately clean* (clipped fraction ≤ 0.10), and *clean* (clipped fraction ≤ 0.01 with no clipped stretch of 32 ms or more). There was also a sneakier problem. Most AKGC417L recordings start with about 90 ms of exact digital silence, which is effectively a device fingerprint. I trim leading exact zeros from every cycle, on every device, so the model can't use that shortcut.

## Method

| Step | What I did |
|---|---|
| Preprocessing | Slice each cycle at its native sample rate, trim leading exact zeros (up to 150 ms), resample to 16 kHz, remove the mean |
| Representation | 64-band log-Mel spectrogram, 50–2000 Hz, 32 ms window, 10 ms hop; each band standardized with statistics from the training split only; cycles keep their natural length and get masked when padded |
| Architectures | Baseline CNN (295k parameters), CRNN (the CNN plus a bidirectional GRU, 353k), ImageNet-pretrained MobileNetV3 (1.52M) |
| Training | AdamW, class-weighted cross-entropy, length-bucketed batches, early stopping on validation macro F1, deterministic cuDNN |
| Main metric | Macro F1 over the four classes, with patient-level bootstrap 95% intervals (2,000 resamples, seed 0) |

The 2 kHz upper limit is deliberate. It means a 4 kHz recording fills the whole frequency range the model looks at, so the model can't spot band-limited devices just by noticing empty upper frequencies. Macro F1 averages the F1 score of each class equally, so a model can't look good by only getting the common "normal" class right. The bootstrap resamples whole patients rather than individual cycles, because cycles from the same person aren't independent.

Model selection used validation data only. I first ran a small learning-rate and dropout study on the baseline CNN, with all four runs declared in advance so I couldn't keep adding runs until one looked lucky. The best was `baseline_cnn_v3` (learning rate 1e-4, dropout 0.3), at a validation macro F1 of 0.4271. With the same settings, the CRNN reached 0.4153 and the pretrained MobileNetV3 reached 0.4381.

Those differences were small, so before picking anything I retrained each architecture with three seeds. I'd written down the decision rule beforehand: an architecture only "wins" if it beats every other one on all three seeds by more than the seed-to-seed noise; otherwise I take the simplest model that's within that noise of the best.

| Architecture | Validation macro F1 (seeds 42 / 1337 / 2026) | Mean ± SD |
|---|---|---|
| **Baseline CNN** | 0.4271 / 0.4226 / 0.4225 | **0.424 ± 0.003** |
| CRNN | 0.4153 / 0.4367 / 0.3869 | 0.413 ± 0.025 |
| MobileNetV3 (pretrained) | 0.4381 / 0.4372 / 0.4194 | 0.432 ± 0.011 |

Nothing beat the others convincingly. The baseline CNN was the smallest model and by far the most stable across seeds, so the rule picked it. The frozen reference model is `baseline_cnn_v3` (seed 42, epoch 13), the run I'd already selected before the seed study. I used the seed study to check that this choice was stable, not to fish for the luckiest seed.

## Results

I keep three kinds of result separate throughout. **Validation** numbers come from development, where I was allowed to look and make decisions. **Test** numbers come from the single locked evaluation on patients the model never saw. **Post-hoc** findings came from analysis I designed after seeing the test results, so they're exploratory only.

### Unseen patients were much harder

| Metric | Validation | **Test (19 unseen patients)** |
|---|---|---|
| **Macro F1** | 0.4271 | **0.3175** (95% CI 0.218–0.413) |
| Balanced accuracy | 0.4568 | 0.4030 |
| Accuracy | 0.5097 | 0.3935 |

The drop from 0.4271 to 0.3175 is the headline result. Some of it was expected, because I chose the model on validation, which naturally flatters its validation score. Still, the validation number sits just above the top of the test interval.

The per-class test F1 scores show where it went wrong: normal 0.403, crackle 0.456, wheeze 0.370 and **both 0.041**. The "both" class essentially failed, with 3 correct out of 78, and most of those cycles were predicted as wheeze. The other big problem was normal cycles: 342 of the 647 normal test cycles were predicted as crackle. Clipped audio doesn't explain the gap, since the *clean* quality group scored 0.332.

### Degraded audio broke the model in consistent ways

Before testing, I'd fixed 15 synthetic degradations: Gaussian noise at 20, 10, 5 and 0 dB SNR; volume scaling by 0.5×, 0.25×, 1.5× and 2×; low-pass filtering at 2000, 1000 and 500 Hz; and temporal dropout, where 10%, 25% or 50% of the audio is replaced by 20 ms gaps of silence. Here are the most informative ones (macro F1):

| Condition | Validation | **Test** |
|---|---|---|
| Clean | 0.427 | **0.318** |
| Gaussian noise, 0 dB SNR | 0.208 | **0.137** |
| Low-pass 1000 Hz | 0.173 | **0.157** |
| Low-pass 500 Hz | 0.240 | 0.246 |
| Temporal dropout 50% | 0.166 | **0.200** |
| Gain 0.25× / 2× | 0.341 / 0.410 | 0.314 / 0.297 |

What convinced me these were real model behaviours, not quirks of one split, is how many of them showed up on both validation and test. Heavy noise pushed nearly every prediction to *crackle* (94% of test cycles at 0 dB). Heavy dropout pushed predictions to *normal*. Changing only the volume shifted predictions, even though volume carries no information about the breath sound. Removing everything above 2 kHz did nothing, which makes sense because the model never looks above 2 kHz.

The result that surprised me most was the low-pass filter. Cutting at 1000 Hz hurt *more* than cutting at 500 Hz, on both splits. That's not what gradual loss of information would look like. It looks more like the model reacting to an unfamiliar input pattern, flooding to wheeze at 1000 Hz and to normal at 500 Hz.

### Confidence wasn't a reliable safety mechanism

Here I was asking two questions. Do the model's probabilities match how often it's actually right (*calibration*), and can its confidence pick out its own mistakes? To fix calibration I used *temperature scaling*, a single number that softens or sharpens the probabilities without changing any prediction. I fitted it on validation with leave-one-patient-out cross-validation and froze it at T = 1.386 for the test.

| | Validation | Test |
|---|---|---|
| Calibration error (ECE), raw → frozen temperature | 0.088 → 0.050 (out-of-fold) | 0.208 → 0.127 |
| Error detection AUROC (1 − confidence / entropy) | 0.599 / 0.620 | 0.542 / 0.554 |
| Error rate when keeping 100% → 50% of cycles | 0.490 → 0.416 | 0.606 → 0.561 |

Temperature scaling carried over partially: it reduced the average overconfidence gap on test by about 40%, from +0.208 to +0.127. But being better calibrated on average isn't the same as knowing *which* predictions are wrong. As an error detector, confidence was close to a coin flip on test (an AUROC of 0.5 means no better than chance). Letting the model abstain on its least confident half of the cycles only lowered the error rate from 0.606 to 0.561. It got worse under degradation, too. Confidence thresholds learned on clean validation data kept *more* inputs as the noise got louder, and between 36% and 42% of noisy test cycles were confidently wrong.

## What I took away from it

Unseen patients were much harder than validation suggested, and the failures under degraded audio were systematic rather than random. The model's sensitivity to volume and bandwidth points to it relying partly on *how* a recording was made, not only on the breath sounds in it. Calibration can fix the model's average confidence without turning that confidence into a dependable warning signal, and abstention gave only limited protection. None of that makes this a useful classifier. What it does do is show, under a strict and honest evaluation, how hard robustness and patient-level generalization still are for respiratory-sound models.

## A post-hoc finding (exploratory, not predeclared)

After the test was locked, I went back to the saved predictions to see where the normal→crackle mistakes came from. **324 of 342 of those errors (95%) came from the six test patients recorded with the AKGC417L device.** All six were affected, and no single patient explains the overall result.

It's tempting to blame the device, but I can't, and I'm not claiming it. Device here is confounded with patient identity, recording conditions, sample rate and diagnosis mix, and I only found this pattern after seeing the test results. It's a lead worth following with a properly predeclared held-out-device experiment, not a conclusion. The details are in [`reports/final_evaluation/exploratory/`](reports/final_evaluation/exploratory/summary.md).

## Reproducibility

The test set is only useful if you don't peek at it. I wanted to avoid the familiar trap of checking test scores, tweaking the model, and checking again until the number looked good. So everything was set up to make a second look impossible, or at least visible.

Every model, hyperparameter, temperature and threshold was chosen on validation, and each analysis plan was written down before its results existed:
- [learning-rate / dropout study](reports/runs/lr_dropout_study_plan.json)
- [seed study](reports/runs/seed_study/plan.json)
- [robustness](reports/robustness/noise_robustness/plan.json)
- [calibration](reports/calibration_uncertainty/plan.json)

Before touching the test set, I froze everything in [`reports/final_evaluation/protocol.md`](reports/final_evaluation/protocol.md) (version 3): the model and its checkpoint hash, the data pipeline, the metrics, the temperature and thresholds, and the 15 degradations. The protocol also records an audit showing nothing had touched the test split, plus a manifest of content hashes (git blob IDs for text files, SHA-256 for the checkpoint, metadata and audio).

The evaluator ([`src/evaluation/final_evaluation.py`](src/evaluation/final_evaluation.py)) refuses to touch test data without an explicit unlock token and a clean, committed working tree. Before unlocking, it reproduced the validation predictions bit for bit. It then ran each test stage exactly once. A STARTED marker blocks re-runs, and the outputs are hash-locked in `clean_test/lock.json` and `robustness_test/lock.json`. Seeds and GPU settings are deterministic, and every confidence interval resamples whole patients.

A few housekeeping details matter if you verify things yourself:
- **Expected `--verify` differences.** After the test, I updated two files that the frozen manifest happened to include: this README and `tests/test_final_protocol.py`. So `final_protocol.py --verify` reports exactly those two and nothing else. Every inference, data, checkpoint, calibration and plan artifact still verifies. The frozen pre-test state is commit `9d36c12`, where verification matches exactly ([`post_evaluation_note.md`](reports/final_evaluation/post_evaluation_note.md)).
- **Line endings.** `.gitattributes` pins the line endings of every hash-locked file, so a fresh clone on any platform reproduces the locked bytes.
- **Checkpoint.** The reference checkpoint is attached to the [v1.0.0 release](https://github.com/rish-mishra/acoustic-biomarker-classifier/releases/tag/v1.0.0) as `baseline_cnn_v3_seed42_best.pt`. It wasn't re-uploaded with v1.0.1, which changed only documentation and tests. Its size and SHA-256 are in [`checkpoint_release.md`](reports/final_evaluation/checkpoint_release.md).
- **Tests.** The test suite (338/338 passing) runs on synthetic data and checks, among other things, that every number quoted here matches the stored results.

### Running it yourself

I used Python 3.11.9, torch 2.11.0+cu130 and an NVIDIA GPU; the exact package list is in [`environment_pip_freeze.txt`](reports/final_evaluation/environment_pip_freeze.txt). The raw audio and model checkpoints (`*.pt`) are git-ignored, so download the dataset separately and grab the checkpoint from the release.

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

The two test stages (`final_evaluation.py clean-test` and `robustness-test`) have already run once and will refuse to run again. Their locked results live under `reports/final_evaluation/`.

## Repository structure

```text
acoustic-biomarker-classifier/
├── data/
│   ├── raw/                  # ICBHI 2017 audio + annotations (git-ignored, not distributed)
│   ├── splits/               # frozen patient-level split + composition summary
│   └── normalization/        # train-only log-Mel band statistics
├── models/                   # checkpoints (git-ignored)
├── paper/                    # research manuscript (preprint) + Zenodo metadata draft
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
├── tests/                    # unit and end-to-end tests (synthetic data)
└── LICENSE                   # MIT, original source code only (see License below)
```

## Limitations and what I'd do next

The biggest limitation is size. With only 19 validation and 19 test patients, the confidence intervals are wide, and the test "both" class comes almost entirely from two patients. Everything also rests on one dataset, one split and one frozen model. In ICBHI, device, diagnosis and sample rate are tangled together, and I didn't evaluate on an external dataset or on a held-out device. The degradations are synthetic, so they probe the model's sensitivities rather than measure real-world recording conditions. Accuracy-based calibration measures can also look deceptively good when predictions collapse onto the majority class. And there's no clinical validation of any kind.

If I kept going, the next step would be a predeclared held-out-device evaluation to test the AKGC417L lead properly. After that, I'd look at saliency maps to see what the model actually attends to, try external datasets, and maybe build a small interactive demo.

## License

The original source code in this repository is released under the [MIT License](LICENSE). That licence covers the project's own code only. It does **not** grant any rights to:
- the ICBHI 2017 Respiratory Sound Database, which isn't redistributed here and stays under its own terms;
- any other third-party dataset;
- third-party pretrained model weights, such as the ImageNet weights used to initialize MobileNetV3, which keep their own licences;
- any other third-party material;
- the research manuscript in `paper/`, which is released separately under CC BY 4.0.

The MIT licence also isn't intended to cover the trained checkpoint attached to the v1.0.0 release. That checkpoint was trained on ICBHI 2017 data and is provided for research reproducibility; any use of it is subject to the dataset's own terms. Derived files included for reproducibility, such as the split, normalization statistics and prediction outputs, don't relicense the underlying dataset.

## Disclaimer

This repository is a research and educational project on respiratory-sound classification, generalization and model reliability. It is not a medical device, it hasn't been clinically validated, and it must not be used for diagnosis or medical decision-making.
