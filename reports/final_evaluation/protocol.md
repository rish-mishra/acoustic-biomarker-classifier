# Frozen final evaluation protocol

**Status: FROZEN — test split NOT evaluated; no test inference has been run.** Frozen 2026-10-24T18:44:30-04:00 at commit `ea25fb793d68ea2b87273c02e6fdfcb1c0a0ed91`. Machine-readable version with every hash: `protocol.json`. Verify before unlocking with `python src/evaluation/final_protocol.py --verify`.

## 1. Frozen model

- `baseline_cnn`, run `baseline_cnn_v3` (seed 42), checkpoint `models/baseline_cnn_v3/best.pt` (epoch 13, 294,740 parameters).
- Checkpoint SHA-256: `2c2c7c03ed00b6562c63ff88be12d43bd8f9ff890074f9dac4d4ce5874e3b567`.
- Validation macro F1 at selection: 0.4271.
- No retraining, no ensembling, no architecture/hyperparameter/checkpoint changes.
- Why seed 42: baseline_cnn_v3 (seed 42) was the reference run selected during the baseline optimization study, before the multi-seed study. The seed study (seeds 42/1337/2026) verified that architecture differences did not exceed seed noise and recommended baseline_cnn under its predeclared rule; it was used to check stability, not to pick the best seed. Seeds 1337 and 2026 are not evaluated on test.

## 2. Frozen data pipeline

- Patient split `data/splits/patient_split.csv` (train/val/test patients 88/19/19), SHA-256 `a5f61582b9048af097db8ec64c3278954d698d251773109a2db78efb26950ab4`.
- Metadata SHA-256 `19b2f39ada720ee4e3ce2284ed6936ad1b3ea18eb0215ebec38e809f0643d1bf`; normalization statistics SHA-256 `9f455708587724e6f7ccc37f088f2ed5978d1847a7000307aa5acf81bdf9bf9c` (train patients only (88), per-mel-band mean/std in dB). These three must also match the hashes stored inside the checkpoint.
- Audio: soundfile, float32 in [-1, 1]; multi-channel averaged to mono; slice [cycle_start_s, cycle_end_s) at the original sample rate; end up to 0.01 s past the file is clipped; leading samples exactly 0.0 removed, capped at 0.15 s, at least one sample kept; applied to every cycle on every device; torchaudio.functional.resample (windowed sinc) to 16000 Hz; no padding, cropping, amplitude normalization or augmentation.
- Log-Mel: 64 mels, 50–2000 Hz, n_fft 512, window 512, hop 160, 16000 Hz, floor 1e-10; DC removal (subtract waveform mean); Hann window; center=True with zero padding; power 2; HTK mel scale; no filter-area normalization; 10*log10(max(power, floor)).
- Model input: train-band standardization with the frozen statistics; batches padded with 0 and a valid-frame mask; sequential (unshuffled) evaluation loader, batch size 32, num_workers 0.
- Labels: normal = 0, crackle = 1, wheeze = 2, both = 3.
- Quality metadata: clipped fraction = share of original-rate samples with |x| >= 0.999; plateau count = contiguous runs with |x| >= 0.999 lasting >= 32 ms; measured on the annotated cycle at its original sample rate, before trimming and resampling.
- Determinism: train.set_determinism(0): cuDNN deterministic, benchmark off, deterministic algorithms (warn_only), CUBLAS_WORKSPACE_CONFIG=:4096:8.

## 3. Primary endpoint

- **Primary metric: macro F1 over the 4 classes on all test cycles (classification_metrics.classification_report; zero division gives 0).** It will not change after viewing test results.
- Secondary: balanced accuracy, accuracy, per-class precision, per-class recall, per-class F1, confusion matrix.
- Intervals: patient-level bootstrap, 2000 resamples, seed 0; Python random.Random(seed); each resample draws len(patients) patient ids with replacement (rng.choice over sorted patient ids) and includes all cycles of each draw; percentile: sorted values at indices int(0.025 n) and int(0.975 n) - 1.

## 4. Quality sensitivity groups (fixed 2026-10-03)

- **all**: every cycle (primary)
- **moderately_clean**: original_clipped_fraction <= 0.1
- **clean**: original_clipped_fraction <= 0.01 and clipped_plateau_count == 0
- Reported per group: n cycles and patients, macro F1 over 4 classes (zero division 0), macro F1 over classes present in the group (analyze_run.group_metrics convention), balanced accuracy (4 classes and classes present), accuracy, per-class F1, per-class support, confusion matrix, patient-bootstrap 95% interval of macro F1 (resampling the group's patients).

## 5. Calibration

- **Frozen temperature T = 1.3856213645470599** (reports/calibration_uncertainty/summary.json -> temperature_scaling.final_validation_temperature (fitted on all 19 validation patients, clean)). Applied once; never fitted or changed on test.
- Reported for raw and frozen-T probabilities: NLL, Brier, 15-bin equal-width ECE, overconfidence gap, reliability diagram; secondary: equal-mass ECE, classwise ECE, MCE (validation definitions).
- ECE binning: top-label ECE, 15 equal-width confidence bins on [0, 1]; cycle with confidence c goes to bin index min(14, max(0, ceil(15 c) - 1)), i.e. bins [0, 1/15], (1/15, 2/15], ..., (14/15, 1]; ECE = sum_b (n_b / N) |acc_b - conf_b|, acc_b = fraction correct and conf_b = mean confidence in bin b; empty bins contribute 0.
- Temperature scaling cannot change any predicted class (T > 0 preserves the argmax), so accuracy, macro F1 and the confusion matrix are identical for raw and scaled probabilities.

## 6. Uncertainty / error detection

- Scores: 1 - MSP, predictive entropy (nats) (primary); 1 - top-2 margin (predeclared in the validation plan).
- AUROC (Mann-Whitney, ties 1/2), AUPRC (average precision; tied scores one threshold), test error rate as the AUPRC chance reference; raw (primary) and frozen-T scaled (secondary). No new score after seeing test results.

## 7. Selective prediction

- Coverages 100%, 90%, 80%, 70%, 60%, 50%; retained n = floor(coverage x N + 0.5).
- Primary: selective risk (error rate among retained) and the risk-coverage curve; AURC and E-AURC; ranking raw MSP, most confident first, stable tie-break by file order. Secondary: macro F1, balanced accuracy, accuracy, retained / rejected counts by true class and by predicted class.
- Frozen clean-validation thresholds: **τ80 = 0.4616909623146057**, **τ50 = 0.5770796537399292**; raw MSP of the last retained clean-validation cycle at 80% / 50% coverage; applied unchanged to test (retain if MSP >= tau).
- Zero division: project-wide convention (classification_metrics._safe_div): any ratio with a zero denominator is 0, so a class with no retained true cycles has recall 0 and a class never predicted has precision 0, both giving F1 0; such a class still counts in the 4-class macro F1 and balanced accuracy and its zero support is shown in the retained counts (as in the validation plan, clarification 3)

## 8. Robustness

- **Primary final test evaluation:** untouched clean test audio through the frozen pipeline.
- **Secondary predeclared robustness evaluation:** the 15 frozen conditions below applied to test audio ONLY after the clean test outputs are written and locked.
- Lock: clean test predictions, logits and metrics are written to disk and their SHA-256 hashes recorded in a lock file before any degraded test inference starts; the degraded run's 'clean' condition must reproduce the locked clean predictions exactly.
- Conditions (frozen): clean, noise_snr20db, noise_snr10db, noise_snr5db, noise_snr0db, gain_0.5x, gain_0.25x, gain_1.5x, gain_2.0x, lowpass_2000hz, lowpass_1000hz, lowpass_500hz, dropout_10pct, dropout_25pct, dropout_50pct.

## 9. Statistical rules

- patient-level resampling only; cycles of one patient are never treated as independent
- no significance fishing, no post-test model selection, no re-running with changed settings
- predeclared analyses and any later exploratory observations are reported in separate, labelled sections
- with 19 test patients intervals are wide; results are estimates with uncertainty, not significance claims
- no medical or diagnostic claims

## 10. Test-access audit

- **Passed: True.** 19 test patients, 169 test recordings.
- Report/model files mentioning a test recording: none; prediction CSVs with test patients: none (32 CSVs checked).
- Records with test_split_evaluated = true: none; non-validation split values in report JSON: none; datasets used by any training run: ['train', 'val'].
- Code calling evaluate(..., allow_test=True): none; test-like files ever committed: none.
- Checkpoint contains: model_state, config, epoch, val_metrics (validation only), class_names, data_sha256.

## 11. Freeze manifest

| File | SHA-256 | Git blob |
|---|---|---|
| `models/baseline_cnn_v3/best.pt` | `2c2c7c03ed00b656…` | `—` |
| `reports/runs/baseline_cnn_v3/run.json` | `54f73f6671ec9573…` | `3866b84d8ad5` |
| `reports/runs/baseline_cnn_v3/history.json` | `d182079ffd9a81b3…` | `6c31340d914f` |
| `reports/runs/baseline_cnn_v3/val_predictions.csv` | `3c6f74320e60a981…` | `c85290f5e260` |
| `reports/runs/baseline_cnn_v3/val_analysis.json` | `5532fafe2dc8e5a1…` | `fe051d6cc0f1` |
| `data/splits/patient_split.csv` | `a5f61582b9048af0…` | `8ff51c9d0a0f` |
| `data/metadata.csv` | `19b2f39ada720ee4…` | `—` |
| `data/normalization/logmel_primary_train_band_stats.json` | `9f455708587724e6…` | `9f6fda76405c` |
| `src/data/build_metadata.py` | `91795ab5c4219434…` | `28f05cfd6fa0` |
| `src/data/create_splits.py` | `9a06e4e64aab1cf7…` | `f7c20e784e6b` |
| `src/data/length_bucketing.py` | `103c5b92be1b730f…` | `d72d5ba65a83` |
| `src/data/normalization.py` | `a9c207d7e1258380…` | `a2a26b895fa9` |
| `src/data/respiratory_dataset.py` | `c882d30350f6f94d…` | `6dab3f4fdbe5` |
| `src/features/audio_preprocessing.py` | `89cee848876f3f36…` | `9adec108ba5f` |
| `src/features/logmel.py` | `7b7528ff5531579f…` | `0eb335e5213e` |
| `src/features/degradations.py` | `62d79e45503a8de4…` | `9da161159ad2` |
| `src/models/train.py` | `22a304a2dfa34912…` | `54496cc7a859` |
| `src/models/baseline_cnn.py` | `5a22836200d3dc93…` | `f4b41460c47b` |
| `src/models/crnn.py` | `90a0d4d412c277b3…` | `43b354579094` |
| `src/models/mobilenet_transfer.py` | `d83211505644ca43…` | `2ec40cebe553` |
| `src/evaluation/analyze_run.py` | `b39e6d8739bdc60b…` | `a3c9c691005b` |
| `src/evaluation/classification_metrics.py` | `3e61698af1c76249…` | `7dcb42b0b9e7` |
| `src/evaluation/compare_runs.py` | `5c2254a6660d2e98…` | `ab641f2e2950` |
| `src/evaluation/calibration_metrics.py` | `49aa6f7c3743d53b…` | `7466443aa246` |
| `src/evaluation/calibration_uncertainty.py` | `5da74111cea23679…` | `4557118575df` |
| `src/evaluation/calibration_report.py` | `0888c22ffe105a80…` | `358fe8af6882` |
| `src/evaluation/noise_robustness.py` | `d988e7521ff0e9f1…` | `de6c501d3664` |
| `src/evaluation/final_protocol.py` | `093ea5c8ceb2206a…` | `42e397ad3b3f` |
| `src/evaluation/final_evaluation.py` | `b0ccf19b034acc16…` | `fecf79c25c46` |
| `tests/test_final_evaluation.py` | `a1518dd2ed19a703…` | `0ebe2a73db13` |
| `tests/test_final_protocol.py` | `b62d098caf9b335d…` | `8aafe5c2de48` |
| `reports/runs/seed_study/plan.json` | `4d204797a0652b35…` | `081d66336f5f` |
| `reports/runs/seed_study/summary.json` | `4657844d6005dbfb…` | `12a6756b2ef8` |
| `reports/robustness/noise_robustness/plan.json` | `d1af254c573522de…` | `237b33952dde` |
| `reports/robustness/noise_robustness/summary.json` | `2afe16248efbcb24…` | `a7ffe6e6d0c8` |
| `reports/robustness/noise_robustness/clean_check.json` | `7c31811e6c071a40…` | `ad3d015ee709` |
| `reports/calibration_uncertainty/plan.json` | `e08de52de6b62a14…` | `7e7821b22676` |
| `reports/calibration_uncertainty/summary.json` | `217b014d4bb57d2e…` | `1c575d330746` |
| `reports/calibration_uncertainty/lopo_temperatures.csv` | `5a3f70e4c6905f42…` | `64d5ce8edaae` |
| `README.md` | `9e048bbda6787d17…` | `cd3b4010e887` |
| `data/raw/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files` (920 .wav, combined) | `44b143939ad86272…` | — |

Full hashes are in `protocol.json`. Git blob ids are stable across LF/CRLF checkouts; raw SHA-256 values describe this working copy.

## 12. Unlock procedure

1. Commit this protocol, src/evaluation/final_evaluation.py and its tests; the working tree must be clean (the test stages refuse to run otherwise) and the commit is recorded in every test output.
2. Run `python src/evaluation/final_protocol.py --verify`; abort if any content identity differs.
3. Run `python src/evaluation/final_evaluation.py dry-run-validation` (VALIDATION only): it must reproduce reports/runs/baseline_cnn_v3/val_predictions.csv and run.json's validation metrics exactly.
4. Unlock once: `python src/evaluation/final_evaluation.py clean-test --unlock-test UNLOCK-ONE-TIME-TEST-EVALUATION-baseline_cnn_v3`. It verifies everything again, writes a STARTED marker, loads the test split once, saves per-cycle logits/probabilities/predictions, computes sections 3-7, and writes clean_test/lock.json with the SHA-256 of every clean output.
5. Only then: `python src/evaluation/final_evaluation.py robustness-test --unlock-test UNLOCK-ONE-TIME-TEST-EVALUATION-baseline_cnn_v3`. It refuses unless the clean lock verifies, first reproduces the locked clean predictions through its own clean condition, then runs the 15 frozen conditions with the frozen T and thresholds (section 8).
6. Write the final report with predeclared results first and any exploratory observations separately. Any bug found after unlock is documented; original outputs are preserved, never silently replaced, and no stage is re-run because a result looks surprising (STARTED markers block re-runs).

## 13. Final evaluator

- Script `src/evaluation/final_evaluation.py`; no stage runs without an explicit subcommand; test stages additionally require the exact unlock token and a clean Git working tree; validation is the only split the dry run can read.
- The temperature, thresholds, coverages, conditions, metrics, checkpoint and bootstrap settings come only from this protocol; the CLI has no flags for them; nothing is trained, fitted or selected on test data.
- `verify`: Stage A: protocol manifest, checkpoint/data hashes, frozen values, provenance
- `dry-run-validation`: Stage B: exact clean inference path on VALIDATION; must reproduce the frozen reference
- `clean-test`: Stages A + C + D: one-time clean test evaluation and its lock
- `robustness-test`: Stage E: fixed degradations on test, only after the clean lock verifies
- Unlock token: `UNLOCK-ONE-TIME-TEST-EVALUATION-baseline_cnn_v3`.

Version 3; supersedes `protocol_v2_superseded.json` (version 2, created 2026-10-24T18:41:49-04:00, SHA-256 `a593971682aa01bd1ed9197a05105f27954b06e24bcf6170529fc347b5887d55`), revised before any test stage started. Changes: final_evaluation.py console messages made ASCII-only (the v2 dry run reproduced the reference exactly but its final print crashed on a cp1252 console); added a cp1252 CLI regression test. No frozen value, metric, threshold, temperature, condition or data artifact changed.

Environment: Python 3.11.9, torch 2.11.0+cu130, torchaudio 2.11.0+cu130, numpy 2.4.6, CUDA build 13.0, NVIDIA GeForce RTX 4050 Laptop GPU; full package list in `reports/final_evaluation/environment_pip_freeze.txt`.
