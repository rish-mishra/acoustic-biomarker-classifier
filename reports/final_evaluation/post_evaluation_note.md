# Post-evaluation maintenance note

This note records what changed in the repository after the one-time final test evaluation, and what did not.

## Timeline

1. **Protocol v3 was frozen before any test access** ([`protocol.md`](protocol.md), [`protocol.json`](protocol.json), SHA-256 `05f7fd0048f49d9a0d018319142fc6b07397b46dff97995e1bcda07eeb524f65`). Versions 1 and 2 are kept as `protocol_v*_superseded.*`; both were revised before any test stage started.
2. **Commit `9d36c12` is the frozen pre-test code state.** It contains protocol v3, the final evaluator and its tests. The clean test ran from a clean working tree at this commit.
3. **The final evaluations were run once and locked:**
   - **Clean test:** [`clean_test/`](clean_test/), locked 2026-10-24T19:34:00-04:00, committed in `ee3930e`.
   - **Fixed robustness test:** [`robustness_test/`](robustness_test/), locked 2026-10-24T19:49:40-04:00, committed in `38722c4`. It ran from commit `ee3930e`, and its clean condition reproduced the locked clean predictions bitwise.
4. **Later changes are documentation, reporting and test maintenance only.** None of them is an experiment.
   - Post-hoc exploratory analysis of the locked predictions, clearly labelled ([`exploratory/`](exploratory/), `6c0aa61`).
   - The rewritten README, the final report and its figures.
   - This note and [`checkpoint_release.md`](checkpoint_release.md).
   - `.gitattributes`, which pins the line endings of every file whose raw SHA-256 is checked. A fresh clone on any platform therefore reproduces the exact locked bytes. No file content changed.
   - An update to the obsolete pre-unlock audit test (below).

## What was not changed

- **Frozen artifacts.** No frozen prediction, metric, checkpoint, split, metadata, normalization statistic, preprocessing or model code, calibration temperature, threshold, degradation definition or evaluation output was changed.
- **Locks.** `clean_test/lock.json` and `robustness_test/lock.json` both still verify.
- **No new work on test.** No new test inference, retraining, model selection, calibration fitting or threshold fitting was performed.
- **Protocol v3 itself is untouched.** Its manifest was not regenerated or edited.

## The audit test update

`tests/test_final_protocol.py::TestAuditAndProtocol::test_audit_passes` was written for the pre-unlock state. It required that no test predictions exist, that no code calls the unlock path, and that no test artifacts are committed. Since the sanctioned one-time evaluation, those artifacts intentionally exist, so the test was updated to handle both states:
- **Pre-unlock** (no `clean_test/` or `robustness_test/`): the original requirement, unchanged.
- **Post-unlock:**
  - every finding must be a sanctioned output, meaning a lock-listed prediction file, a labelled post-hoc exploratory derivative of one, or the frozen evaluator itself;
  - both locks must verify and be linked;
  - the results must carry the frozen protocol's hash, checkpoint, temperature, thresholds, coverages, bootstrap settings and conditions;
  - no training run may have used the test split;
  - nothing frozen may differ from protocol v3 except the files listed below.

  This post-unlock check lives in `TestPostUnlockAudit`. It allowlists explicit paths only, never patterns that would admit arbitrary test files.

A later update (`v1.0.1`) changed two tests, with no effect on any result:
- **Report figures.** The post-unlock allowlist was extended, by exact path, to three final-report figures derived from locked outputs: `risk_coverage_validation_vs_test.svg`, `robustness_validation_vs_test.svg` and `test_confusion_matrix.svg`. A test re-renders each one in memory from the locked outputs and requires the file to match.
- **Line endings.** The line-ending regression test was made CRLF-safe: it normalises to LF before building its CRLF copy.

## Verifying against protocol v3

`python src/evaluation/final_protocol.py --verify` compares the *current* working tree with the protocol-v3 manifest. At later documentation commits it reports these expected differences, all of them documentation or test maintenance:
- `README.md`
- `tests/test_final_protocol.py`

Every inference, data, checkpoint, calibration and plan artifact still verifies.

For exact historical verification, check out the frozen commit, where every manifest entry matches:

```bash
git checkout 9d36c12
python src/evaluation/final_protocol.py --verify
```
