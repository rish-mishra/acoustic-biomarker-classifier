"""Tests for src/evaluation/final_evaluation.py.

Everything runs on a synthetic project (synthetic audio, synthetic train/val/TEST patients, a tiny CPU model,
a synthetic protocol and manifest). The real test split is never touched; the only real-repository checks
are that the default CLI and wrong tokens refuse before doing anything.
"""

import dataclasses
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("evaluation", "models", "data", "features"):
    sys.path.insert(0, str(REPO_ROOT / "src" / sub))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_run as ar  # noqa: E402
import calibration_metrics as cmx  # noqa: E402
import compare_runs as cr  # noqa: E402
import degradations as dg  # noqa: E402
import final_evaluation as fe  # noqa: E402
import final_protocol as fp  # noqa: E402
import normalization as nz  # noqa: E402
import train as tr  # noqa: E402
from test_respiratory_dataset import SyntheticICBHI  # noqa: E402

TOKEN = fe.UNLOCK_TOKEN
T_FROZEN = 1.7
TAUS = {"tau_80": 0.40, "tau_50": 0.55}


def build_project(root: Path) -> fe.Context:
    data = SyntheticICBHI(root / "data")
    norm = nz.compute_band_stats(data.dataset("train"))
    stats = root / "data" / "stats.json"
    nz.save_normalizer(norm, stats, data.metadata, data.split_file)
    cfg = tr.TrainConfig(run_name="m", max_epochs=2, batch_size=4, device="cpu", checkpoint_dir=str(root / "models"),
                         reports_dir=str(root / "reports"), bucket_pool_multiplier=2)
    tr.fit(cfg, data.dataset("train", normalizer=norm), data.dataset("val", normalizer=norm), log=lambda *_: None,
           normalization_path=stats)
    ar.analyze("m", root / "models", root / "reports", dataset=data.dataset("val", normalizer=norm), device="cpu")
    cal = root / "reports" / "cal_summary.json"
    cal.write_text(json.dumps({"temperature_scaling": {"final_validation_temperature": {"temperature": T_FROZEN}},
                               "part5_degradation": {"fixed_clean_thresholds": TAUS}}))
    files = ["models/m/best.pt", "data/metadata.csv", "data/patient_split.csv", "data/stats.json",
             "reports/m/val_predictions.csv", "reports/m/run.json", "reports/cal_summary.json"]
    protocol = {
        "version": 1,
        "1_frozen_model": {"checkpoint_sha256": fp.sha256(root / "models/m/best.pt")},
        "2_frozen_data_pipeline": {
            "patient_split": {"sha256": fp.sha256(data.split_file), "patients": {"test": 2}},
            "metadata": {"sha256": fp.sha256(data.metadata), "test_rows": 8},
            "normalization": {"sha256": fp.sha256(stats)}},
        "3_primary_endpoint": {"confidence_intervals": {"n_resamples": 40, "seed": 0}},
        "5_calibration": {"frozen_temperature": T_FROZEN},
        "7_selective_prediction": {"coverage_levels": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                                   "frozen_validation_thresholds": TAUS},
        "8_robustness": {"conditions": [c.as_dict() for c in dg.PREDECLARED_CONDITIONS]},
        "11_freeze_manifest": fp.build_manifest(files, audio_dir=None, root=root),
    }
    (root / "protocol.json").write_text(json.dumps(protocol))
    return fe.Context(root=root, protocol_path=root / "protocol.json", out_dir=root / "out",
                      checkpoint_dir=root / "models", model_run="m", metadata=data.metadata,
                      split_file=data.split_file, audio_dir=data.audio, normalization=stats,
                      reference_val_predictions=root / "reports/m/val_predictions.csv",
                      reference_run_json=root / "reports/m/run.json", calibration_summary=cal,
                      audio_manifest_dir=None, device="cpu", require_cuda=False, require_clean_git=False)


def frozen_hashes(ctx):
    paths = [ctx.checkpoint_dir / ctx.model_run / "best.pt", ctx.metadata, ctx.split_file, ctx.normalization,
             ctx.protocol_path, ctx.reference_val_predictions, ctx.reference_run_json, ctx.calibration_summary]
    return {str(p): fp.sha256(p) for p in paths}


class TestFinalEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ctx = build_project(Path(cls.tmp.name))
        cls.before = frozen_hashes(cls.ctx)
        with mock.patch.object(cmx, "fit_temperature", side_effect=AssertionError("fitting on test")), \
                mock.patch.object(cmx, "lopo_temperatures", side_effect=AssertionError("fitting on test")):
            cls.metrics = fe.clean_test(cls.ctx, TOKEN)
            cls.robust = fe.robustness_test(cls.ctx, TOKEN)
        cls.after = frozen_hashes(cls.ctx)
        cls.clean_rows = fe.read_rows(cls.ctx.out_dir / "clean_test" / "predictions.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fresh_out(self, name):
        return dataclasses.replace(self.ctx, out_dir=Path(self.tmp.name) / name)

    # ---- guards
    def test_default_cli_and_bad_tokens_refuse(self):
        p = fe.build_parser()
        for argv in ([], ["clean-test"], ["robustness-test"], ["clean-test", "--unlock-test", TOKEN, "--temperature", "2"],
                     ["dry-run-validation", "--split", "test"]):
            with self.assertRaises(SystemExit, msg=argv):
                with mock.patch("sys.stderr"):
                    p.parse_args(argv)
        real = REPO_ROOT / "reports" / "final_evaluation"
        existed = {d: (real / d).exists() for d in ("clean_test", "robustness_test")}
        with mock.patch("sys.stderr"):
            self.assertEqual(fe.main(["clean-test", "--unlock-test", "wrong"]), 1)
            self.assertEqual(fe.main(["robustness-test", "--unlock-test", ""]), 1)
        self.assertEqual(existed, {d: (real / d).exists() for d in existed})  # refused before touching anything
        with self.assertRaises(fe.TestAccessError):
            fe.check_token(None)

    def test_test_split_refused_without_unlock(self):
        with self.assertRaises(fe.TestAccessError):
            fe.make_dataset(self.ctx, "test")
        with self.assertRaises(fe.TestAccessError):
            fe.make_dataset(self.ctx, "test", condition=dg.PREDECLARED_CONDITIONS[1])
        with self.assertRaises(PermissionError):
            fe.make_dataset(self.ctx, "train")
        ds = fe.make_dataset(self.ctx, "test", allow_test=True)
        model, _ = fe.load_model(self.ctx, fe.device_for(self.ctx))
        with self.assertRaises(fe.TestAccessError):
            fe.infer(model, ds, "cpu")
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertFalse(model.training)

    def test_clean_git_required(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q", d], check=True)
            (Path(d) / "x.txt").write_text("x")
            with self.assertRaises(fe.FrozenArtifactError):
                fe.assert_clean_git(Path(d))
            for cmd in (["add", "x.txt"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"]):
                subprocess.run(["git", "-C", d, *cmd], check=True)
            self.assertEqual(len(fe.assert_clean_git(Path(d))), 40)
        ctx = dataclasses.replace(self.fresh_out("dirty"), require_clean_git=True, root=Path(self.tmp.name))
        with self.assertRaises(fe.FrozenArtifactError):  # the synthetic project is not a clean git tree
            fe.clean_test(ctx, TOKEN)
        self.assertFalse((ctx.out_dir / "clean_test").exists())

    # ---- Stage A
    def test_verify_detects_changed_artifacts(self):
        fe.verify(self.ctx)
        targets = [self.ctx.checkpoint_dir / "m" / "best.pt", self.ctx.split_file, self.ctx.normalization,
                   self.ctx.metadata, self.ctx.calibration_summary]
        for path in targets:
            original = path.read_bytes()
            try:
                path.write_bytes(original + b" ")
                with self.assertRaises(fe.FrozenArtifactError, msg=str(path)):
                    fe.verify(self.ctx)
            finally:
                path.write_bytes(original)
        fe.verify(self.ctx)

    def test_verify_detects_changed_protocol_values(self):
        path = self.ctx.protocol_path
        original = path.read_text()
        try:
            for mutate in (lambda p: p["5_calibration"].update(frozen_temperature=1.0),
                           lambda p: p["7_selective_prediction"]["frozen_validation_thresholds"].update(tau_50=0.6),
                           lambda p: p["8_robustness"]["conditions"].pop(),
                           lambda p: p["1_frozen_model"].update(checkpoint_sha256="0" * 64)):
                p = json.loads(original)
                mutate(p)
                path.write_text(json.dumps(p))
                with self.assertRaises(fe.FrozenArtifactError):
                    fe.verify(self.ctx)
        finally:
            path.write_text(original)

    # ---- Stage B
    def test_validation_dry_run_reproduces(self):
        ctx = self.fresh_out("dry")
        r = fe.dry_run_validation(ctx)
        self.assertTrue(r["ok"])
        self.assertEqual(r["predictions"]["max_abs_probability_difference"], 0.0)
        self.assertEqual(r["predictions"]["n_probability_values_bitwise_identical"], r["predictions"]["n_probability_values"])
        self.assertFalse(r["test_split_evaluated"])
        self.assertTrue((ctx.out_dir / "validation_dry_run" / "comparison.json").exists())

    def test_cli_output_survives_a_cp1252_console(self):
        out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        with mock.patch.object(fe, "Context", lambda: self.fresh_out("cli")), mock.patch("sys.stdout", out):
            self.assertEqual(fe.main(["verify"]), 0)
            self.assertEqual(fe.main(["dry-run-validation"]), 0)
        out.flush()
        self.assertIn(b"Stage B OK", out.buffer.getvalue())

    def test_compare_predictions_detects_differences(self):
        ref = fe.read_rows(self.ctx.reference_val_predictions)
        rows = [dict(r) for r in ref]
        self.assertTrue(fe.compare_predictions(ref, rows, "x")["ok"])
        rows[0]["pred"] = str((int(rows[0]["pred"]) + 1) % 4)
        self.assertFalse(fe.compare_predictions(ref, rows, "x")["ok"])
        self.assertFalse(fe.compare_predictions(ref, list(reversed(ref)), "x")["ok"])
        bumped = [dict(r) for r in ref]
        bumped[1]["prob_normal"] = str(float(bumped[1]["prob_normal"]) + 1e-3)
        out = fe.compare_predictions(ref, bumped, "x")
        self.assertFalse(out["ok"])
        self.assertAlmostEqual(out["max_abs_probability_difference"], 1e-3, places=9)

    # ---- Stage C/D
    def test_clean_outputs_and_frozen_artifacts_unchanged(self):
        d = self.ctx.out_dir / "clean_test"
        for name in (*fe.LOCKED_CLEAN_FILES, "lock.json", "STARTED.json"):
            self.assertTrue((d / name).exists(), name)
        self.assertEqual(self.before, self.after)
        self.assertEqual(self.metrics["n_cycles"], 8)
        self.assertEqual({r["patient_id"] for r in self.clean_rows}, {"105", "106"})  # synthetic test patients only
        for key in ("primary", "secondary", "quality_groups", "calibration", "error_detection",
                    "selective_prediction", "frozen_validation_thresholds"):
            self.assertIn(key, self.metrics)

    def test_frozen_temperature_applied_without_fitting(self):
        a = fe.arrays(self.clean_rows)
        expect = cmx.softmax(a["logits"] / T_FROZEN)
        got = np.array([[float(r[f"prob_frozen_t_{c}"]) for c in fe.CLASS_NAMES] for r in self.clean_rows])
        np.testing.assert_allclose(got, expect, atol=1e-12)
        self.assertEqual(self.metrics["calibration"]["frozen_temperature"], T_FROZEN)
        np.testing.assert_array_equal(got.argmax(1), a["preds"])
        self.assertEqual(self.metrics["calibration"]["raw"]["accuracy"], self.metrics["calibration"]["frozen_t"]["accuracy"])

    def test_frozen_thresholds_applied_exactly(self):
        a = fe.arrays(self.clean_rows)
        msp, wrong = cmx.confidence(cmx.softmax(a["logits"])), a["preds"] != a["labels"]
        th = self.metrics["frozen_validation_thresholds"]["raw"]
        for name, tau in TAUS.items():
            self.assertEqual(th[name]["tau"], tau)
            self.assertAlmostEqual(th[name]["realized_coverage"], float((msp >= tau).mean()))
            self.assertAlmostEqual(th[name]["confidently_wrong_share"], float((wrong & (msp >= tau)).mean()))

    def test_patient_level_bootstrap_matches_established_convention(self):
        preds = {"m": {(r["recording_filename"], int(r["cycle_index"])): r for r in self.clean_rows}}
        ref = cr.patient_bootstrap(preds, "m", n_boot=40, seed=0)
        self.assertEqual(self.metrics["primary"]["macro_f1_ci95"], ref["runs"]["m"]["macro_f1_ci95"])
        self.assertEqual(self.metrics["bootstrap"], {"unit": "patient", "n_resamples": 40, "seed": 0})

    def test_quality_groups(self):
        q = self.metrics["quality_groups"]
        self.assertEqual(set(q), {"all", "moderately_clean", "clean"})
        self.assertEqual(q["all"]["n_cycles"], 8)
        self.assertAlmostEqual(q["all"]["macro_f1"], self.metrics["primary"]["macro_f1"])

    def test_missing_class_subgroup_uses_zero_division_convention(self):
        a = {"labels": np.array([0, 0, 1, 1, 3, 3]), "preds": np.array([0, 1, 1, 1, 0, 3]),
             "patient": ["a", "a", "b", "b", "c", "c"],
             "quality": [{"original_clipped_fraction": 0.0, "clipped_plateau_count": 0}] * 4
             + [{"original_clipped_fraction": 0.5, "clipped_plateau_count": 3}] * 2}
        g = fe.quality_group_metrics(a, n_boot=20, seed=0)
        clean = g["clean"]
        self.assertEqual((clean["n_cycles"], clean["n_patients"]), (4, 2))
        self.assertEqual(clean["support"], [2, 2, 0, 0])
        self.assertEqual(clean["per_class_f1"][2:], [0.0, 0.0])  # absent classes kept with F1 0
        self.assertAlmostEqual(clean["macro_f1"], sum(clean["per_class_f1"]) / 4)
        self.assertAlmostEqual(clean["macro_f1_present_classes"], sum(clean["per_class_f1"][:2]) / 2)
        empty = fe.quality_group_metrics({**a, "quality": [{"original_clipped_fraction": 0.5,
                                                            "clipped_plateau_count": 3}] * 6}, 20, 0)
        self.assertEqual(empty["clean"], {"n_cycles": 0, "n_patients": 0})

    def test_lock_verifies_and_clean_results_never_overwritten(self):
        d = self.ctx.out_dir / "clean_test"
        lock = fe.verify_lock(d, self.ctx.protocol_path)
        self.assertEqual(set(lock["files"]), set(fe.LOCKED_CLEAN_FILES))
        before = {n: fp.sha256(d / n) for n in fe.LOCKED_CLEAN_FILES}
        with self.assertRaises(fe.ResultsLockedError):
            fe.clean_test(self.ctx, TOKEN)
        self.assertEqual(before, {n: fp.sha256(d / n) for n in fe.LOCKED_CLEAN_FILES})

    def test_started_marker_blocks_rerun_after_failure(self):
        ctx = self.fresh_out("crashed")
        (ctx.out_dir / "clean_test").mkdir(parents=True)
        (ctx.out_dir / "clean_test" / "STARTED.json").write_text("{}")
        with self.assertRaises(fe.ResultsLockedError):
            fe.clean_test(ctx, TOKEN)

    # ---- Stage E
    def test_robustness_refuses_without_clean_lock(self):
        ctx = self.fresh_out("nolock")
        with self.assertRaises(fe.ResultsLockedError):
            fe.robustness_test(ctx, TOKEN)
        self.assertFalse((ctx.out_dir / "robustness_test").exists())

    def test_robustness_refuses_tampered_lock_or_changed_protocol(self):
        ctx = self.fresh_out("tampered")
        shutil.copytree(self.ctx.out_dir / "clean_test", ctx.out_dir / "clean_test")
        p = ctx.out_dir / "clean_test" / "predictions.csv"
        fe.verify_lock(ctx.out_dir / "clean_test", ctx.protocol_path)  # intact copy verifies
        p.write_bytes(p.read_bytes() + b"\n")
        with self.assertRaises(fe.ResultsLockedError):
            fe.robustness_test(ctx, TOKEN)
        self.assertFalse((ctx.out_dir / "robustness_test").exists())

    def test_robustness_outputs_and_clean_reproduction(self):
        s = self.robust
        self.assertTrue(s["clean_reproduction"]["ok"])
        self.assertEqual(s["clean_reproduction"]["max_abs_probability_difference"], 0.0)
        self.assertEqual(list(s["conditions"]), [c.name for c in dg.PREDECLARED_CONDITIONS])
        self.assertEqual(s["frozen_temperature"], T_FROZEN)
        self.assertEqual(s["conditions"]["clean"]["macro_f1_change_abs"], 0.0)
        d = self.ctx.out_dir / "robustness_test"
        fe.verify_lock(d, self.ctx.protocol_path)
        with self.assertRaises(fe.ResultsLockedError):
            fe.robustness_test(self.ctx, TOKEN)

    def test_reproduction_check_raises_on_difference(self):
        rows = [dict(r) for r in self.clean_rows]
        fe.check_reproduction(self.clean_rows, rows)
        rows[2]["pred"] = str((int(rows[2]["pred"]) + 1) % 4)
        with self.assertRaises(fe.ReproductionError):
            fe.check_reproduction(self.clean_rows, rows)


if __name__ == "__main__":
    unittest.main()
