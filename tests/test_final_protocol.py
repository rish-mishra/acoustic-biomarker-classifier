"""Tests for src/evaluation/final_protocol.py (no test-split access; reads frozen validation artifacts only)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))

import final_protocol as fp  # noqa: E402

SMALL = ["data/splits/patient_split.csv", "models/baseline_cnn_v3/best.pt", "src/models/train.py"]


class TestManifest(unittest.TestCase):
    def test_entries_and_verify(self):
        m = fp.build_manifest(SMALL, audio_dir=None)
        self.assertEqual(set(m["files"]), set(SMALL))
        for e in m["files"].values():
            self.assertEqual(len(e["sha256"]), 64)
        self.assertEqual(fp.verify_manifest(m, audio_dir=None), [])

    def test_verify_detects_change_and_missing(self):
        m = fp.build_manifest(SMALL, audio_dir=None)
        m["files"]["src/models/train.py"]["git_blob"] = "0" * 40  # tracked text: blob id is the identity
        m["files"]["no/such/file.txt"] = {"sha256": "0" * 64}
        self.assertEqual(sorted(fp.verify_manifest(m, audio_dir=None)), ["no/such/file.txt", "src/models/train.py"])

    def test_checkpoint_hash_matches_recorded_data_hashes(self):
        import torch
        ckpt = torch.load(REPO_ROOT / "models" / "baseline_cnn_v3" / "best.pt", map_location="cpu", weights_only=False)
        for key, rel in (("metadata", "data/metadata.csv"), ("split_file", "data/splits/patient_split.csv"),
                         ("normalization_stats", "data/normalization/logmel_primary_train_band_stats.json")):
            self.assertEqual(ckpt["data_sha256"][key], fp.sha256(REPO_ROOT / rel))

    def test_manifest_files_exist(self):
        for rel in fp.MANIFEST_FILES:
            self.assertTrue((REPO_ROOT / rel).exists(), rel)


class TestLineEndings(unittest.TestCase):
    """Fresh-clone / core.autocrlf behaviour of --verify."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "code.py").write_bytes(b"def f():\n    return 1\n")
        (self.root / "blob.bin").write_bytes(b"\x00\x01binary\r\n\x02")
        self.manifest = fp.build_manifest(["code.py", "blob.bin"], audio_dir=None, root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_identities(self):
        self.assertEqual(self.manifest["files"]["code.py"]["identity"], "git_blob")
        self.assertEqual(self.manifest["files"]["blob.bin"]["identity"], "sha256")
        self.assertIsNone(self.manifest["files"]["blob.bin"]["git_blob"])

    def test_crlf_checkout_of_same_content_verifies(self):
        (self.root / "code.py").write_bytes(b"def f():\r\n    return 1\r\n")
        self.assertNotEqual(fp.sha256(self.root / "code.py"), self.manifest["files"]["code.py"]["sha256"])
        self.assertEqual(fp.verify_manifest(self.manifest, audio_dir=None, root=self.root), [])

    def test_genuine_content_change_fails(self):
        for content in (b"def f():\n    return 2\n", b"def f():\r\n    return 2\r\n", b"def f():\n    return 1\n\n"):
            (self.root / "code.py").write_bytes(content)
            self.assertEqual(fp.verify_manifest(self.manifest, audio_dir=None, root=self.root), ["code.py"], content)

    def test_binary_and_external_use_exact_sha256(self):
        (self.root / "blob.bin").write_bytes(b"\x00\x01binary\n\x02")  # CRLF -> LF in a binary file is a change
        self.assertEqual(fp.verify_manifest(self.manifest, audio_dir=None, root=self.root), ["blob.bin"])

    def test_v1_entries_without_identity_use_sha256(self):
        m = {"files": {"code.py": {"sha256": self.manifest["files"]["code.py"]["sha256"]}}}
        self.assertEqual(fp.verify_manifest(m, audio_dir=None, root=self.root), [])
        (self.root / "code.py").write_bytes(b"def f():\r\n    return 1\r\n")
        self.assertEqual(fp.verify_manifest(m, audio_dir=None, root=self.root), ["code.py"])

    def test_canonical_blob_matches_git_hash_object(self):
        for rel in ("README.md", "src/models/train.py"):
            data = (REPO_ROOT / rel).read_bytes()
            self.assertEqual(fp.canonical_blob_id(data), fp.git("hash-object", rel))
            self.assertEqual(fp.canonical_blob_id(data.replace(b"\n", b"\r\n")), fp.git("hash-object", rel))

    def test_repo_classification(self):
        self.assertTrue(fp.is_external("data/metadata.csv"))
        self.assertTrue(fp.is_external("models/baseline_cnn_v3/best.pt"))
        self.assertFalse(fp.is_external("src/models/train.py"))
        self.assertEqual(fp.file_entry("data/metadata.csv")["identity"], "sha256")
        self.assertEqual(fp.file_entry("src/models/train.py")["identity"], "git_blob")


class TestSupersede(unittest.TestCase):
    def test_rules(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "protocol.json").write_text(json.dumps({"version": 1, "created": "t0"}))
            (out / "protocol.md").write_text("v1")
            (out / "clean_test").mkdir()
            with self.assertRaises(PermissionError):
                fp.supersede(out, "x")
            self.assertTrue((out / "protocol.json").exists())
            (out / "clean_test").rmdir()
            info = fp.supersede(out, "added evaluator")
            self.assertEqual((info["version"], info["file"], info["changes"]), (1, "protocol_v1_superseded.json",
                                                                               "added evaluator"))
            self.assertFalse((out / "protocol.json").exists())
            self.assertTrue((out / "protocol_v1_superseded.json").exists() and (out / "protocol_v1_superseded.md").exists())
            self.assertEqual(info["sha256"], fp.sha256(out / "protocol_v1_superseded.json"))


class TestAuditAndProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = fp.test_access_audit()
        cls.protocol = fp.build_protocol({**cls.audit, "test_cycles": 0}, fp.build_manifest(fp.MANIFEST_FILES, None),
                                         {k: "x" for k in ("python", "platform", "torch", "torchaudio", "numpy",
                                                           "soundfile", "cuda_build", "gpu", "pip_freeze")})

    def test_audit_passes(self):
        """Test-access audit in both states.

        Pre-unlock (no clean_test/ or robustness_test/): the original requirement, i.e. the audit passes with no
        test predictions, no callers of the test-unlock path and no committed test artifacts.
        Post-unlock: every audit finding must be a sanctioned, lock-verified final-evaluation output or the frozen
        evaluator itself; see TestPostUnlockAudit and reports/final_evaluation/post_evaluation_note.md.
        """
        self.assertEqual(self.audit["datasets_used_by_any_training_run"], ["train", "val"])
        self.assertEqual(self.audit["test_patients"], 19)
        # findings that must stay empty in BOTH states (test mentions / flags / splits outside final_evaluation/)
        for key in ("report_or_model_files_mentioning_a_test_recording", "json_records_with_test_split_evaluated_true",
                    "non_validation_split_values_in_report_json"):
            self.assertEqual(self.audit[key], [], key)
        if not any((FINAL / d).exists() for d in ("clean_test", "robustness_test")):
            self.assertTrue(self.audit["passed"], self.audit)  # pre-unlock: original requirement, unchanged
        else:
            post = PostUnlock()
            self.assertEqual(sorted(norm(p) for p in self.audit["prediction_csvs_containing_test_patients"]),
                             sorted(post.sanctioned_prediction_csvs))
            for line in self.audit["code_calling_evaluate_with_allow_test_true"]:
                self.assertIn(line.split(":", 1)[0], ALLOW_TEST_FILES, line)
            for path in self.audit["test_like_files_ever_committed"]:
                self.assertTrue(path.startswith(SANCTIONED_TEST_DIRS) or path in TEST_NAMED_NON_OUTPUTS, path)

    def test_frozen_values_come_from_artifacts(self):
        cal = json.loads((REPO_ROOT / "reports/calibration_uncertainty/summary.json").read_text(encoding="utf-8"))
        p = self.protocol
        self.assertEqual(p["5_calibration"]["frozen_temperature"],
                         cal["temperature_scaling"]["final_validation_temperature"]["temperature"])
        th = p["7_selective_prediction"]["frozen_validation_thresholds"]
        self.assertEqual((th["tau_80"], th["tau_50"]), (cal["part5_degradation"]["fixed_clean_thresholds"]["tau_80"],
                                                        cal["part5_degradation"]["fixed_clean_thresholds"]["tau_50"]))
        self.assertEqual(p["7_selective_prediction"]["coverage_levels"], [1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        rob = json.loads((REPO_ROOT / "reports/robustness/noise_robustness/plan.json").read_text(encoding="utf-8"))
        self.assertEqual(p["8_robustness"]["conditions"], rob["conditions"])
        self.assertEqual(len(p["8_robustness"]["conditions"]), 15)
        self.assertEqual(p["1_frozen_model"]["checkpoint_sha256"], fp.sha256(REPO_ROOT / "models/baseline_cnn_v3/best.pt"))
        self.assertEqual((p["1_frozen_model"]["seed"], p["1_frozen_model"]["checkpoint_epoch"]), (42, 13))
        self.assertEqual(p["2_frozen_data_pipeline"]["label_mapping"], {"normal": 0, "crackle": 1, "wheeze": 2, "both": 3})
        self.assertEqual(p["3_primary_endpoint"]["confidence_intervals"]["n_resamples"], 2000)
        self.assertIn("never fitted or modified on test", p["5_calibration"]["application"])

    def test_markdown_renders(self):
        md = fp.render_markdown(self.protocol)
        for text in ("Primary metric: macro F1", "Frozen temperature T = ", "τ80 =", "Test-access audit",
                     "Freeze manifest", "Unlock procedure", "cannot change any predicted class"):
            self.assertIn(text, md)


# ---------------------------------------------------------------- post-unlock audit (sanctioned outputs only)

FINAL = REPO_ROOT / "reports" / "final_evaluation"
SANCTIONED_TEST_DIRS = ("reports/final_evaluation/clean_test/", "reports/final_evaluation/robustness_test/")
EXPLORATORY_DIR = "reports/final_evaluation/exploratory/"
EXPLORATORY_LABEL = "POST-HOC EXPLORATORY — NOT PREDECLARED"
# files whose path contains "test" but which are code, not test outputs (explicit list, never a pattern)
TEST_NAMED_NON_OUTPUTS = {"src/evaluation/exploratory_test_analysis.py"}
# the only files allowed to call the unlock path; all of them are frozen in protocol v3 (checked below)
ALLOW_TEST_FILES = {"src/evaluation/final_evaluation.py", "src/evaluation/final_protocol.py",
                    "tests/test_final_evaluation.py"}
# files allowed to differ from the protocol-v3 manifest: post-evaluation documentation / test maintenance only
POST_EVALUATION_MAINTENANCE = {"README.md", "tests/test_final_protocol.py"}


def norm(p) -> str:
    return str(p).replace("\\", "/")


def unsanctioned_test_outputs(root: Path, test_patients: set[str], sanctioned: list[str]) -> list[str]:
    """CSV files under <root>/reports containing test-patient rows that are neither a sanctioned (lock-listed)
    prediction file nor a labelled post-hoc exploratory derivative; also reports sanctioned files that are missing."""
    import csv
    found = set()
    for p in (Path(root) / "reports").rglob("*.csv"):
        with open(p, newline="", encoding="utf-8") as f:
            lines = [line for line in f if not line.startswith("#")]
        rows = list(csv.DictReader(lines))
        if rows and "patient_id" in rows[0] and {r["patient_id"] for r in rows} & test_patients:
            found.add(norm(p.relative_to(root)))
    bad = []
    for p in sorted(found):
        if p in sanctioned:
            continue
        labelled = p.startswith(EXPLORATORY_DIR) and \
            (Path(root) / p).read_text(encoding="utf-8").startswith(f"# {EXPLORATORY_LABEL}")
        if not labelled:
            bad.append(p)
    return bad + [f"missing: {p}" for p in sanctioned if p not in found]


class TestAuditRejectsViolations(unittest.TestCase):
    """The post-unlock audit must not admit arbitrary test files."""

    def test_scan_flags_unsanctioned_and_unlabelled_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ok = "reports/final_evaluation/clean_test/predictions.csv"
            for rel, header in ((ok, ""), ("reports/runs/x/val_predictions.csv", ""),
                                ("reports/final_evaluation/exploratory/a.csv", f"# {EXPLORATORY_LABEL}\n"),
                                ("reports/final_evaluation/exploratory/b.csv", ""),
                                ("reports/other/leak.csv", "")):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                pid = "999" if "val_predictions" in rel else "105"
                (root / rel).write_text(f"{header}patient_id,pred\n{pid},0\n", encoding="utf-8")
            self.assertEqual(unsanctioned_test_outputs(root, {"105"}, [ok]),
                             ["reports/final_evaluation/exploratory/b.csv", "reports/other/leak.csv"])
            self.assertEqual(unsanctioned_test_outputs(root, {"105"}, [ok, "reports/final_evaluation/robustness_test/"
                                                                          "conditions/clean/predictions.csv"])[-1],
                             "missing: reports/final_evaluation/robustness_test/conditions/clean/predictions.csv")

    def test_test_named_files_outside_sanctioned_dirs_are_rejected(self):
        for path in ("reports/test_predictions.csv", "reports/final_evaluation/test_rerun/metrics.json",
                     "src/evaluation/test_inference.py"):
            self.assertFalse(path.startswith(SANCTIONED_TEST_DIRS) or path in TEST_NAMED_NON_OUTPUTS, path)


class PostUnlock:
    def __init__(self):
        import final_evaluation as fe
        self.fe = fe
        self.protocol_path = FINAL / "protocol.json"
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.clean_lock = fe.verify_lock(FINAL / "clean_test", self.protocol_path)
        self.robust_lock = fe.verify_lock(FINAL / "robustness_test", self.protocol_path) \
            if (FINAL / "robustness_test").exists() else None
        self.sanctioned_prediction_csvs = sorted(
            [f"reports/final_evaluation/clean_test/{n}" for n in self.clean_lock["files"] if n.endswith("predictions.csv")]
            + ([f"reports/final_evaluation/robustness_test/{n}" for n in self.robust_lock["files"]
                if n.endswith("predictions.csv")] if self.robust_lock else []))


@unittest.skipUnless((FINAL / "clean_test").exists(), "pre-unlock state: covered by test_audit_passes")
class TestPostUnlockAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.post = PostUnlock()
        cls.protocol = cls.post.protocol
        cls.metrics = json.loads((FINAL / "clean_test" / "metrics.json").read_text(encoding="utf-8"))
        cls.robust = json.loads((FINAL / "robustness_test" / "summary.json").read_text(encoding="utf-8")) \
            if cls.post.robust_lock else None
        import csv
        with open(REPO_ROOT / "data" / "splits" / "patient_split.csv", newline="", encoding="utf-8") as f:
            cls.test_patients = {r["patient_id"] for r in csv.DictReader(f) if r["split"] == "test"}

    def test_locks_verify_and_are_linked(self):
        self.assertEqual(self.post.clean_lock["protocol_sha256"], fp.sha256(self.post.protocol_path))
        self.assertEqual(self.post.clean_lock["checkpoint_sha256"], self.protocol["1_frozen_model"]["checkpoint_sha256"])
        self.assertTrue((FINAL / "clean_test" / "STARTED.json").exists())
        if self.robust is not None:
            self.assertEqual(self.robust["clean_lock"], self.post.clean_lock)  # robustness ran on the locked clean result
            self.assertTrue(self.robust["clean_reproduction"]["ok"])
            self.assertEqual(self.robust["clean_reproduction"]["max_abs_probability_difference"], 0.0)
            self.assertTrue((FINAL / "robustness_test" / "STARTED.json").exists())

    def test_no_unsanctioned_test_outputs_anywhere(self):
        """Comment-aware scan of every output CSV under reports/ for test-patient rows (data/ holds only the
        dataset definition: metadata and the split itself)."""
        self.assertEqual(unsanctioned_test_outputs(REPO_ROOT, self.test_patients, self.post.sanctioned_prediction_csvs), [])
        ex = json.loads((FINAL / "exploratory" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((ex["label"], ex["source"]), (EXPLORATORY_LABEL,
                                                       "locked clean_test and robustness_test prediction files only"))

    def test_results_correspond_to_the_frozen_protocol(self):
        frozen_t = self.protocol["5_calibration"]["frozen_temperature"]
        th = self.protocol["7_selective_prediction"]["frozen_validation_thresholds"]
        ci = self.protocol["3_primary_endpoint"]["confidence_intervals"]
        prov = self.metrics["provenance"]
        self.assertEqual((prov["protocol_sha256"], prov["protocol_version"]), (fp.sha256(self.post.protocol_path), 3))
        self.assertEqual(prov["checkpoint_sha256"], self.protocol["1_frozen_model"]["checkpoint_sha256"])
        self.assertEqual(prov["git_status"], [])  # evaluated from a clean, committed tree
        self.assertEqual(self.metrics["calibration"]["frozen_temperature"], frozen_t)
        self.assertEqual(self.metrics["frozen_validation_thresholds"]["taus"], {"tau_80": th["tau_80"], "tau_50": th["tau_50"]})
        self.assertEqual(self.metrics["selective_prediction"]["coverage_levels"],
                         self.protocol["7_selective_prediction"]["coverage_levels"])
        self.assertEqual(self.metrics["bootstrap"], {"unit": "patient", "n_resamples": ci["n_resamples"], "seed": ci["seed"]})
        self.assertEqual(self.metrics["n_patients"], self.protocol["2_frozen_data_pipeline"]["patient_split"]["patients"]["test"])
        self.assertEqual(self.metrics["n_cycles"], self.protocol["2_frozen_data_pipeline"]["metadata"]["test_rows"])
        if self.robust is not None:
            self.assertEqual(self.robust["frozen_temperature"], frozen_t)
            self.assertEqual(self.robust["frozen_thresholds"], {"tau_80": th["tau_80"], "tau_50": th["tau_50"]})
            self.assertEqual(list(self.robust["conditions"]), [c["name"] for c in self.protocol["8_robustness"]["conditions"]])

    def test_no_training_selection_or_fitting_on_test(self):
        cal = json.loads((REPO_ROOT / "reports/calibration_uncertainty/summary.json").read_text(encoding="utf-8"))
        # temperature and thresholds are the validation-fitted values, not refitted
        self.assertEqual(self.protocol["5_calibration"]["frozen_temperature"],
                         cal["temperature_scaling"]["final_validation_temperature"]["temperature"])
        th = self.protocol["7_selective_prediction"]["frozen_validation_thresholds"]
        self.assertEqual({"tau_80": th["tau_80"], "tau_50": th["tau_50"]}, cal["part5_degradation"]["fixed_clean_thresholds"])
        # the evaluated checkpoint is the frozen one; no training run ever used the test split
        self.assertEqual(fp.sha256(REPO_ROOT / "models/baseline_cnn_v3/best.pt"), self.protocol["1_frozen_model"]["checkpoint_sha256"])
        for p in (REPO_ROOT / "reports").rglob("run.json"):
            run = json.loads(p.read_text(encoding="utf-8"))
            self.assertFalse(run.get("test_split_evaluated"), p)
            self.assertLessEqual(set(run.get("data", {}).get("datasets", {})), {"train", "val"}, p)

    def test_frozen_artifacts_still_match_protocol_v3(self):
        """Everything frozen still verifies; only documented post-evaluation maintenance files may differ."""
        changed = set(fp.verify_manifest(self.protocol["11_freeze_manifest"], audio_dir=None))
        self.assertLessEqual(changed, POST_EVALUATION_MAINTENANCE)
        for rel in ALLOW_TEST_FILES - POST_EVALUATION_MAINTENANCE:  # the unlock path is the frozen evaluator
            self.assertNotIn(rel, changed)
        note = (FINAL / "post_evaluation_note.md").read_text(encoding="utf-8")
        for rel in POST_EVALUATION_MAINTENANCE:
            self.assertIn(f"`{rel}`", note)


if __name__ == "__main__":
    unittest.main()
