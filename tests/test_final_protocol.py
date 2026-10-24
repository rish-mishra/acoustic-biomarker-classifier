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
        self.assertTrue(self.audit["passed"], self.audit)
        self.assertEqual(self.audit["datasets_used_by_any_training_run"], ["train", "val"])
        self.assertEqual(self.audit["test_patients"], 19)

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


if __name__ == "__main__":
    unittest.main()
