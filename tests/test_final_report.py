"""Checks that the README and final report quote the stored artifacts correctly and link to files that exist."""

import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
REPORT = REPO_ROOT / "reports" / "final_report" / "final_report.md"
NOTE = REPO_ROOT / "reports" / "final_evaluation" / "post_evaluation_note.md"
RECORD = REPO_ROOT / "reports" / "final_evaluation" / "checkpoint_release.md"
FROZEN_COMMIT = "9d36c12"
MAINTENANCE = {"README.md", "tests/test_final_protocol.py"}  # documented in post_evaluation_note.md


def J(rel):
    return json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))


class TestFinalDocuments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = README.read_text(encoding="utf-8")
        cls.report = REPORT.read_text(encoding="utf-8")
        cls.test = J("reports/final_evaluation/clean_test/metrics.json")
        cls.rob = J("reports/final_evaluation/robustness_test/summary.json")["conditions"]
        cls.val_rob = J("reports/robustness/noise_robustness/summary.json")["conditions"]
        cls.cal = J("reports/calibration_uncertainty/summary.json")
        cls.seed = J("reports/runs/seed_study/summary.json")["per_architecture"]

    def both(self, text):
        for name, doc in (("README", self.readme), ("final report", self.report)):
            self.assertTrue(text in doc, f"{name} should contain {text!r}")

    def test_primary_and_secondary_test_numbers(self):
        p, s = self.test["primary"], self.test["secondary"]
        self.both(f"{p['macro_f1']:.4f}")
        lo, hi = p["macro_f1_ci95"]
        self.both(f"{lo:.3f}–{hi:.3f}")
        self.both(f"{s['balanced_accuracy']:.4f}")
        self.both(f"{s['accuracy']:.4f}")
        self.both(f"{s['per_class']['both']['f1']:.3f}")
        run = J("reports/runs/baseline_cnn_v3/run.json")["best_validation"]
        self.both(f"{run['macro_f1']:.4f}")
        self.both(f"{self.test['quality_groups']['clean']['macro_f1']:.3f}")

    def test_seed_study_numbers(self):
        for arch in self.seed.values():
            m = arch["macro_f1"]
            self.both(f"{m['mean']:.3f} ± {m['sd']:.3f}")
            self.assertIn(" / ".join(f"{v:.4f}" for v in m["values"]), self.readme)
            self.assertIn(" | ".join(f"{v:.4f}" for v in m["values"]), self.report)

    def test_robustness_numbers(self):
        for name in ("noise_snr0db", "lowpass_1000hz", "lowpass_500hz", "dropout_50pct"):
            self.both(f"{self.rob[name]['macro_f1']:.3f}")
        for name in ("noise_snr0db", "lowpass_500hz", "dropout_50pct"):
            self.both(f"{self.val_rob[name]['macro_f1']:.3f}")
        self.both(f"{self.val_rob['lowpass_1000hz']['macro_f1']:.3f}")

    def test_calibration_and_uncertainty_numbers(self):
        c = self.test["calibration"]
        self.assertIn(f"{c['raw']['ece']:.3f} → {c['frozen_t']['ece']:.3f}", self.readme)
        self.assertIn(f"ECE {c['raw']['ece']:.3f}", self.report)
        self.assertIn(f"ECE {c['frozen_t']['ece']:.3f}", self.report)
        t = self.cal["temperature_scaling"]["final_validation_temperature"]["temperature"]
        self.assertIn(f"T = {t:.3f}", self.readme)
        self.assertIn(f"T = {t:.4f}", self.report)
        self.assertIn(repr(t), self.report)
        ed = self.test["error_detection"]["raw"]
        self.both(f"{ed['1-msp']['auroc']:.3f}")
        self.both(f"{ed['entropy']['auroc']:.3f}")
        vr, vl = self.cal["part1_calibration_raw"], self.cal["part2_calibration_lopo"]
        self.both(f"{vr['ece']:.3f}")
        self.both(f"{vl['ece']:.3f}")
        lv = self.test["selective_prediction"]["rankings"]["msp_raw"]["levels"]
        self.assertIn(f"{lv[0]['selective_risk']:.3f} → {lv[-1]['selective_risk']:.3f}", self.readme)
        self.assertIn(f"from {lv[0]['selective_risk']:.3f} at full coverage to {lv[-1]['selective_risk']:.3f}", self.report)
        vl = self.cal["part4_selective_prediction"]["rankings"]["msp_raw"]["levels"]
        self.assertIn(f"{vl[0]['selective_risk']:.3f} → {vl[-1]['selective_risk']:.3f}", self.readme)
        self.assertIn(f"from {vl[0]['selective_risk']:.3f} at full coverage to {vl[-1]['selective_risk']:.3f}", self.report)

    def test_robustness_tables_exact(self):
        f = lambda names, src: " / ".join(f"{src[n]['macro_f1']:.3f}" for n in names)  # noqa: E731
        for names in (["noise_snr20db", "noise_snr10db", "noise_snr5db", "noise_snr0db"],
                      ["gain_0.5x", "gain_0.25x", "gain_1.5x", "gain_2.0x"],
                      ["lowpass_2000hz", "lowpass_1000hz", "lowpass_500hz"],
                      ["dropout_10pct", "dropout_25pct", "dropout_50pct"]):
            row = f(names, self.rob)
            self.assertTrue(row.replace(f"{self.rob[names[-1]]['macro_f1']:.3f}", f"**{self.rob[names[-1]]['macro_f1']:.3f}**")
                            in self.report or row in self.report
                            or any(row.replace(x, f"**{x}**") in self.report
                                   for x in (f"{self.rob[n]['macro_f1']:.3f}" for n in names)), row)
        self.assertIn(f(["gain_0.25x", "gain_2.0x"], self.val_rob), self.readme)
        self.assertIn(f(["gain_0.25x", "gain_2.0x"], self.rob), self.readme)

    def test_exploratory_numbers_and_labels(self):
        ex = J("reports/final_evaluation/exploratory/summary.json")["concentration"]
        akgc = next(r for r in ex["devices"]["table"] if r["unit"] == "AKGC417L")
        self.both(f"{akgc['count']} of {ex['patients']['total']}")
        self.assertIn("POST-HOC EXPLORATORY — NOT PREDECLARED", self.report)
        for doc in (self.readme, self.report):
            self.assertIn("confounded", doc)
            self.assertRegex(doc, r"not\*?\*? (a )?medical device")

    def test_no_stale_status_statements(self):
        """Current documentation only; frozen protocol files are historical and intentionally unchanged."""
        docs = [README, REPORT, NOTE, RECORD]
        docs += [p for p in (REPO_ROOT / "reports").rglob("*.md")
                 if "protocol" not in p.name and p.parent.name not in ("runs",)]
        for path in docs:
            text = path.read_text(encoding="utf-8")
            for stale in ("TBD", "No models have been trained", "not yet evaluated", "Methodology (Planned)",
                          "| Planned |", "yet to be written", "will be added as components", "early-stage"):
                self.assertNotIn(stale, text, f"{path.relative_to(REPO_ROOT)}: {stale!r}")

    def test_only_documented_maintenance_files_differ_from_the_frozen_manifest(self):
        import subprocess
        import sys
        sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))
        import final_protocol as fp
        manifest = J("reports/final_evaluation/protocol.json")["11_freeze_manifest"]
        self.assertLessEqual(set(fp.verify_manifest(manifest, audio_dir=None)), MAINTENANCE)
        for rel in MAINTENANCE:  # the frozen versions are preserved at the pre-test commit
            frozen = subprocess.run(["git", "show", f"{FROZEN_COMMIT}:{rel}"], cwd=REPO_ROOT, capture_output=True)
            if frozen.returncode == 0:
                self.assertEqual(fp.canonical_blob_id(frozen.stdout), manifest["files"][rel]["git_blob"], rel)

    def test_checkpoint_release_record(self):
        import hashlib
        rec = (REPO_ROOT / "reports/final_evaluation/checkpoint_release.md").read_text(encoding="utf-8")
        proto = J("reports/final_evaluation/protocol.json")
        sha = proto["1_frozen_model"]["checkpoint_sha256"]
        for text in (sha, "baseline_cnn_v3_seed42_best.pt", "models/baseline_cnn_v3/best.pt", "| Seed | 42 |",
                     f"| Best epoch | {proto['1_frozen_model']['checkpoint_epoch']} ", "version 3",
                     hashlib.sha256((REPO_ROOT / "reports/final_evaluation/protocol.json").read_bytes()).hexdigest()):
            self.assertIn(text, rec)
        ckpt = REPO_ROOT / "models/baseline_cnn_v3/best.pt"
        if ckpt.exists():  # local copy present (it is git-ignored): record must describe it exactly
            self.assertEqual(hashlib.sha256(ckpt.read_bytes()).hexdigest(), sha)
            self.assertIn(f"| Size | {ckpt.stat().st_size} bytes |", rec)

    def test_hash_locked_files_have_pinned_line_endings(self):
        pins = {}
        for line in (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                path, attr = line.split()
                pins[path] = attr
        files = [p for p in (REPO_ROOT / "reports/final_evaluation").rglob("*") if p.is_file() and p.name != "runner.lock"]
        files += [REPO_ROOT / "data/splits/patient_split.csv",
                  REPO_ROOT / "data/normalization/logmel_primary_train_band_stats.json"]
        for p in files:
            rel = p.relative_to(REPO_ROOT).as_posix()
            self.assertIn(rel, pins, f"{rel} is not pinned in .gitattributes")
            self.assertEqual(pins[rel], "eol=crlf" if b"\r\n" in p.read_bytes() else "eol=lf", rel)

    def test_relative_links_resolve(self):
        for path, doc in ((README, self.readme), (REPORT, self.report), (NOTE, NOTE.read_text(encoding="utf-8")),
                          (RECORD, RECORD.read_text(encoding="utf-8"))):
            for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", doc):
                if target.startswith("http"):
                    continue
                self.assertTrue((path.parent / target).exists(), f"{path.name}: broken link {target}")


if __name__ == "__main__":
    unittest.main()
