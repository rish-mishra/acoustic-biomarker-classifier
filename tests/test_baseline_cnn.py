"""Tests for the baseline CNN, masking, metrics, length bucketing, and training utilities."""

import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from unittest import mock
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("models", "evaluation", "data"):
    sys.path.insert(0, str(REPO_ROOT / "src" / sub))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as tr  # noqa: E402
from baseline_cnn import BaselineCNN, MaskedGroupNorm, count_parameters, downsample_lengths  # noqa: E402
from classification_metrics import classification_report, confusion_matrix  # noqa: E402
from length_bucketing import LengthBucketBatchSampler, approximate_frame_lengths  # noqa: E402
from test_respiratory_dataset import SyntheticICBHI  # noqa: E402

import normalization as nz  # noqa: E402
import respiratory_dataset as rd  # noqa: E402


def padded_batch(lengths, n_mels=64, pad_fill=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = max(lengths)
    x = torch.full((len(lengths), 1, n_mels, T), pad_fill)
    singles = []
    for i, n in enumerate(lengths):
        s = torch.randn(1, 1, n_mels, n, generator=g)
        x[i, ..., :n] = s
        singles.append(s)
    mask = torch.arange(T).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    return x, mask, singles


class TestModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = BaselineCNN().eval()

    def test_output_shape_and_params(self):
        x, mask, _ = padded_batch([21, 100, 254])
        self.assertEqual(tuple(self.model(x, mask).shape), (3, 4))
        # convs 292,752 + masked GroupNorm affine 960 + classifier 1,028
        self.assertEqual(count_parameters(self.model), 294_740)

    def test_no_batchnorm(self):
        self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.model.modules()))
        self.assertTrue(any(isinstance(m, MaskedGroupNorm) for m in self.model.modules()))

    def test_output_independent_of_padding_and_batch(self):
        lengths = [21, 37, 50, 1, 2, 160]
        for pad_fill in (0.0, 123.0, float("nan")):
            with self.subTest(pad_fill=pad_fill):
                x, mask, singles = padded_batch(lengths, pad_fill=pad_fill)
                with torch.no_grad():
                    batched = self.model(x, mask)
                    for i, s in enumerate(singles):
                        alone = self.model(s, torch.ones(1, s.shape[-1], dtype=torch.bool))
                        self.assertTrue(torch.allclose(batched[i], alone[0], atol=1e-5), (pad_fill, i))
                self.assertTrue(torch.isfinite(batched).all())

    def test_padded_inputs_get_zero_gradient(self):
        x, mask, _ = padded_batch([30, 80])
        x.requires_grad_(True)
        self.model.train()
        self.model(x, mask).sum().backward()
        self.assertTrue(torch.all(x.grad[0, ..., 30:] == 0))
        self.assertTrue(x.grad[0, ..., :30].abs().sum() > 0)

    def test_masked_groupnorm_matches_groupnorm_without_padding(self):
        x = torch.randn(2, 16, 8, 20)
        ours, ref = MaskedGroupNorm(8, 16), nn.GroupNorm(8, 16)
        out = ours(x, torch.ones(2, 20, dtype=torch.bool))
        self.assertTrue(torch.allclose(out, ref(x), atol=1e-5))

    def test_masked_groupnorm_ignores_padding(self):
        x = torch.randn(1, 16, 8, 30)
        norm = MaskedGroupNorm(8, 16)
        mask = torch.arange(30).unsqueeze(0) < 20
        noisy = x.clone()
        noisy[..., 20:] = 1e4
        a, b = norm(x, mask), norm(noisy, mask)
        self.assertTrue(torch.allclose(a[..., :20], b[..., :20], atol=1e-5))
        self.assertTrue(torch.all(b[..., 20:] == 0))

    def test_downsample_lengths(self):
        self.assertEqual(downsample_lengths(torch.tensor([1, 2, 3, 21, 254])).tolist(), [1, 1, 2, 11, 127])

    def test_rejects_bad_inputs(self):
        x, mask, _ = padded_batch([10, 20])
        with self.assertRaises(ValueError):
            self.model(x[:, 0], mask)
        with self.assertRaises(ValueError):
            self.model(x, mask[:, :5])
        with self.assertRaises(ValueError):
            self.model(x, torch.zeros_like(mask))


class TestMetrics(unittest.TestCase):
    def test_confusion_matrix(self):
        cm = confusion_matrix(torch.tensor([0, 0, 1, 2, 3, 3]), torch.tensor([0, 1, 1, 2, 3, 0]), 4)
        self.assertEqual(cm.tolist(), [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]])

    def test_report(self):
        cm = confusion_matrix(torch.tensor([0, 0, 1, 2, 3, 3]), torch.tensor([0, 1, 1, 2, 3, 0]), 4)
        r = classification_report(cm, ("a", "b", "c", "d"))
        self.assertAlmostEqual(r["per_class"]["a"]["precision"], 0.5)
        self.assertAlmostEqual(r["per_class"]["a"]["recall"], 0.5)
        self.assertAlmostEqual(r["per_class"]["b"]["f1"], 2 * 0.5 * 1 / 1.5)
        self.assertAlmostEqual(r["per_class"]["c"]["f1"], 1.0)
        self.assertAlmostEqual(r["accuracy"], 4 / 6)
        self.assertAlmostEqual(r["balanced_accuracy"], (0.5 + 1 + 1 + 0.5) / 4)
        self.assertAlmostEqual(r["macro_f1"], (0.5 + 2 / 3 + 1 + 2 / 3) / 4)

    def test_absent_class_gives_zero_not_error(self):
        r = classification_report(confusion_matrix(torch.tensor([0, 0]), torch.tensor([0, 0]), 4), "abcd")
        self.assertEqual(r["per_class"]["b"], {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
        self.assertAlmostEqual(r["macro_f1"], 0.25)


class TestBucketing(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.lengths = torch.randint(20, 1600, (1000,), generator=g).tolist()

    def test_every_index_once_per_epoch(self):
        s = LengthBucketBatchSampler(self.lengths, 32, seed=1, pool_multiplier=5)
        for epoch in range(3):
            s.set_epoch(epoch)
            batches = list(s)
            self.assertEqual(sorted(i for b in batches for i in b), list(range(1000)))
            self.assertEqual(len(batches), len(s))
            self.assertTrue(all(1 <= len(b) <= 32 for b in batches))

    def test_reproducible_and_changes_per_epoch(self):
        a = LengthBucketBatchSampler(self.lengths, 32, seed=1)
        b = LengthBucketBatchSampler(self.lengths, 32, seed=1)
        self.assertEqual(a.batches(0), b.batches(0))
        self.assertNotEqual(a.batches(0), a.batches(1))
        self.assertNotEqual(a.batches(0), LengthBucketBatchSampler(self.lengths, 32, seed=2).batches(0))

    def test_reduces_padding(self):
        def padding(batches):
            cells = sum(len(b) * max(self.lengths[i] for i in b) for b in batches)
            return 1 - sum(self.lengths) / cells
        bucketed = padding(LengthBucketBatchSampler(self.lengths, 32, seed=0).batches(0))
        order = torch.randperm(1000, generator=torch.Generator().manual_seed(0)).tolist()
        plain = padding([order[i:i + 32] for i in range(0, 1000, 32)])
        self.assertLess(bucketed, plain / 3)

    def test_approximate_lengths(self):
        self.assertEqual(approximate_frame_lengths([{"cycle_duration_s": "0.2"}, {"cycle_duration_s": 2.537}]),
                         [21, 254])


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = SyntheticICBHI(Path(self.tmp.name) / "data")
        norm = nz.compute_band_stats(self.data.dataset("train"))
        self.stats_path = Path(self.tmp.name) / "stats.json"
        nz.save_normalizer(norm, self.stats_path, self.data.metadata, self.data.split_file)
        self.train = self.data.dataset("train", normalizer=norm)
        self.val = self.data.dataset("val", normalizer=norm)
        self.test = self.data.dataset("test", normalizer=norm)

    def tearDown(self):
        self.tmp.cleanup()

    def cfg(self, **kw):
        base = dict(run_name="t", max_epochs=2, patience=5, batch_size=4, device="cpu",
                    checkpoint_dir=str(Path(self.tmp.name) / "models"),
                    reports_dir=str(Path(self.tmp.name) / "reports"), bucket_pool_multiplier=2)
        base.update(kw)
        return tr.TrainConfig(**base)

    def fit(self, cfg, **kw):
        return tr.fit(cfg, self.train, self.val, log=lambda *_: None, normalization_path=self.stats_path, **kw)

    def test_class_weights_from_train_only(self):
        w = tr.class_weights_from_train(self.train)
        self.assertTrue(torch.allclose(w, torch.ones(4)))  # synthetic train is balanced (2 per class)
        for ds in (self.val, self.test):
            with self.assertRaises(ValueError):
                tr.class_weights_from_train(ds)

    def test_test_split_is_refused(self):
        model = BaselineCNN()
        loader = tr.make_eval_loader(self.test, 4)
        with self.assertRaises(PermissionError):
            tr.evaluate(model, loader, torch.device("cpu"))
        with self.assertRaises(ValueError):
            tr.fit(self.cfg(), self.train, self.test, log=lambda *_: None, normalization_path=self.stats_path)

    def test_fit_writes_checkpoint_and_is_deterministic(self):
        r1 = self.fit(self.cfg(run_name="a"))
        r2 = self.fit(self.cfg(run_name="b"))
        self.assertEqual(r1["checkpoint"], Path(self.tmp.name) / "models" / "a" / "best.pt")
        self.assertEqual(r1["report_dir"], Path(self.tmp.name) / "reports" / "a")
        self.assertTrue(r1["checkpoint"].exists())
        self.assertEqual(sorted(p.name for p in r1["report_dir"].iterdir()), ["history.json", "run.json"])
        self.assertEqual([h["train_loss"] for h in r1["history"]], [h["train_loss"] for h in r2["history"]])
        ckpt = torch.load(r1["checkpoint"], weights_only=False)
        self.assertEqual(ckpt["epoch"], r1["best"]["epoch"])
        self.assertIn("macro_f1", ckpt["val_metrics"])
        self.assertEqual(ckpt["data_sha256"]["split_file"], tr.sha256_file(self.data.split_file))

    def test_run_record_contents(self):
        r = self.fit(self.cfg(run_name="rec", seed=7))
        run = json.loads((r["report_dir"] / "run.json").read_text())
        history = json.loads((r["report_dir"] / "history.json").read_text())
        self.assertEqual((run["run_name"], run["seed"], run["status"]), ("rec", 7, "completed"))
        self.assertEqual(run["config"]["seed"], 7)
        self.assertEqual(run["config"]["lr"], 1e-3)
        self.assertEqual(run["model"]["parameter_count"], 294_740)
        self.assertEqual(set(run["class_weights"]), set(rd.CLASS_NAMES))
        self.assertFalse(run["test_split_evaluated"])
        self.assertEqual(run["stop_reason"], "max_epochs_reached (2)")
        self.assertEqual(run["epochs_completed"], 2)
        self.assertGreater(run["total_training_time_s"], 0)
        self.assertEqual(run["best_epoch"], r["best"]["epoch"])
        best = run["best_validation"]
        self.assertEqual(len(best["confusion_matrix"]), 4)
        for name in rd.CLASS_NAMES:
            self.assertEqual(set(best["per_class"][name]), {"precision", "recall", "f1", "support"})
        for key in ("macro_f1", "balanced_accuracy", "accuracy", "loss"):
            self.assertIn(key, best)
        data = run["data"]
        self.assertEqual(data["split_file"]["sha256"], tr.sha256_file(self.data.split_file))
        self.assertEqual(data["metadata"]["sha256"], tr.sha256_file(self.data.metadata))
        self.assertEqual(data["normalization_stats"]["sha256"], tr.sha256_file(self.stats_path))
        self.assertTrue(all(data["checks"].values()), data["checks"])
        self.assertEqual(set(data["datasets"]), {"train", "val"})  # test split never touched
        env = run["environment"]
        for key in ("python", "torch", "torch_cuda_build", "cudnn", "cuda_available", "device"):
            self.assertIn(key, env)
        self.assertIn("git_commit", run["code"])
        self.assertEqual([h["epoch"] for h in history], [0, 1])
        for h in history:
            for key in ("lr", "train_loss", "val_loss", "val_macro_f1", "val_balanced_accuracy", "val_accuracy"):
                self.assertIn(key, h)
            self.assertEqual(h["lr"], 1e-3)

    def test_existing_run_is_not_overwritten(self):
        self.fit(self.cfg(run_name="dup", max_epochs=1))
        with self.assertRaises(FileExistsError):
            self.fit(self.cfg(run_name="dup", max_epochs=1))
        self.fit(self.cfg(run_name="dup", max_epochs=1), overwrite=True)

    def test_failed_run_is_recorded(self):
        def boom(*_):
            raise RuntimeError("simulated failure")
        original = tr.evaluate
        tr.evaluate = boom
        try:
            with self.assertRaises(RuntimeError):
                self.fit(self.cfg(run_name="fail", max_epochs=1))
        finally:
            tr.evaluate = original
        run = json.loads((Path(self.tmp.name) / "reports" / "fail" / "run.json").read_text())
        self.assertEqual(run["status"], "failed")
        self.assertIn("simulated failure", run["error"])

    def test_mismatched_stats_file_is_flagged(self):
        other = Path(self.tmp.name) / "other_stats.json"
        norm = self.train.normalizer
        shifted = nz.BandNormalizer(norm.mean + 1, norm.std, norm.config, norm.train_patients,
                                    norm.n_cycles, norm.n_frames)
        nz.save_normalizer(shifted, other, self.data.metadata, self.data.split_file)
        r = tr.fit(self.cfg(run_name="mm", max_epochs=1), self.train, self.val, log=lambda *_: None,
                   normalization_path=other)
        self.assertFalse(r["run"]["data"]["checks"]["normalizer_used_matches_stats_file"])

    def test_early_stopping(self):
        r = self.fit(self.cfg(max_epochs=20, patience=1, lr=0.0))
        self.assertEqual(len(r["history"]), 2)  # lr=0 -> no improvement after epoch 0
        self.assertEqual(r["run"]["stop_reason"], "early_stopping: no val macro-F1 improvement for 1 epochs")

    def test_tiny_overfit(self):
        tr.set_determinism(0)
        model = BaselineCNN(dropout=0.0)
        loader = tr.make_eval_loader(self.train, 8)
        batch = next(iter(loader))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(150):
            opt.zero_grad()
            loss = loss_fn(model(batch["features"], batch["mask"]), batch["labels"])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(batch["features"], batch["mask"]).argmax(1)
        self.assertEqual(pred.tolist(), batch["labels"].tolist())
        self.assertLess(loss.item(), 0.05)

    def test_subset_keeps_split_and_patients(self):
        idx = tr.stratified_subset_indices(self.train, per_class=1, seed=0)
        sub = tr.subset(self.train, idx)
        self.assertEqual(len(sub), 4)
        self.assertEqual(sub.split, "train")
        self.assertTrue(sub.patients <= self.train.patients)
        self.assertEqual(Counter(r["cycle_label"] for r in sub.rows), Counter(rd.CLASS_NAMES))


class TestCodeVersion(unittest.TestCase):
    def fake_git(self, porcelain: str):
        def run(args, **_):
            out = porcelain if args[:2] == ["git", "status"] else "abc123\n"
            return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")
        return mock.patch.object(tr.subprocess, "run", side_effect=run)

    def test_uncommitted_paths_keep_first_character(self):
        # regression: stripping ' M src/...' made the first path 'rc/...'
        with self.fake_git(" M src/evaluation/analyze_run.py\n?? reports/runs/seed_study/\nM  src/models/train.py\n"):
            code = tr.code_version()
        self.assertEqual(code["git_commit"], "abc123")
        self.assertTrue(code["git_dirty"])
        self.assertEqual(code["git_uncommitted_paths"],
                         ["src/evaluation/analyze_run.py", "reports/runs/seed_study/", "src/models/train.py"])

    def test_clean_tree(self):
        with self.fake_git(""):
            code = tr.code_version()
        self.assertEqual((code["git_dirty"], code["git_uncommitted_paths"]), (False, []))


if __name__ == "__main__":
    unittest.main()
