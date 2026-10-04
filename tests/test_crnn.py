"""Tests for the CRNN (padding/masking safety) and its integration with training and analysis."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("models", "evaluation", "data"):
    sys.path.insert(0, str(REPO_ROOT / "src" / sub))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_run as ar  # noqa: E402
import normalization as nz  # noqa: E402
import train as tr  # noqa: E402
from baseline_cnn import BaselineCNN, count_parameters  # noqa: E402
from crnn import CRNN  # noqa: E402
from test_baseline_cnn import padded_batch  # noqa: E402
from test_respiratory_dataset import SyntheticICBHI  # noqa: E402


class TestCRNNModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = CRNN().eval()

    def test_shape_and_size(self):
        x, mask, _ = padded_batch([21, 100, 254])
        self.assertEqual(tuple(self.model(x, mask).shape), (3, 4))
        self.assertEqual(count_parameters(self.model), 353_365)
        self.assertLess(count_parameters(self.model), 1_500_000)
        self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.model.modules()))

    def test_temporal_resolution(self):
        x, mask, _ = padded_batch([254])
        out, lengths = self.model.encode(x, mask)
        self.assertEqual(lengths.tolist(), [127])  # one 2x time pooling: 20 ms steps
        self.assertEqual(tuple(out.shape), (1, 127, 256))

    def test_output_independent_of_padding_and_batch(self):
        lengths = [21, 37, 50, 1, 2, 3, 160]
        for pad_fill in (0.0, 123.0, float("nan")):
            with self.subTest(pad_fill=pad_fill):
                x, mask, singles = padded_batch(lengths, pad_fill=pad_fill)
                with torch.no_grad():
                    batched = self.model(x, mask)
                    for i, s in enumerate(singles):
                        alone = self.model(s, torch.ones(1, s.shape[-1], dtype=torch.bool))
                        self.assertTrue(torch.allclose(batched[i], alone[0], atol=1e-5), (pad_fill, i))
                self.assertTrue(torch.isfinite(batched).all())

    def test_extra_padding_does_not_change_predictions(self):
        x, mask, _ = padded_batch([40, 90, 133])
        extra = 77
        x2 = torch.cat([x, torch.randn(3, 1, 64, extra) * 50], dim=-1)
        mask2 = torch.cat([mask, torch.zeros(3, extra, dtype=torch.bool)], dim=1)
        with torch.no_grad():
            self.assertTrue(torch.allclose(self.model(x, mask), self.model(x2, mask2), atol=1e-5))

    def test_gru_outputs_match_unpadded_on_valid_steps(self):
        x, mask, singles = padded_batch([30, 75])
        with torch.no_grad():
            out, lengths = self.model.encode(x, mask)
            alone, alone_len = self.model.encode(singles[0], torch.ones(1, 30, dtype=torch.bool))
        n = int(alone_len[0])
        self.assertEqual(int(lengths[0]), n)
        self.assertTrue(torch.allclose(out[0, :n], alone[0], atol=1e-5))
        self.assertTrue(torch.all(out[0, n:] == 0))  # pad_packed_sequence zero-fills padded steps

    def test_attention_masks_padding(self):
        x, mask, _ = padded_batch([20, 64, 9])
        with torch.no_grad():
            out, lengths = self.model.encode(x, mask)
            w = self.model.attention_weights(out, lengths)
        self.assertTrue(torch.allclose(w.sum(dim=1), torch.ones(3), atol=1e-6))
        for i, n in enumerate(lengths.tolist()):
            self.assertTrue(torch.all(w[i, n:] == 0))
            self.assertTrue(torch.all(w[i, :n] > 0))

    def test_padded_inputs_get_zero_gradient(self):
        x, mask, _ = padded_batch([30, 80])
        x.requires_grad_(True)
        self.model.train()
        self.model(x, mask).sum().backward()
        self.assertTrue(torch.all(x.grad[0, ..., 30:] == 0))
        self.assertTrue(x.grad[0, ..., :30].abs().sum() > 0)

    def test_rejects_bad_inputs(self):
        x, mask, _ = padded_batch([10, 20])
        with self.assertRaises(ValueError):
            self.model(x[:, :, :32], mask)
        with self.assertRaises(ValueError):
            self.model(x, mask[:, :5])
        with self.assertRaises(ValueError):
            self.model(x, torch.zeros_like(mask))


class TestModelFactory(unittest.TestCase):
    def test_baseline_construction_unchanged(self):
        torch.manual_seed(42)
        direct = BaselineCNN(4, (16, 32, 64, 128), dropout=0.3)
        torch.manual_seed(42)
        built = tr.build_model(tr.TrainConfig())
        self.assertIsInstance(built, BaselineCNN)
        for (k, a), (_, b) in zip(direct.state_dict().items(), built.state_dict().items()):
            self.assertTrue(torch.equal(a, b), k)

    def test_crnn_and_legacy_configs(self):
        self.assertIsInstance(tr.build_model(tr.TrainConfig(model="crnn")), CRNN)
        legacy = {"channels": [16, 32, 64, 128], "dropout": 0.3}  # saved before the `model` field existed
        self.assertIsInstance(tr.build_model(legacy), BaselineCNN)
        with self.assertRaises(ValueError):
            tr.build_model({"model": "transformer", "channels": [16], "dropout": 0.3})


class TestCRNNTraining(unittest.TestCase):
    def test_fit_and_analyze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = SyntheticICBHI(root / "data")
            norm = nz.compute_band_stats(data.dataset("train"))
            stats = root / "stats.json"
            nz.save_normalizer(norm, stats, data.metadata, data.split_file)
            train, val = data.dataset("train", normalizer=norm), data.dataset("val", normalizer=norm)
            cfg = tr.TrainConfig(run_name="c", model="crnn", max_epochs=2, batch_size=4, device="cpu",
                                 checkpoint_dir=str(root / "models"), reports_dir=str(root / "reports"),
                                 bucket_pool_multiplier=2)
            r1 = tr.fit(cfg, train, val, log=lambda *_: None, normalization_path=stats)
            run = json.loads((r1["report_dir"] / "run.json").read_text())
            self.assertEqual((run["model"]["name"], run["model"]["class"]), ("crnn", "CRNN"))
            self.assertEqual(run["model"]["parameter_count"], 353_365)
            self.assertIn("GRU", run["model"]["architecture"])
            cfg2 = tr.TrainConfig(**{**tr.asdict(cfg), "run_name": "c2"})
            r2 = tr.fit(cfg2, train, val, log=lambda *_: None, normalization_path=stats)
            self.assertEqual([h["train_loss"] for h in r1["history"]], [h["train_loss"] for h in r2["history"]])
            res = ar.analyze("c", root / "models", root / "reports", dataset=val, device="cpu")
            self.assertTrue(res["reproduces_logged_val_confusion_matrix"])

    def test_tiny_overfit(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = SyntheticICBHI(Path(tmp) / "data")
            norm = nz.compute_band_stats(data.dataset("train"))
            batch = next(iter(tr.make_eval_loader(data.dataset("train", normalizer=norm), 8)))
        tr.set_determinism(0)
        model = CRNN(dropout=0.0)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(150):
            opt.zero_grad()
            loss = loss_fn(model(batch["features"], batch["mask"]), batch["labels"])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            self.assertEqual(model(batch["features"], batch["mask"]).argmax(1).tolist(), batch["labels"].tolist())


if __name__ == "__main__":
    unittest.main()
