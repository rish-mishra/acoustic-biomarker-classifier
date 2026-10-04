"""Tests for the padding-aware ImageNet-pretrained MobileNetV3-Small transfer model."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torchvision
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("models", "evaluation", "data"):
    sys.path.insert(0, str(REPO_ROOT / "src" / sub))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_run as ar  # noqa: E402
import mobilenet_transfer as mt  # noqa: E402
import normalization as nz  # noqa: E402
import train as tr  # noqa: E402
from baseline_cnn import count_parameters  # noqa: E402
from test_baseline_cnn import padded_batch  # noqa: E402
from test_respiratory_dataset import SyntheticICBHI  # noqa: E402

WEIGHTS_CACHED = (Path(torch.hub.get_dir()) / "checkpoints" / Path(mt.PRETRAINED_WEIGHTS.url).name).exists()


class TestComponents(unittest.TestCase):
    def test_masked_batchnorm_matches_batchnorm_without_padding(self):
        torch.manual_seed(0)
        bn = nn.BatchNorm2d(8, eps=1e-3, momentum=0.01)
        bn.running_mean.normal_()
        bn.running_var.uniform_(0.5, 2)
        ours = mt.MaskedBatchNorm2d(bn)
        x = torch.randn(4, 8, 6, 20)
        lengths = torch.full((4,), 20)
        bn.train(), ours.train()
        self.assertTrue(torch.allclose(ours(x, lengths), bn(x), atol=1e-5))
        self.assertTrue(torch.allclose(ours.running_mean, bn.running_mean, atol=1e-6))
        self.assertTrue(torch.allclose(ours.running_var, bn.running_var, atol=1e-5))
        bn.eval(), ours.eval()
        self.assertTrue(torch.allclose(ours(x, lengths), bn(x), atol=1e-5))

    def test_masked_batchnorm_ignores_padding(self):
        bn = mt.MaskedBatchNorm2d(nn.BatchNorm2d(8))
        x = torch.randn(2, 8, 4, 30)
        lengths = torch.tensor([30, 12])
        noisy = x.clone()
        noisy[1, ..., 12:] = 1e4
        a = bn.train()(x, lengths)
        rm = bn.running_mean.clone()
        bn2 = mt.MaskedBatchNorm2d(nn.BatchNorm2d(8)).train()
        b = bn2(noisy, lengths)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
        self.assertTrue(torch.allclose(rm, bn2.running_mean))
        self.assertTrue(torch.all(b[1, ..., 12:] == 0))

    def test_gray_conv_equals_rgb_conv_on_replicated_input(self):
        rgb = nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False)
        gray = mt.rgb_to_gray_conv(rgb)
        x = torch.randn(2, 1, 64, 37)
        self.assertTrue(torch.allclose(gray(x), rgb(x.repeat(1, 3, 1, 1)), atol=1e-5))

    def test_conv_out_lengths(self):
        for k, s, p in ((3, 2, 1), (5, 2, 2), (5, 1, 2), (1, 1, 0)):
            conv = nn.Conv2d(1, 1, k, stride=s, padding=p)
            for n in (1, 2, 3, 20, 21, 254):
                with self.subTest(k=k, s=s, n=n):
                    expected = conv(torch.zeros(1, 1, 8, n)).shape[-1]
                    self.assertEqual(int(mt.conv_out_lengths(conv, torch.tensor([n]))[0]), expected)


class TestPaddingInvariance(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = mt.MobileNetV3Transfer(pretrained=False, dropout=0.0)

    def test_eval_output_independent_of_padding_and_batch(self):
        self.model.eval()
        lengths = [21, 37, 50, 1, 2, 33, 160]
        for pad_fill in (0.0, 123.0, float("nan")):
            with self.subTest(pad_fill=pad_fill):
                x, mask, singles = padded_batch(lengths, pad_fill=pad_fill)
                with torch.no_grad():
                    batched = self.model(x, mask)
                    for i, s in enumerate(singles):
                        alone = self.model(s, torch.ones(1, s.shape[-1], dtype=torch.bool))
                        self.assertTrue(torch.allclose(batched[i], alone[0], atol=1e-5), (pad_fill, i))
                self.assertTrue(torch.isfinite(batched).all())

    def test_train_mode_extra_padding_changes_nothing(self):
        x, mask, _ = padded_batch([40, 90, 133])
        x2 = torch.cat([x, torch.randn(3, 1, 64, 61) * 50], dim=-1)
        mask2 = torch.cat([mask, torch.zeros(3, 61, dtype=torch.bool)], dim=1)
        a, b = mt.MobileNetV3Transfer(pretrained=False, dropout=0.0), mt.MobileNetV3Transfer(pretrained=False, dropout=0.0)
        b.load_state_dict(a.state_dict())
        a.train(), b.train()
        out_a, out_b = a(x, mask), b(x2, mask2)
        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-4))
        for (k, va), (_, vb) in zip(a.state_dict().items(), b.state_dict().items()):
            self.assertTrue(torch.allclose(va.float(), vb.float(), atol=1e-5), k)  # running stats too

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
            self.model(x, mask[:, :5])
        with self.assertRaises(ValueError):
            self.model(x, torch.zeros_like(mask))


@unittest.skipUnless(WEIGHTS_CACHED, "ImageNet weights not downloaded")
class TestPretrained(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mt.MobileNetV3Transfer(pretrained=True)
        cls.reference = torchvision.models.mobilenet_v3_small(weights=mt.PRETRAINED_WEIGHTS).eval()

    def test_weights_loaded(self):
        ref, mine = self.reference.state_dict(), self.model.state_dict()
        carried = [k for k in ref if k in mine and mine[k].shape == ref[k].shape and torch.equal(mine[k], ref[k])]
        self.assertEqual(len(carried), len(ref) - 3)
        self.assertTrue(torch.equal(mine["features.0.0.weight"], ref["features.0.0.weight"].sum(1, keepdim=True)))
        self.assertEqual(tuple(mine["classifier.3.weight"].shape), (4, 1024))
        self.assertEqual(count_parameters(self.model), 1_521_668)

    def test_matches_torchvision_on_replicated_input_without_padding(self):
        """Block-wise masked forward == torchvision MobileNetV3 on (x, x, x) when nothing is padded."""
        x = torch.randn(3, 1, 64, 150)
        self.model.eval()
        with torch.no_grad():
            feats = x
            lengths = torch.full((3,), 150)
            for layer in self.model.features:
                feats, lengths = self.model._run(layer, feats, lengths)
            ours = self.model.classifier[:2](mt._masked_mean(feats, lengths).flatten(1))
            ref = self.reference.classifier[:2](self.reference.avgpool(self.reference.features(x.repeat(1, 3, 1, 1))).flatten(1))
        self.assertTrue(torch.allclose(ours, ref, atol=1e-4))

    def test_pretrained_eval_padding_invariance(self):
        x, mask, singles = padded_batch([21, 64, 200])
        self.model.eval()
        with torch.no_grad():
            batched = self.model(x, mask)
            for i, s in enumerate(singles):
                alone = self.model(s, torch.ones(1, s.shape[-1], dtype=torch.bool))
                self.assertTrue(torch.allclose(batched[i], alone[0], atol=1e-5))

    def test_fit_and_analyze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = SyntheticICBHI(root / "data")
            norm = nz.compute_band_stats(data.dataset("train"))
            stats = root / "stats.json"
            nz.save_normalizer(norm, stats, data.metadata, data.split_file)
            train, val = data.dataset("train", normalizer=norm), data.dataset("val", normalizer=norm)
            cfg = tr.TrainConfig(run_name="m", model="mobilenetv3_pretrained", max_epochs=1, batch_size=4,
                                 device="cpu", checkpoint_dir=str(root / "models"),
                                 reports_dir=str(root / "reports"), bucket_pool_multiplier=2)
            r = tr.fit(cfg, train, val, log=lambda *_: None, normalization_path=stats)
            run = json.loads((r["report_dir"] / "run.json").read_text())
            self.assertEqual(run["model"]["class"], "MobileNetV3Transfer")
            self.assertEqual(run["model"]["pretrained_weights"]["file_sha256"][:8], "047dcff4")
            res = ar.analyze("m", root / "models", root / "reports", dataset=val, device="cpu")
            self.assertTrue(res["reproduces_logged_val_confusion_matrix"])


if __name__ == "__main__":
    unittest.main()
