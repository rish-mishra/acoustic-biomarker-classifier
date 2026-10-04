"""Train the baseline CNN on the frozen train split; select on validation macro F1.

The test split is never loaded here. evaluate() refuses a test dataset unless
allow_test=True, which is reserved for the final, one-time evaluation.

Pipeline (all frozen): RespiratoryCycleDataset (PRIMARY_CONFIG Log-Mel) ->
train-only BandNormalizer -> collate_cycles (batch padding + mask).

Training details:
  * Weighted cross-entropy, weights = N / (K * n_c) from TRAIN label counts only.
  * AdamW; gradient-norm clipping.
  * Length-bucketed training batches (LengthBucketBatchSampler), reshuffled
    reproducibly each epoch; validation uses plain sequential batches.
  * Early stopping on validation macro F1 (ties keep the earlier epoch).
  * Seeds for random / numpy / torch / CUDA, cuDNN deterministic mode, and
    torch.use_deterministic_algorithms (warn_only, so a missing deterministic
    kernel is reported instead of crashing).

Outputs (an existing run is never overwritten unless overwrite=True):
  models/<run_name>/best.pt           best checkpoint (weights; ignored by Git)
  reports/runs/<run_name>/run.json    run name, seed, full config, parameter count,
                                      class weights, data / split / normalization
                                      SHA-256 identifiers, code version, environment
                                      (Python, PyTorch, CUDA, cuDNN, GPU), status,
                                      best epoch, stop reason, total time, and the
                                      best epoch's full validation report
                                      (confusion matrix, per-class P/R/F1)
  reports/runs/<run_name>/history.json  one record per epoch (train loss, val loss,
                                      macro F1, balanced accuracy, accuracy, learning
                                      rate, per-class F1, confusion matrix, timing)

Data loading: keep num_workers=0 (the default). On this Windows machine a
num_workers=3 smoke run stalled for over 10 minutes; single-process loading
plus the in-memory recording cache is the tested configuration.

Usage::

    python src/models/train.py --run-name baseline_cnn_v1
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required for deterministic cuBLAS

import argparse
import copy
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1]
for _sub in ("data", "features", "models", "evaluation"):
    if str(SRC_DIR / _sub) not in sys.path:
        sys.path.insert(0, str(SRC_DIR / _sub))

import normalization as nz  # noqa: E402
import respiratory_dataset as rd  # noqa: E402
from baseline_cnn import BaselineCNN, count_parameters  # noqa: E402
from crnn import CRNN  # noqa: E402
from mobilenet_transfer import MobileNetV3Transfer  # noqa: E402
from classification_metrics import classification_report, confusion_matrix  # noqa: E402
from length_bucketing import LengthBucketBatchSampler, approximate_frame_lengths  # noqa: E402

REPO_ROOT = SRC_DIR.parent
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "models"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "runs"


MODELS = ("baseline_cnn", "crnn", "mobilenetv3_pretrained")


@dataclass
class TrainConfig:
    run_name: str = "baseline_cnn"
    model: str = "baseline_cnn"  # "crnn" uses CRNN's fixed default architecture; `channels` applies to baseline_cnn only
    seed: int = 42
    max_epochs: int = 60
    patience: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-2
    dropout: float = 0.3
    channels: tuple[int, ...] = (16, 32, 64, 128)
    grad_clip_norm: float = 5.0
    use_bucketing: bool = True
    bucket_pool_multiplier: int = 50
    num_workers: int = 0
    device: str = "cuda"
    checkpoint_dir: str = str(DEFAULT_CHECKPOINT_DIR)
    reports_dir: str = str(DEFAULT_REPORTS_DIR)
    extra: dict = field(default_factory=dict)


def build_model(config: dict | TrainConfig) -> nn.Module:
    """Construct the network described by a TrainConfig (or a saved config dict)."""
    cfg = asdict(config) if isinstance(config, TrainConfig) else config
    name = cfg.get("model", "baseline_cnn")  # runs saved before this field existed are baseline_cnn
    if name == "baseline_cnn":
        return BaselineCNN(len(rd.CLASS_NAMES), tuple(cfg["channels"]), dropout=cfg["dropout"])
    if name == "crnn":
        return CRNN(len(rd.CLASS_NAMES), dropout=cfg["dropout"])
    if name == "mobilenetv3_pretrained":
        return MobileNetV3Transfer(len(rd.CLASS_NAMES), dropout=cfg["dropout"], pretrained=True)
    raise ValueError(f"unknown model {name!r}; expected one of {MODELS}")


def pretrained_weights_info(model_name: str) -> dict:
    """Provenance of pretrained weights (empty for models trained from scratch)."""
    if model_name != "mobilenetv3_pretrained":
        return {}
    import torchvision
    from mobilenet_transfer import PRETRAINED_WEIGHTS

    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(PRETRAINED_WEIGHTS.url).name
    return {"pretrained_weights": {
        "name": str(PRETRAINED_WEIGHTS), "url": PRETRAINED_WEIGHTS.url,
        "file_sha256": sha256_file(path) if path.exists() else None,
        "torchvision": torchvision.__version__,
        "imagenet_metrics": PRETRAINED_WEIGHTS.meta["_metrics"]["ImageNet-1K"],
    }}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_cmd(args: list[str]) -> str | None:
    try:
        out = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info(device: torch.device) -> dict:
    import soundfile
    import torchaudio

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "numpy": np.__version__,
        "soundfile": soundfile.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update({
            "gpu_name": props.name,
            "gpu_total_memory_mb": round(props.total_memory / 2 ** 20),
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "nvidia_driver": _run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        })
    return info


def code_version() -> dict:
    status = _run_cmd(["git", "status", "--porcelain"])
    return {
        "git_commit": _run_cmd(["git", "rev-parse", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
        "git_uncommitted_paths": None if status is None else [line[3:] for line in status.splitlines()],
    }


def data_provenance(train_ds, val_ds, normalization_path: Path) -> dict:
    """SHA-256 identifiers for every frozen input, plus consistency checks."""
    normalization_path = Path(normalization_path)
    stats_payload = json.loads(normalization_path.read_text(encoding="utf-8"))
    file_norm = nz.load_normalizer(normalization_path)
    used = train_ds.normalizer
    split_sha = sha256_file(train_ds.split_path)
    metadata_sha = sha256_file(train_ds.metadata_path)
    return {
        "metadata": {"path": str(train_ds.metadata_path), "sha256": metadata_sha},
        "split_file": {"path": str(train_ds.split_path), "sha256": split_sha},
        "normalization_stats": {"path": str(normalization_path), "sha256": sha256_file(normalization_path)},
        "audio_dir": str(train_ds.audio_dir),
        "checks": {
            "normalizer_used_matches_stats_file": used is not None and bool(
                torch.equal(used.mean, file_norm.mean) and torch.equal(used.std, file_norm.std)),
            "stats_file_fitted_on_current_split_file": stats_payload.get("split_file_sha256") == split_sha,
            "stats_file_fitted_on_current_metadata": stats_payload.get("metadata_sha256") == metadata_sha,
            "val_uses_same_split_file": Path(val_ds.split_path) == Path(train_ds.split_path),
        },
        "datasets": {
            ds.split: {"cycles": len(ds), "patients": len(ds.patients), "label_counts": ds.label_counts()}
            for ds in (train_ds, val_ds)
        },
        "log_mel_config": asdict(train_ds.config),
    }


def _write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def class_weights_from_train(dataset: rd.RespiratoryCycleDataset) -> torch.Tensor:
    """Balanced weights N / (K * n_c) from TRAIN label counts only."""
    if dataset.split != "train":
        raise ValueError(f"class weights must come from the train split, not {dataset.split!r}")
    counts = torch.tensor([dataset.label_counts()[c] for c in rd.CLASS_NAMES], dtype=torch.float64)
    if bool((counts == 0).any()):
        raise ValueError(f"train split is missing a class: {dict(zip(rd.CLASS_NAMES, counts.tolist()))}")
    return (counts.sum() / (len(counts) * counts)).to(torch.float32)


def subset(dataset: rd.RespiratoryCycleDataset, indices) -> rd.RespiratoryCycleDataset:
    """Same split and pipeline, restricted to ``indices`` (shares the recording cache)."""
    out = copy.copy(dataset)
    out.rows = [dataset.rows[i] for i in indices]
    out.patients = frozenset(r["patient_id"] for r in out.rows)
    return out


def stratified_subset_indices(dataset: rd.RespiratoryCycleDataset, per_class: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    chosen = []
    for name in rd.CLASS_NAMES:
        idx = [i for i, r in enumerate(dataset.rows) if r["cycle_label"] == name]
        chosen += sorted(rng.sample(idx, min(per_class, len(idx))))
    return sorted(chosen)


def make_train_loader(dataset, cfg: TrainConfig) -> DataLoader:
    g = torch.Generator().manual_seed(cfg.seed)
    common = dict(collate_fn=rd.collate_cycles, num_workers=cfg.num_workers,
                  persistent_workers=cfg.num_workers > 0, pin_memory=cfg.device.startswith("cuda"))
    if cfg.use_bucketing:
        sampler = LengthBucketBatchSampler(approximate_frame_lengths(dataset.rows), cfg.batch_size, cfg.seed,
                                           cfg.bucket_pool_multiplier)
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, generator=g, **common)


def make_eval_loader(dataset, batch_size: int, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=rd.collate_cycles,
                      num_workers=num_workers, persistent_workers=num_workers > 0)


def _set_epoch(loader: DataLoader, epoch: int) -> None:
    # Plain shuffling needs nothing here: its seeded generator advances deterministically each epoch.
    if isinstance(loader.batch_sampler, LengthBucketBatchSampler):
        loader.batch_sampler.set_epoch(epoch)


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip_norm, epoch: int,
                    max_batches: int | None = None) -> dict:
    model.train()
    _set_epoch(loader, epoch)
    total_loss, n, cells, valid = 0.0, 0, 0, 0
    data_time = compute_time = 0.0
    t_prev = time.perf_counter()
    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        t_data = time.perf_counter()
        data_time += t_data - t_prev
        x = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x, mask), y)
        loss.backward()
        if grad_clip_norm:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
        cells += mask.numel()
        valid += int(batch["lengths"].sum())
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_prev = time.perf_counter()
        compute_time += t_prev - t_data
    return {"loss": total_loss / max(n, 1), "cycles": n, "padding_fraction": 1 - valid / max(cells, 1),
            "data_time_s": data_time, "compute_time_s": compute_time}


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None, allow_test: bool = False) -> dict:
    """Predict a whole split and compute metrics. Refuses the test split unless allow_test."""
    if getattr(loader.dataset, "split", None) == "test" and not allow_test:
        raise PermissionError("the test split is reserved for the final evaluation; pass allow_test=True only then")
    model.eval()
    preds, labels, total_loss = [], [], 0.0
    for batch in loader:
        x = batch["features"].to(device)
        mask = batch["mask"].to(device)
        y = batch["labels"].to(device)
        logits = model(x, mask)
        if loss_fn is not None:
            total_loss += float(loss_fn(logits, y)) * len(y)
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(y.cpu())
    y_true, y_pred = torch.cat(labels), torch.cat(preds)
    report = classification_report(confusion_matrix(y_true, y_pred, len(rd.CLASS_NAMES)), rd.CLASS_NAMES)
    if loss_fn is not None:
        report["loss"] = total_loss / len(y_true)
    return report


def fit(cfg: TrainConfig, train_ds, val_ds, max_train_batches: int | None = None, log=print,
        normalization_path: Path = nz.DEFAULT_STATS_PATH, overwrite: bool = False) -> dict:
    if train_ds.split != "train" or val_ds.split != "val":
        raise ValueError("fit() trains on the train split and selects on the val split only")
    report_dir = Path(cfg.reports_dir) / cfg.run_name
    ckpt_dir = Path(cfg.checkpoint_dir) / cfg.run_name
    ckpt_path = ckpt_dir / "best.pt"
    if not overwrite and ((report_dir / "run.json").exists() or ckpt_path.exists()):
        raise FileExistsError(f"run {cfg.run_name!r} already exists ({report_dir} / {ckpt_dir}); "
                              "choose a new run name or pass overwrite=True")

    set_determinism(cfg.seed)
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    weights = class_weights_from_train(train_ds).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_loader = make_train_loader(train_ds, cfg)
    val_loader = make_eval_loader(val_ds, cfg.batch_size, cfg.num_workers)

    report_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    provenance = data_provenance(train_ds, val_ds, normalization_path)
    run = {
        "run_name": cfg.run_name,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "config": asdict(cfg),
        "max_train_batches": max_train_batches,
        "model": {"name": cfg.model, "class": type(model).__name__, "parameter_count": count_parameters(model),
                  "architecture": str(model), **pretrained_weights_info(cfg.model)},
        "class_names": list(rd.CLASS_NAMES),
        "class_weights": dict(zip(rd.CLASS_NAMES, weights.tolist())),
        "class_weight_source": "train split label counts: N / (K * n_c)",
        "optimizer": {"type": type(optimizer).__name__, **{k: v for k, v in optimizer.defaults.items()
                                                           if isinstance(v, (int, float, bool, str, tuple))}},
        "loss": "CrossEntropyLoss(weight=class_weights)",
        "selection_metric": "val_macro_f1",
        "test_split_evaluated": False,
        "data": provenance,
        "code": code_version(),
        "environment": environment_info(device),
        "checkpoint": str(ckpt_path),
    }
    _write_json(report_dir / "run.json", run)

    log(f"device={device} params={count_parameters(model):,} class_weights={[round(w, 3) for w in weights.tolist()]}")
    history, best, best_val, stale = [], None, None, 0
    stop_reason = f"max_epochs_reached ({cfg.max_epochs})"
    t_start = time.perf_counter()
    try:
        for epoch in range(cfg.max_epochs):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_stats = train_one_epoch(model, train_loader, optimizer, loss_fn, device, cfg.grad_clip_norm,
                                          epoch, max_train_batches)
            t_train = time.perf_counter() - t0
            val = evaluate(model, val_loader, device, loss_fn)
            record = {"epoch": epoch, "lr": lr, **{f"train_{k}": v for k, v in train_stats.items()},
                      "train_time_s": t_train, "epoch_time_s": time.perf_counter() - t0,
                      "val_loss": val["loss"], "val_macro_f1": val["macro_f1"],
                      "val_balanced_accuracy": val["balanced_accuracy"], "val_accuracy": val["accuracy"],
                      "val_per_class_f1": {k: v["f1"] for k, v in val["per_class"].items()},
                      "val_confusion_matrix": val["confusion_matrix"],
                      "gpu_peak_mem_mb": torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else 0.0}
            history.append(record)
            improved = best is None or val["macro_f1"] > best["val_macro_f1"]
            if improved:
                best, best_val, stale = record, val, 0
                torch.save({"model_state": model.state_dict(), "config": asdict(cfg), "epoch": epoch,
                            "val_metrics": val, "class_names": rd.CLASS_NAMES,
                            "data_sha256": {k: provenance[k]["sha256"]
                                            for k in ("metadata", "split_file", "normalization_stats")}},
                           ckpt_path)
            else:
                stale += 1
            log(f"epoch {epoch:>3} | train loss {train_stats['loss']:.4f} | val loss {val['loss']:.4f} "
                f"macro-F1 {val['macro_f1']:.4f} bal-acc {val['balanced_accuracy']:.4f} acc {val['accuracy']:.4f} | "
                f"{record['epoch_time_s']:.1f}s pad {train_stats['padding_fraction']:.1%}{' *' if improved else ''}")
            _write_json(report_dir / "history.json", history)
            if stale >= cfg.patience:
                stop_reason = f"early_stopping: no val macro-F1 improvement for {cfg.patience} epochs"
                log(stop_reason)
                break
    except BaseException as e:
        run.update(status="failed", error=f"{type(e).__name__}: {e}", epochs_completed=len(history),
                   total_training_time_s=time.perf_counter() - t_start)
        _write_json(report_dir / "run.json", run)
        raise

    run.update(
        status="completed",
        finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        epochs_completed=len(history),
        stop_reason=stop_reason,
        total_training_time_s=time.perf_counter() - t_start,
        best_epoch=best["epoch"],
        best_validation=best_val,  # confusion matrix, per-class precision/recall/F1, macro F1, bal. acc., acc.
        peak_gpu_memory_mb=max(h["gpu_peak_mem_mb"] for h in history),
    )
    _write_json(report_dir / "run.json", run)
    return {"best": best, "history": history, "model": model, "report_dir": report_dir,
            "checkpoint": ckpt_path, "run": run}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the baseline CNN (train split) with validation model selection.")
    d = TrainConfig()
    p.add_argument("--run-name", default=d.run_name)
    p.add_argument("--model", choices=MODELS, default=d.model)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--max-epochs", type=int, default=d.max_epochs)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--no-bucketing", action="store_true")
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--device", default=d.device)
    p.add_argument("--checkpoint-dir", default=d.checkpoint_dir)
    p.add_argument("--reports-dir", default=d.reports_dir)
    p.add_argument("--normalization-stats", type=Path, default=nz.DEFAULT_STATS_PATH)
    p.add_argument("--overwrite", action="store_true", help="replace an existing run with the same name")
    p.add_argument("--max-train-batches", type=int, default=None, help="limit batches per epoch (smoke tests)")
    a = p.parse_args(argv)
    cfg = TrainConfig(run_name=a.run_name, model=a.model, seed=a.seed, max_epochs=a.max_epochs, patience=a.patience,
                      batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay, dropout=a.dropout,
                      use_bucketing=not a.no_bucketing, num_workers=a.num_workers, device=a.device,
                      checkpoint_dir=a.checkpoint_dir, reports_dir=a.reports_dir)
    normalizer = nz.load_normalizer(a.normalization_stats)
    train_ds = rd.RespiratoryCycleDataset("train", normalizer=normalizer)
    val_ds = rd.RespiratoryCycleDataset("val", normalizer=normalizer)
    result = fit(cfg, train_ds, val_ds, a.max_train_batches, normalization_path=a.normalization_stats,
                 overwrite=a.overwrite)
    b = result["best"]
    print(f"best epoch {b['epoch']}: val macro-F1 {b['val_macro_f1']:.4f}; {result['run']['stop_reason']}")
    print(f"reports: {result['report_dir']}  checkpoint: {result['checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
