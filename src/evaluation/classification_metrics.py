"""Multi-class metrics from a confusion matrix (rows = true class, columns = predicted).

Per-class precision / recall / F1 use 0.0 when undefined (no predictions or
no support). Macro F1 and balanced accuracy (mean per-class recall) weight
every class equally; overall accuracy is reported as a secondary metric.
"""

from __future__ import annotations

from typing import Sequence

import torch


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    y_true, y_pred = torch.as_tensor(y_true).long().flatten(), torch.as_tensor(y_pred).long().flatten()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) and (min(y_true.min(), y_pred.min()) < 0 or max(y_true.max(), y_pred.max()) >= num_classes):
        raise ValueError("labels out of range")
    return torch.bincount(y_true * num_classes + y_pred, minlength=num_classes ** 2).view(num_classes, num_classes)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def classification_report(cm: torch.Tensor, class_names: Sequence[str]) -> dict:
    cm = cm.to(torch.float64)
    k = len(class_names)
    if cm.shape != (k, k):
        raise ValueError(f"confusion matrix shape {tuple(cm.shape)} does not match {k} classes")
    per_class = {}
    for i, name in enumerate(class_names):
        tp = float(cm[i, i])
        precision = _safe_div(tp, float(cm[:, i].sum()))
        recall = _safe_div(tp, float(cm[i].sum()))
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": _safe_div(2 * precision * recall, precision + recall),
            "support": int(cm[i].sum()),
        }
    return {
        "per_class": per_class,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / k,
        "balanced_accuracy": sum(v["recall"] for v in per_class.values()) / k,
        "accuracy": _safe_div(float(cm.trace()), float(cm.sum())),
        "confusion_matrix": cm.long().tolist(),
        "n": int(cm.sum()),
    }
