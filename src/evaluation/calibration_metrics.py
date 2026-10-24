"""Calibration, uncertainty, error-detection and selective-prediction metrics (NumPy).

Conventions (see reports/calibration_uncertainty/plan.json):
  * probs: (N, K) softmax probabilities; labels: (N,) int class indices.
  * Natural logarithms throughout.
  * Zero division gives 0 (same as classification_metrics._safe_div), so a
    class with no support has recall 0 and a never-predicted class has
    precision 0; both give F1 0 and the class still counts in macro averages.
  * Ties are broken deterministically by input order (stable sorts).

Nothing here touches data on disk; callers pass validation arrays only.
"""

from __future__ import annotations

import math
import random

import numpy as np

N_BINS = 15


# ---------------------------------------------------------------- basics

def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def logits_from_probs(probs: np.ndarray) -> np.ndarray:
    """ln p: equal to the model's logits up to a per-row constant, which softmax ignores."""
    if (probs <= 0).any():
        raise ValueError("probabilities must be strictly positive to recover logits")
    return np.log(probs)


def check_inputs(probs: np.ndarray, labels: np.ndarray) -> None:
    if probs.ndim != 2 or labels.shape != (probs.shape[0],) or probs.shape[0] == 0:
        raise ValueError(f"expected (N, K) probs and (N,) labels, got {probs.shape} and {labels.shape}")
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1.0, atol=1e-5):
        raise ValueError("probs must be finite, non-negative rows summing to 1")
    if labels.min() < 0 or labels.max() >= probs.shape[1]:
        raise ValueError("labels out of range")


def confidence(probs: np.ndarray) -> np.ndarray:
    return probs.max(axis=1)


def predictions(probs: np.ndarray) -> np.ndarray:
    return probs.argmax(axis=1)


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-300, 1.0)
    return -(probs * np.log(p)).sum(axis=1)


def margin(probs: np.ndarray) -> np.ndarray:
    s = np.sort(probs, axis=1)
    return s[:, -1] - s[:, -2]


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(probs[np.arange(len(labels)), labels]).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


# ---------------------------------------------------------------- calibration error

def equal_width_bin(conf: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Bin index for [0, 1/n], (1/n, 2/n], ..., ((n-1)/n, 1]."""
    return np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    idx = equal_width_bin(conf, n_bins)
    out = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        out.append({"bin": b, "lower": b / n_bins, "upper": (b + 1) / n_bins, "n": n,
                    "accuracy": float(correct[m].mean()) if n else None,
                    "mean_confidence": float(conf[m].mean()) if n else None})
    return out


def _weighted_gap(groups: list[np.ndarray], conf: np.ndarray, hit: np.ndarray) -> tuple[float, float]:
    n = len(conf)
    gaps = [(len(g) / n, abs(float(hit[g].mean()) - float(conf[g].mean()))) for g in groups if len(g)]
    return sum(w * g for w, g in gaps), max((g for _, g in gaps), default=0.0)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    """Top-label ECE with equal-width bins (primary)."""
    conf, correct = confidence(probs), (predictions(probs) == labels).astype(float)
    idx = equal_width_bin(conf, n_bins)
    return _weighted_gap([np.flatnonzero(idx == b) for b in range(n_bins)], conf, correct)[0]


def mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    conf, correct = confidence(probs), (predictions(probs) == labels).astype(float)
    idx = equal_width_bin(conf, n_bins)
    return _weighted_gap([np.flatnonzero(idx == b) for b in range(n_bins)], conf, correct)[1]


def ece_equal_mass(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    """Top-label ECE with n_bins contiguous groups of (near-)equal size after a stable sort by confidence."""
    conf, correct = confidence(probs), (predictions(probs) == labels).astype(float)
    order = np.argsort(conf, kind="stable")
    return _weighted_gap(list(np.array_split(order, n_bins)), conf, correct)[0]


def classwise_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    """Mean over classes of the equal-width-binned |frequency of class k - mean p_k|."""
    vals = []
    for k in range(probs.shape[1]):
        pk, hit = probs[:, k], (labels == k).astype(float)
        idx = equal_width_bin(pk, n_bins)
        vals.append(_weighted_gap([np.flatnonzero(idx == b) for b in range(n_bins)], pk, hit)[0])
    return float(np.mean(vals))


# ---------------------------------------------------------------- classification

def classification_summary(labels: np.ndarray, preds: np.ndarray, k: int) -> dict:
    """Confusion-based metrics with the project's zero-division-gives-0 convention."""
    cm = np.bincount(labels * k + preds, minlength=k * k).reshape(k, k).astype(float)
    tp, support, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)  # noqa: E731
    precision, recall = div(tp, predicted), div(tp, support)
    f1 = div(2 * precision * recall, precision + recall)
    present = support > 0
    n = cm.sum()
    return {
        "n": int(n), "accuracy": float(tp.sum() / n) if n else 0.0,
        "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
        "macro_f1_present_classes": float(f1[present].mean()) if present.any() else 0.0,
        "balanced_accuracy_present_classes": float(recall[present].mean()) if present.any() else 0.0,
        "per_class_f1": f1.tolist(), "per_class_recall": recall.tolist(), "per_class_precision": precision.tolist(),
        "support": support.astype(int).tolist(), "predicted_counts": predicted.astype(int).tolist(),
        "confusion_matrix": cm.astype(int).tolist(),
    }


# ---------------------------------------------------------------- temperature scaling

def nll_at_inverse_temperature(logits: np.ndarray, labels: np.ndarray, beta: float) -> float:
    ls = log_softmax(logits * beta)
    return float(-ls[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray, beta_bounds: tuple[float, float] = (0.02, 50.0),
                    tol: float = 1e-9) -> dict:
    """T minimizing mean NLL, by golden-section search over beta = 1/T (NLL is convex in beta)."""
    if len(labels) == 0:
        raise ValueError("cannot fit a temperature on zero cycles")
    lo, hi = beta_bounds
    if not 0 < lo < hi:
        raise ValueError(f"invalid beta bounds {beta_bounds}")
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll_at_inverse_temperature(logits, labels, c), nll_at_inverse_temperature(logits, labels, d)
    while b - a > tol:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll_at_inverse_temperature(logits, labels, c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll_at_inverse_temperature(logits, labels, d)
    beta = (a + b) / 2
    return {"temperature": 1.0 / beta, "inverse_temperature": beta,
            "nll": nll_at_inverse_temperature(logits, labels, beta),
            "at_bound": bool(beta - lo < 1e-6 or hi - beta < 1e-6), "n": int(len(labels))}


def apply_temperature(logits: np.ndarray, temperature) -> np.ndarray:
    """softmax(logits / T); ``temperature`` is a scalar or one value per row."""
    t = np.asarray(temperature, dtype=float)
    if (t <= 0).any():
        raise ValueError("temperature must be > 0")
    return softmax(logits / (t[:, None] if t.ndim else t))


def lopo_temperatures(logits: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict:
    """Leave-one-group-out temperatures: for each group g, T fitted on all rows not in g."""
    out = {}
    for gid in sorted(set(groups.tolist())):
        keep = groups != gid
        out[gid] = fit_temperature(logits[keep], labels[keep])
    return out


# ---------------------------------------------------------------- error detection

def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties counted 1/2 (Mann-Whitney U). NaN if one class is absent."""
    pos = positive.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    avg_rank = ends - (counts - 1) / 2  # 1-based average rank of each distinct value
    rank_sum = avg_rank[inv][pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the precision-recall step curve, positives = ``positive``; tied scores form one threshold."""
    pos = positive.astype(bool)
    n_pos = int(pos.sum())
    if n_pos == 0:
        return float("nan")
    vals, inv = np.unique(-scores, return_inverse=True)  # ascending -score = descending score
    tp = np.bincount(inv, weights=pos, minlength=len(vals)).cumsum()
    fp = np.bincount(inv, weights=~pos, minlength=len(vals)).cumsum()
    precision, recall = tp / (tp + fp), tp / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


# ---------------------------------------------------------------- selective prediction

def retention_order(score: np.ndarray, higher_is_more_confident: bool = True) -> np.ndarray:
    """Indices from most to least confident; ties keep input order."""
    return np.argsort(-score if higher_is_more_confident else score, kind="stable")


def n_retained(coverage: float, n: int) -> int:
    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    return max(1, math.floor(coverage * n + 0.5))


def risk_coverage(order: np.ndarray, correct: np.ndarray) -> dict:
    """Selective risk at every n = 1..N for a retention order, plus AURC and E-AURC."""
    err = 1.0 - correct[order].astype(float)
    n = np.arange(1, len(err) + 1)
    risk = np.cumsum(err) / n
    n_correct = int(correct.sum())
    oracle = np.maximum(0, n - n_correct) / n
    return {"coverage": n / len(err), "risk": risk, "oracle_risk": oracle,
            "aurc": float(risk.mean()), "e_aurc": float(risk.mean() - oracle.mean())}


def selective_at(order: np.ndarray, labels: np.ndarray, preds: np.ndarray, coverage: float, k: int) -> dict:
    n_keep = n_retained(coverage, len(order))
    kept, rejected = order[:n_keep], order[n_keep:]
    s = classification_summary(labels[kept], preds[kept], k)
    counts = lambda idx, arr: np.bincount(arr[idx], minlength=k).astype(int).tolist()  # noqa: E731
    return {"coverage": coverage, "n_retained": n_keep, "n_rejected": int(len(rejected)),
            "selective_risk": 1.0 - s["accuracy"], **{key: s[key] for key in (
                "accuracy", "macro_f1", "balanced_accuracy", "macro_f1_present_classes",
                "balanced_accuracy_present_classes", "per_class_f1", "per_class_recall")},
            "retained_true_counts": counts(kept, labels), "rejected_true_counts": counts(rejected, labels),
            "retained_pred_counts": counts(kept, preds), "rejected_pred_counts": counts(rejected, preds)}


# ---------------------------------------------------------------- bootstrap

def patient_bootstrap_indices(groups: list[str], n_boot: int = 2000, seed: int = 0) -> list[np.ndarray]:
    """Row indices of each patient-level bootstrap resample (same RNG scheme as compare_runs.patient_bootstrap)."""
    by: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    patients = sorted(by)
    rng = random.Random(seed)
    return [np.array([i for p in (rng.choice(patients) for _ in patients) for i in by[p]]) for _ in range(n_boot)]


def percentile_ci(values) -> list[float]:
    s = sorted(v for v in values if not math.isnan(v))
    if not s:
        return [float("nan"), float("nan")]
    return [s[int(0.025 * len(s))], s[int(0.975 * len(s)) - 1]]
