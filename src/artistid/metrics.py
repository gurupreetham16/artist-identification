from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, f1_score


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def classification_metrics(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> dict:
    p = softmax(logits, temperature)
    pred = p.argmax(1)
    n, c = p.shape
    top5 = np.argsort(-p, axis=1)[:, : min(5, c)]
    onehot = np.eye(c)[y]
    conf = p.max(1)
    correct = (pred == y).astype(float)
    ece, edges = 0.0, np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "top5": float((top5 == y[:, None]).any(1).mean()),
        "nll": float(-np.log(p[np.arange(n), y] + 1e-12).mean()),
        "brier": float(((p - onehot) ** 2).sum(1).mean()),
        "ece": float(ece),
        "n": int(n),
    }


def per_artist_f1(logits: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    pred = logits.argmax(1)
    f1 = f1_score(y, pred, average=None, labels=list(range(len(names))))
    return {n: float(v) for n, v in zip(names, f1)}


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact two-sided McNemar test on paired per-image correctness."""
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    n = a_only + b_only
    p = float(binomtest(min(a_only, b_only), n, 0.5).pvalue) if n else 1.0
    return {"a_only": a_only, "b_only": b_only, "p_value": p}
