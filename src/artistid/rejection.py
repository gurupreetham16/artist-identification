"""Unknown-artist rejection: calibrated confidence + similarity to per-artist prototypes.

gate = mix * max calibrated class probability + (1 - mix) * rescaled max prototype cosine.
An image is accepted when gate > threshold. All fitting uses calibration / development data;
the test split is never used to choose mix, temperature or threshold.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

from .metrics import softmax


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def fit_prototypes(feats: np.ndarray, labels: np.ndarray, n_classes: int, k: int = 3, seed: int = 0) -> np.ndarray:
    """k L2-normalised prototypes per artist: k-means centroids, or the mean if too few images."""
    feats = l2(feats)
    protos = np.zeros((n_classes, k, feats.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        x = feats[labels == c]
        if len(x) == 0:
            raise ValueError(f"class {c} has no training features")
        if k > 1 and len(x) >= k:
            cents = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x).cluster_centers_
        else:
            cents = np.repeat(x.mean(0, keepdims=True), k, axis=0)
        protos[c] = l2(cents)
    return protos


def prototype_score(feats: np.ndarray, protos: np.ndarray) -> np.ndarray:
    sims = l2(feats) @ protos.reshape(-1, protos.shape[-1]).T
    return (sims.max(1) + 1.0) / 2.0


def fit_temperature(logits: np.ndarray, y: np.ndarray, grid) -> float:
    best_t, best = 1.0, np.inf
    idx = np.arange(len(y))
    for t in grid:
        nll = -np.log(softmax(logits, t)[idx, y] + 1e-12).mean()
        if nll < best:
            best, best_t = nll, float(t)
    return best_t


def gate_scores(logits, feats, protos, temperature: float, mix: float) -> np.ndarray:
    conf = softmax(logits, temperature).max(1)
    return mix * conf + (1.0 - mix) * prototype_score(feats, protos)


def threshold_for_far(unknown_scores: np.ndarray, target_far: float) -> float:
    """Accept iff score > threshold. Picks the lowest threshold whose false-acceptance
    rate on these unknown scores is at most target_far."""
    s = np.sort(unknown_scores)[::-1]
    m = int(np.floor(target_far * len(s)))
    return float(s[min(m, len(s) - 1)])


def rejection_metrics(known_scores, known_correct, unknown_scores, threshold: float) -> dict:
    accept = known_scores > threshold
    labels = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
    scores = np.concatenate([known_scores, unknown_scores])
    return {
        "known_coverage": float(accept.mean()),
        "accepted_known_accuracy": float(known_correct[accept].mean()) if accept.any() else float("nan"),
        "unknown_false_acceptance": float((unknown_scores > threshold).mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "n_known": int(len(known_scores)),
        "n_unknown": int(len(unknown_scores)),
    }


def select_mix(val_logits, val_feats, unk_dev_logits, unk_dev_feats, protos, temperature, mix_grid, target_far):
    """Pick the mix with the highest known-validation coverage at the target unknown FAR
    measured on the unknown-development cohort."""
    best = (-1.0, None, None)
    for mix in mix_grid:
        kv = gate_scores(val_logits, val_feats, protos, temperature, mix)
        uv = gate_scores(unk_dev_logits, unk_dev_feats, protos, temperature, mix)
        thr = threshold_for_far(uv, target_far)
        cov = float((kv > thr).mean())
        if cov > best[0]:
            best = (cov, float(mix), thr)
    return {"mix": best[1], "coverage_val": best[0], "threshold_dev": best[2]}
