"""Statistical significance testing (Reviewer 2: "no statistical significance analysis").

Provides:
  * paired Wilcoxon signed-rank test of a ratio vs the real-only baseline
    across the repeated-seed metrics;
  * Cliff's delta effect size;
  * Holm-Bonferroni correction across the multiple ratio comparisons;
  * DeLong's test for comparing two correlated ROC AUCs on the same test set.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def wilcoxon_vs_baseline(baseline: list[float], candidate: list[float]) -> dict:
    """Paired Wilcoxon signed-rank test (candidate vs baseline), same seeds/order."""
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if len(baseline) != len(candidate):
        raise ValueError("Paired test requires equal-length, aligned samples.")
    diff = candidate - baseline
    if np.allclose(diff, 0):
        return {"statistic": float("nan"), "p_value": 1.0, "median_diff": 0.0}
    res = stats.wilcoxon(candidate, baseline, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "median_diff": float(np.median(diff)),
        "mean_diff": float(diff.mean()),
    }


def cliffs_delta(baseline: list[float], candidate: list[float]) -> float:
    """Cliff's delta effect size in [-1, 1]. Positive => candidate tends higher."""
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(baseline, dtype=float)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return float((gt - lt) / (len(a) * len(b)))


def holm_correction(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni correction across a family of comparisons.

    Returns {name: {p_raw, p_adjusted, reject}}.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    prev = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - rank) * p))
        prev = adj
        out[name] = {"p_raw": float(p), "p_adjusted": float(adj), "reject": bool(adj < alpha)}
    return out


# ---------------------------------------------------------------------------
# DeLong's test for two correlated AUCs (Sun & Xu 2014 fast implementation).
# ---------------------------------------------------------------------------
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    tie = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        tie[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = tie
    return out


def _fast_delong(preds_sorted: np.ndarray, n_pos: int):
    m, n = n_pos, preds_sorted.shape[1] - n_pos
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(preds_sorted[r])
    auc = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return auc, delongcov


def delong_auc_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    """DeLong test comparing AUC of model A vs B on the SAME samples.

    Returns {"auc_a", "auc_b", "z", "p_value"} (two-sided).
    """
    y_true = np.asarray(y_true)
    order = (-y_true).argsort(kind="mergesort")
    label_1_count = int(y_true.sum())
    preds = np.vstack((np.asarray(prob_a), np.asarray(prob_b)))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z, p = float("nan"), 1.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float(z), "p_value": float(p)}
