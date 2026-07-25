"""Sanity tests for the statistical machinery (Reviewer 2: significance analysis)."""
from __future__ import annotations

import numpy as np
import pytest

from breastsynth.evaluation.stats import (
    cliffs_delta,
    delong_auc_test,
    holm_correction,
    wilcoxon_vs_baseline,
)
from breastsynth.metrics.classification import bootstrap_ci, classification_metrics


def test_wilcoxon_detects_consistent_improvement():
    base = [0.60, 0.61, 0.59, 0.62, 0.60, 0.61, 0.60, 0.59, 0.61, 0.60]
    cand = [b + 0.05 for b in base]
    res = wilcoxon_vs_baseline(base, cand)
    assert res["p_value"] < 0.05
    assert res["mean_diff"] > 0


def test_wilcoxon_no_difference_is_not_significant():
    base = [0.60, 0.61, 0.59, 0.62, 0.60]
    res = wilcoxon_vs_baseline(base, list(base))
    assert res["p_value"] >= 0.05


def test_cliffs_delta_bounds_and_sign():
    assert cliffs_delta([0, 0, 0], [1, 1, 1]) == pytest.approx(1.0)
    assert cliffs_delta([1, 1, 1], [0, 0, 0]) == pytest.approx(-1.0)
    assert -1.0 <= cliffs_delta([1, 2, 3], [2, 2, 2]) <= 1.0


def test_holm_correction_monotone_and_rejects():
    corrected = holm_correction({"a": 0.001, "b": 0.04, "c": 0.5})
    assert corrected["a"]["p_adjusted"] <= corrected["b"]["p_adjusted"] <= corrected["c"]["p_adjusted"]
    assert corrected["a"]["reject"] is True
    assert corrected["c"]["reject"] is False


def test_delong_separable_beats_random():
    rng = np.random.RandomState(0)
    y = np.array([0] * 50 + [1] * 50)
    good = np.concatenate([rng.uniform(0, 0.4, 50), rng.uniform(0.6, 1.0, 50)])
    rand = rng.uniform(0, 1, 100)
    res = delong_auc_test(y, good, rand)
    assert res["auc_a"] > res["auc_b"]
    assert res["p_value"] < 0.05


def test_bootstrap_ci_contains_mean():
    vals = [0.60, 0.62, 0.61, 0.63, 0.59, 0.60, 0.61, 0.62]
    mean, lo, hi = bootstrap_ci(vals, n_boot=2000, seed=0)
    assert lo <= mean <= hi


def test_classification_metrics_perfect():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["auc"] == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
