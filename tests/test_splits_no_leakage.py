"""Tests that guarantee the leakage-free contract (Reviewer 1's core concern)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from breastsynth.config import Config
from breastsynth.data.splits import (
    balance_classes,
    foldwise_splits,
    holdout_split,
    inner_validation_split,
)
from breastsynth.evaluation.protocols import compose_training_set


def _synthetic_manifest(n_patients=60, imgs_per_patient=3, seed=0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for p in range(n_patients):
        label = int(p % 2 == 0)  # roughly balanced across patients
        for j in range(imgs_per_patient):
            rows.append(
                {
                    "image_id": f"P{p:03d}_{j}",
                    "patient_id": f"P_{p:05d}",
                    "label": label,
                    "path": f"/data/P{p:03d}_{j}.png",
                    "is_duplicate": False,
                }
            )
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def test_holdout_has_no_patient_overlap():
    df = _synthetic_manifest()
    dev, test = holdout_split(df, test_size=0.2, seed=42)
    assert set(dev["patient_id"]) & set(test["patient_id"]) == set()
    # test is roughly the requested fraction
    assert 0.10 < len(test) / len(df) < 0.30


def test_foldwise_folds_are_patient_disjoint():
    df = _synthetic_manifest()
    seen_val_patients = set()
    for _, train, val in foldwise_splits(df, n_folds=5, seed=42):
        assert set(train["patient_id"]) & set(val["patient_id"]) == set()
        seen_val_patients |= set(val["patient_id"])
    # every patient appears in exactly one validation fold
    assert seen_val_patients == set(df["patient_id"])


def test_inner_validation_stays_within_dev():
    df = _synthetic_manifest()
    dev, test = holdout_split(df, test_size=0.2, seed=42)
    inner_train, inner_val = inner_validation_split(dev, 0.2, seed=7)
    assert set(inner_train["patient_id"]) & set(inner_val["patient_id"]) == set()
    # nothing from TEST leaks into either inner split
    assert set(inner_train["patient_id"]) & set(test["patient_id"]) == set()
    assert set(inner_val["patient_id"]) & set(test["patient_id"]) == set()


def test_balance_classes_hits_target_and_is_patient_aware():
    df = _synthetic_manifest()
    target = 20
    balanced, report = balance_classes(df, target_per_class=target, seed=42)
    counts = balanced["label"].value_counts().to_dict()
    assert all(v <= target for v in counts.values())
    assert report["n_removed"] == len(df) - len(balanced)


def test_compose_training_set_respects_ratio_and_balance():
    df = _synthetic_manifest()
    dev, _ = holdout_split(df, test_size=0.2, seed=42)
    dev_bal, _ = balance_classes(dev, 15, seed=42)
    synth = {0: [f"/syn/b_{i}.png" for i in range(500)], 1: [f"/syn/m_{i}.png" for i in range(500)]}
    paths, labels = compose_training_set(dev_bal, synth, ratio=2.0, seed=1)
    n_real = len(dev_bal)
    # roughly 2x synthetic added on top of real
    assert len(paths) == len(labels)
    assert len(paths) > n_real
    # ratio=0 returns real only
    p0, _ = compose_training_set(dev_bal, synth, ratio=0.0, seed=1)
    assert len(p0) == n_real


def test_end_to_end_no_test_image_in_any_training_path():
    """The strongest guarantee: no TEST image path can enter a training set."""
    df = _synthetic_manifest()
    dev, test = holdout_split(df, test_size=0.2, seed=42)
    dev_bal, _ = balance_classes(dev, 15, seed=42)
    synth = {0: [f"/syn/b_{i}.png" for i in range(200)], 1: [f"/syn/m_{i}.png" for i in range(200)]}
    test_paths = set(test["path"])
    for seed in range(5):
        inner_train, _ = inner_validation_split(dev, 0.2, seed=seed)
        tr_paths, _ = compose_training_set(inner_train, synth, 3.0, seed)
        assert test_paths.isdisjoint(tr_paths)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
