"""Leakage-free experimental protocols (see METHODOLOGY.md sections 2 & 7).

`run_holdout_protocol` is the primary, reported protocol:
  * TEST is locked; only DEV feeds training and synthetic selection;
  * for each synthetic:real ratio and each of N seeds, a classifier is trained on
    (inner-DEV real + synthetic) with early stopping on an inner-DEV val split,
    then evaluated on the locked TEST set;
  * per-ratio metrics get bootstrap CIs; each ratio is compared to the real-only
    baseline with a paired Wilcoxon test (Holm-corrected) + Cliff's delta, and
    AUCs are compared with DeLong on the seed-ensembled TEST predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from breastsynth.config import Config
from breastsynth.data.splits import inner_validation_split
from breastsynth.evaluation.stats import (
    cliffs_delta,
    delong_auc_test,
    holm_correction,
    wilcoxon_vs_baseline,
)
from breastsynth.metrics.classification import bootstrap_ci, classification_metrics

# NOTE: torch-dependent imports (models/training/dataset) are done lazily inside
# run_holdout_protocol so that compose_training_set and the split logic — and
# their tests — import without requiring the torch stack.


def compose_training_set(
    real_df: pd.DataFrame,
    synthetic_by_class: dict[int, list[str]],
    ratio: float,
    seed: int,
    label_col: str = "label",
) -> tuple[list[str], list[int]]:
    """Real training images + `ratio` x (#real per class) synthetic images.

    ratio=0 -> real only. Synthetic images are sampled (seeded, without
    replacement, capped at availability) per class so classes stay balanced.
    """
    rng = np.random.RandomState(seed)
    paths = list(real_df["path"])
    labels = list(real_df[label_col].astype(int))
    if ratio <= 0:
        return paths, labels

    for cls, pool in synthetic_by_class.items():
        n_real_cls = int((real_df[label_col] == cls).sum())
        n_syn = min(int(round(ratio * n_real_cls)), len(pool))
        if n_syn <= 0:
            continue
        chosen = rng.choice(pool, size=n_syn, replace=False)
        paths.extend(chosen.tolist())
        labels.extend([int(cls)] * n_syn)
    return paths, labels


def _test_loader(test_df, cfg, device):
    from torch.utils.data import DataLoader

    from breastsynth.data.dataset import MammogramDataset, build_transforms

    ds = MammogramDataset.from_frame(test_df, build_transforms(cfg.classifier.image_size, train=False))
    return DataLoader(ds, batch_size=cfg.classifier.batch_size, shuffle=False)


def run_holdout_protocol(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    synthetic_by_class: dict[int, list[str]],
    cfg: Config,
    device: str = "cuda",
) -> dict:
    from breastsynth.models.classifier import predict_probs
    from breastsynth.training.train_classifier import train_classifier

    test_loader = _test_loader(test_df, cfg, device)
    test_labels = test_df[cfg.data.label_column].astype(int).tolist()

    # per_ratio_metrics[ratio][metric] = list over seeds
    per_ratio_metrics: dict[float, dict[str, list[float]]] = {}
    # ensemble_probs[ratio] = mean positive-prob per test sample across seeds
    ensemble_probs: dict[float, np.ndarray] = {}

    import time

    n_models = len(cfg.classifier.ratios) * cfg.split.n_seeds
    done = 0
    t0 = time.time()
    for ratio in cfg.classifier.ratios:
        tag = "real-only" if ratio == 0 else f"{ratio:g}:1"
        acc, f1, auc = [], [], []
        prob_stack = []
        for s in range(cfg.split.n_seeds):
            seed = cfg.split.seed + s
            inner_train, inner_val = inner_validation_split(
                dev_df,
                cfg.classifier.inner_val_size,
                cfg.data.label_column,
                cfg.data.patient_column,
                seed=seed,
            )
            tr_paths, tr_labels = compose_training_set(
                inner_train, synthetic_by_class, ratio, seed, cfg.data.label_column
            )
            model, _ = train_classifier(
                tr_paths,
                tr_labels,
                inner_val["path"].tolist(),
                inner_val[cfg.data.label_column].astype(int).tolist(),
                cfg.classifier,
                device=device,
                seed=seed,
            )
            yt, yp = predict_probs(model, test_loader, device)
            m = classification_metrics(yt, yp)
            acc.append(m["accuracy"])
            f1.append(m["f1"])
            auc.append(m["auc"])
            prob_stack.append(yp)
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (n_models - done)
            print(
                f"[{done:>2}/{n_models}] ratio {tag:>9} seed {s + 1}/{cfg.split.n_seeds} "
                f"| test auc={m['auc']:.3f} f1={m['f1']:.3f} "
                f"| elapsed {elapsed / 60:.1f}m eta {eta / 60:.1f}m",
                flush=True,
            )
        per_ratio_metrics[ratio] = {"accuracy": acc, "f1": f1, "auc": auc}
        ensemble_probs[ratio] = np.mean(np.array(prob_stack), axis=0)

    return _summarise(per_ratio_metrics, ensemble_probs, np.asarray(test_labels), cfg)


def _summarise(per_ratio_metrics, ensemble_probs, test_labels, cfg: Config) -> dict:
    baseline = 0.0
    summary = {"per_ratio": {}, "stats": {}}
    for ratio, metrics in per_ratio_metrics.items():
        entry = {}
        for name, vals in metrics.items():
            mean, lo, hi = bootstrap_ci(vals, seed=cfg.split.seed)
            entry[name] = {
                "mean": mean,
                "std": float(np.std(vals)),
                "ci95": [lo, hi],
                "per_seed": vals,
            }
        summary["per_ratio"][str(ratio)] = entry

    # Paired Wilcoxon vs baseline + Cliff's delta, Holm-corrected across ratios.
    if baseline in per_ratio_metrics:
        base_metrics = per_ratio_metrics[baseline]
        for metric in ("accuracy", "f1", "auc"):
            raw_p = {}
            details = {}
            for ratio, metrics in per_ratio_metrics.items():
                if ratio == baseline:
                    continue
                w = wilcoxon_vs_baseline(base_metrics[metric], metrics[metric])
                d = cliffs_delta(base_metrics[metric], metrics[metric])
                raw_p[str(ratio)] = w["p_value"]
                details[str(ratio)] = {**w, "cliffs_delta": d}
            corrected = holm_correction(raw_p)
            for ratio_key, c in corrected.items():
                details[ratio_key].update(c)
            summary["stats"][f"wilcoxon_{metric}"] = details

        # DeLong on seed-ensembled TEST predictions (each ratio vs baseline).
        delong = {}
        base_probs = ensemble_probs[baseline]
        for ratio, probs in ensemble_probs.items():
            if ratio == baseline:
                continue
            delong[str(ratio)] = delong_auc_test(test_labels, probs, base_probs)
        summary["stats"]["delong_auc_vs_baseline"] = delong

    return summary
