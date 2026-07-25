#!/usr/bin/env python
"""Run the leakage-free classification experiment (hold-out protocol).

For each synthetic:real ratio and each seed, trains EfficientNet-B0 on
(inner-DEV real + synthetic) and evaluates on the LOCKED TEST set. Emits
per-ratio metrics with 95% CIs plus Wilcoxon/Cliff's-delta/DeLong significance.

    python scripts/04_run_classification.py --config configs/default.yaml \
        --synthetic results/synthetic --out results/classification
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from breastsynth.config import load_config
from breastsynth.evaluation.protocols import run_holdout_protocol
from breastsynth.metrics.fidelity import list_images
from breastsynth.pipeline import prepare_holdout
from breastsynth.runlog import RunReport
from breastsynth.seed import seed_everything
from breastsynth.viz.plots import plot_ratio_results

_CLASS_NAMES = {0: "BENIGN", 1: "MALIGNANT"}


def _load_synthetic_by_class(folder: str) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {c: [] for c in _CLASS_NAMES}
    for p in list_images(folder):
        for cls, name in _CLASS_NAMES.items():
            if Path(p).stem.upper().startswith(name):
                out[cls].append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--synthetic", default="results/synthetic")
    ap.add_argument("--out", default="results/classification")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.split.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = prepare_holdout(cfg)
    synthetic_by_class = _load_synthetic_by_class(args.synthetic)
    print("Synthetic available:",
          {_CLASS_NAMES[c]: len(v) for c, v in synthetic_by_class.items()})
    print(f"DEV real (balanced): {len(data.dev_balanced)}  TEST (balanced): {len(data.test_balanced)}")

    report = RunReport("classification_holdout", cfg.to_dict(), cfg.results_dir, f"cls_{cfg.hash()}")
    report.set("split_report", data.balance_report)

    summary = run_holdout_protocol(
        data.dev_balanced, data.test_balanced, synthetic_by_class, cfg, device=args.device
    )

    # Flat CSV for the paper tables.
    rows = []
    for ratio, m in summary["per_ratio"].items():
        rows.append(
            {
                "ratio": ratio,
                "accuracy_mean": m["accuracy"]["mean"], "accuracy_ci_lo": m["accuracy"]["ci95"][0],
                "accuracy_ci_hi": m["accuracy"]["ci95"][1],
                "f1_mean": m["f1"]["mean"], "f1_ci_lo": m["f1"]["ci95"][0], "f1_ci_hi": m["f1"]["ci95"][1],
                "auc_mean": m["auc"]["mean"], "auc_ci_lo": m["auc"]["ci95"][0], "auc_ci_hi": m["auc"]["ci95"][1],
            }
        )
    pd.DataFrame(rows).to_csv(out / "classification_summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    report.add_metric("summary", summary)
    report.save()

    for metric in ("auc", "f1", "accuracy"):
        plot_ratio_results(summary, out / f"ratio_{metric}.png", metric=metric)

    print("\n=== TEST results (mean [95% CI]) ===")
    for ratio, m in summary["per_ratio"].items():
        tag = "real-only" if float(ratio) == 0 else f"{int(float(ratio))}:1"
        print(f"  {tag:>9}: acc={m['accuracy']['mean']:.3f} "
              f"f1={m['f1']['mean']:.3f} auc={m['auc']['mean']:.3f}")
    print(f"\nSaved to {out} and results/cls_{cfg.hash()}/")


if __name__ == "__main__":
    main()
