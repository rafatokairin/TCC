#!/usr/bin/env python
"""External validation (Reviewer 2): evaluate a CBIS-DDSM-trained model on a
DIFFERENT dataset, with no retraining on it.

It reuses the exact leakage-free hold-out protocol, but swaps the locked TEST set
for the external dataset: for each synthetic:real ratio and seed, the classifier
is trained on CBIS-DDSM DEV (real + synthetic) and evaluated on the external set.
Reports per-ratio metrics with 95% CIs plus Wilcoxon/Cliff's-delta/DeLong vs the
real-only baseline — the standard external-generalisation check.

    # 1) download + build external manifest (MIAS shown)
    python scripts/download_data.py --dataset mias        # prints <path>
    python scripts/06_external_validation.py --config configs/default.yaml \
        --dataset mias --root <path> --synthetic results/synthetic \
        --out results/external_mias

Auto-download: pass --download to fetch <dataset> via kagglehub automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from breastsynth.config import load_config
from breastsynth.data.external import build_external_manifest
from breastsynth.evaluation.protocols import run_holdout_protocol
from breastsynth.metrics.fidelity import list_images
from breastsynth.pipeline import prepare_holdout
from breastsynth.seed import seed_everything

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
    ap.add_argument("--dataset", required=True, help="external dataset name (e.g. mias)")
    ap.add_argument("--root", default=None, help="local path of the external dataset")
    ap.add_argument("--download", action="store_true", help="download via kagglehub first")
    ap.add_argument("--synthetic", default="results/synthetic")
    ap.add_argument("--out", default="results/external")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.split.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    root = args.root
    if args.download or root is None:
        from breastsynth.data.download import download_kaggle

        root = download_kaggle(args.dataset)

    # Build the external manifest (preprocessed the same way as training data).
    ext = build_external_manifest(args.dataset, root, out / "external_data", size=cfg.data.image_size)
    print(f"External set '{args.dataset}': {len(ext)} images, "
          f"class counts {ext[cfg.data.label_column].value_counts().to_dict()}")

    # CBIS-DDSM DEV as the training pool; external set as the evaluation set.
    data = prepare_holdout(cfg)
    synthetic_by_class = _load_synthetic_by_class(args.synthetic)

    summary = run_holdout_protocol(
        data.dev_balanced, ext, synthetic_by_class, cfg, device=args.device
    )
    (out / "external_summary.json").write_text(json.dumps(summary, indent=2))

    rows = []
    for ratio, m in summary["per_ratio"].items():
        rows.append(
            {"ratio": ratio, "accuracy_mean": m["accuracy"]["mean"], "f1_mean": m["f1"]["mean"],
             "auc_mean": m["auc"]["mean"], "auc_ci_lo": m["auc"]["ci95"][0], "auc_ci_hi": m["auc"]["ci95"][1]}
        )
    pd.DataFrame(rows).to_csv(out / "external_summary.csv", index=False)

    print(f"\n=== External ({args.dataset}) results (mean [95% CI]) ===")
    for ratio, m in summary["per_ratio"].items():
        tag = "real-only" if float(ratio) == 0 else f"{int(float(ratio))}:1"
        print(f"  {tag:>9}: acc={m['accuracy']['mean']:.3f} f1={m['f1']['mean']:.3f} auc={m['auc']['mean']:.3f}")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
