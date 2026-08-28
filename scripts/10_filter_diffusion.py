#!/usr/bin/env python
"""Filter a folder of pre-generated diffusion images by LPIPS + audit memorisation.

The diffusion generator (scripts/09) produces a pool of images up front (unlike
the GAN's on-the-fly rejection sampling). This selects, per class, the images
whose minimum LPIPS distance to the DEV reals is below tau (most realistic),
caps at n_target, and audits memorisation (near-copies of DEV training images) —
critical for diffusion + few-shot LoRA, which can replicate training data.

    python scripts/10_filter_diffusion.py --config configs/full_cbis.yaml \
        --raw results_full/diffusion/raw --out results_full/diffusion/filtered \
        --tau 0.30 --n-target 780
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import lpips
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from breastsynth.config import load_config
from breastsynth.metrics.lpips_metrics import memorization_rate, nearest_real, to_ref_batch
from breastsynth.pipeline import prepare_holdout
from breastsynth.selection.lpips_filter import load_reference_reals

_NAMES = {0: "BENIGN", 1: "MALIGNANT"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/full_cbis.yaml")
    ap.add_argument("--raw", default="results_full/diffusion/raw")
    ap.add_argument("--out", default="results_full/diffusion/filtered")
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--n-target", type=int, default=780)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw)
    data = prepare_holdout(cfg)
    loss_fn = lpips.LPIPS(net=cfg.selection.lpips_net).to(args.device)
    size = cfg.data.image_size
    t = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))])

    report = {"tau": args.tau, "per_class": {}}
    for cls, name in _NAMES.items():
        reals = data.dev_full[data.dev_full[cfg.data.label_column] == cls]["path"].tolist()
        ref = to_ref_batch(load_reference_reals(reals, size, cfg.selection.n_reference_reals,
                                                seed=cfg.split.seed), args.device)
        files = sorted(raw.glob(f"{name}_*.png"))
        scored = []
        for f in tqdm(files, desc=f"filter {name}"):
            img = t(Image.open(f).convert("L")).unsqueeze(0)
            d, _ = nearest_real(loss_fn, img, ref, args.device)
            scored.append((d, f))
        scored.sort()
        kept = [(d, f) for d, f in scored if d < args.tau][: args.n_target]
        for i, (d, f) in enumerate(kept, 1):
            shutil.copy(f, out / f"{name}_{i:04d}.png")
        dists = np.array([d for d, _ in kept])
        report["per_class"][name] = {
            "pool": len(files), "retained": len(kept),
            "acceptance_rate": len(kept) / max(1, len(files)),
            "min_lpips_mean": float(dists.mean()) if len(dists) else float("nan"),
            "memorization": memorization_rate(dists, eps=cfg.selection.memorization_eps),
        }
        print(f"  {name}: kept {len(kept)}/{len(files)} (tau={args.tau}), "
              f"minLPIPS mean={report['per_class'][name]['min_lpips_mean']:.3f}, "
              f"memorised={report['per_class'][name]['memorization']['n_flagged']}")
    (out / "filter_report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved filtered diffusion set -> {out}")


if __name__ == "__main__":
    main()
