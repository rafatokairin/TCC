#!/usr/bin/env python
"""Build a higher-resolution image set from CBIS-DDSM for the SAME manifest ids.

Our manifest `image_id`s are CBIS-DDSM SeriesInstanceUIDs. In the awsaf49 Kaggle
layout the full-mammogram JPEGs live under `<cbis>/jpeg/<SeriesInstanceUID>/*.jpg`.
This script pulls each manifest id's JPEG at native resolution, orients it left,
resizes to the target size, and writes `data/dataset256/<image_id>.jpg` so the
existing patient-level manifest/splits are reused verbatim at higher resolution.

    python scripts/build_highres_dataset.py \
        --manifest data/manifest.csv \
        --cbis-jpeg-root <cbis>/jpeg \
        --out data/dataset256 --size 256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from breastsynth.data.preprocess import IMAGE_EXTS, orient_left
from PIL import Image


def _find_jpeg(jpeg_root: Path, image_id: str) -> Path | None:
    folder = jpeg_root / image_id
    if folder.is_dir():
        imgs = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if imgs:
            # full-mammogram series usually has a single image; take the largest
            return max(imgs, key=lambda p: p.stat().st_size)
    # fallback: a file named by the id anywhere under the root
    for p in jpeg_root.rglob(f"{image_id}*"):
        if p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--cbis-jpeg-root", required=True, help="<cbis>/jpeg directory")
    ap.add_argument("--out", default="data/dataset256")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jpeg_root = Path(args.cbis_jpeg_root)
    df = pd.read_csv(args.manifest)

    found, missing = 0, []
    for image_id in df["image_id"].astype(str):
        src = _find_jpeg(jpeg_root, image_id)
        if src is None:
            missing.append(image_id)
            continue
        img = Image.open(src).convert("L")
        img = orient_left(img)
        img.resize((args.size, args.size), Image.BILINEAR).save(out / f"{image_id}.jpg")
        found += 1

    print(f"Wrote {found}/{len(df)} images to {out} at {args.size}x{args.size}")
    if missing:
        print(f"  {len(missing)} ids not found under {jpeg_root} (first few: {missing[:5]})")


if __name__ == "__main__":
    main()
