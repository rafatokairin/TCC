#!/usr/bin/env python
"""Build the FULL CBIS-DDSM full-mammogram manifest (~2857 images, ~1460 patients).

The 553-image subset used previously severely limited statistical power (locked
TEST n=98). This assembles the complete full-mammogram set from the awsaf49 Kaggle
layout, with correct patient ids and pathology, so the leakage-free study can be
run with real power.

Join logic (robust to case-name suffixes):
  * case-description CSVs give (patient, laterality, view) -> pathology
    (a full mammogram is MALIGNANT if ANY of its lesions is malignant);
  * dicom_info.csv 'full mammogram images' rows give
    (patient, laterality, view) -> SeriesInstanceUID (= image_id) + jpeg path.
Images are oriented-left and resized; a manifest with patient_id/label/path is
written and near-duplicates are pHash-flagged.

    python scripts/build_full_cbis_manifest.py \
        --cbis-root ~/.cache/kagglehub/.../versions/1 \
        --out-images data/dataset_full256 --size 256 --out data/manifest_full.csv
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from breastsynth.data.dedup import flag_near_duplicates
from breastsynth.data.preprocess import orient_left

_PID = re.compile(r"(P_\d{5})")
_LABEL = {"MALIGNANT": 1, "BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0}


def _norm(c):
    return c.strip().lower()


def _key(patient, lat, view):
    return f"{patient}|{str(lat).strip().upper()[:1]}|{str(view).strip().upper()}"


def _case_labels(cbis_root: Path) -> dict[str, int]:
    """(patient|L/R|view) -> label (malignant if any lesion malignant)."""
    labels: dict[str, int] = {}
    for c in glob.glob(str(cbis_root / "csv" / "*case_description*.csv")):
        df = pd.read_csv(c)
        df.columns = [_norm(x) for x in df.columns]
        for _, r in df.iterrows():
            m = _PID.search(str(r.get("patient_id", "")))
            if not m:
                continue
            k = _key(m.group(1), r.get("left or right breast"), r.get("image view"))
            lab = _LABEL.get(str(r.get("pathology", "")).strip().upper())
            if lab is None:
                continue
            labels[k] = max(labels.get(k, 0), lab)
    return labels


def _full_images(cbis_root: Path) -> pd.DataFrame:
    """Full-mammogram rows from dicom_info -> (key, image_id, jpeg path)."""
    di = pd.read_csv(cbis_root / "csv" / "dicom_info.csv")
    di = di[di["SeriesDescription"].astype(str).str.contains("full", case=False, na=False)].copy()
    rows = []
    for _, r in di.iterrows():
        name = str(r.get("PatientID", ""))
        m = _PID.search(name)
        if not m:
            continue
        lat = "RIGHT" if "RIGHT" in name.upper() else ("LEFT" if "LEFT" in name.upper() else "")
        view = "MLO" if "MLO" in name.upper() else ("CC" if "CC" in name.upper() else "")
        img_path = str(r.get("image_path", ""))
        # image_path like 'CBIS-DDSM/jpeg/<seriesUID>/1-1.jpg'; series UID is the folder.
        parts = [p for p in re.split(r"[/\\]", img_path) if p]
        series = parts[-2] if len(parts) >= 2 else ""
        rows.append({"key": _key(m.group(1), lat, view), "patient_id": m.group(1),
                     "image_id": series, "rel": img_path})
    return pd.DataFrame(rows).drop_duplicates("image_id")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cbis-root", required=True)
    ap.add_argument("--out-images", default="data/dataset_full256")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="data/manifest_full.csv")
    ap.add_argument("--dedup-threshold", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.cbis_root).expanduser()
    labels = _case_labels(root)
    imgs = _full_images(root)
    imgs["label"] = imgs["key"].map(labels)
    imgs = imgs.dropna(subset=["label"]).copy()
    imgs["label"] = imgs["label"].astype(int)
    print(f"matched {len(imgs)} full mammograms to labels "
          f"({imgs['patient_id'].nunique()} patients); dist {imgs['label'].value_counts().to_dict()}")

    out_dir = Path(args.out_images)
    out_dir.mkdir(parents=True, exist_ok=True)
    jpeg_root = root / "jpeg"
    kept = []
    for r in imgs.itertuples():
        folder = jpeg_root / r.image_id
        cand = sorted(folder.glob("*.jpg")) if folder.is_dir() else []
        if not cand:
            hits = list(jpeg_root.rglob(f"{r.image_id}/*.jpg"))
            cand = hits[:1]
        if not cand:
            continue
        src = max(cand, key=lambda p: p.stat().st_size)
        img = orient_left(Image.open(src).convert("L")).resize((args.size, args.size), Image.BILINEAR)
        dst = out_dir / f"{r.image_id}.jpg"
        img.save(dst)
        kept.append({"image_id": r.image_id, "patient_id": r.patient_id,
                     "view": None, "pathology": "MALIGNANT" if r.label else "BENIGN",
                     "label": r.label, "path": str(dst)})
    man = pd.DataFrame(kept)
    print(f"wrote {len(man)} images to {out_dir} at {args.size}px")

    dup = flag_near_duplicates({m.image_id: m.path for m in man.itertuples()},
                               threshold=args.dedup_threshold)
    man = man.merge(dup[["image_id", "phash", "cluster_id", "is_duplicate"]], on="image_id")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    man.sort_values("image_id").to_csv(args.out, index=False)
    print(f"manifest -> {args.out}  (near-dup flagged: {int(man['is_duplicate'].sum())} | "
          f"patients: {man['patient_id'].nunique()})")


if __name__ == "__main__":
    main()
