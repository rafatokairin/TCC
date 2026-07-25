"""External-validation datasets (Reviewer 2: validation on additional datasets).

Builds a manifest for a *second* dataset so a CBIS-DDSM-trained classifier can be
evaluated on it without retraining on it. Images are preprocessed the same way as
training data (resize + orient-left) so the domains are comparable.

Supported:
  * MIAS / mini-MIAS (`kmader/mias-mammography`): PGM images + Info.txt. Label is
    malignant (severity 'M') = 1, everything else (benign 'B', normal 'NORM') = 0.
  * generic: a folder of images + a labels CSV (columns: image, label).

INbreast is registered as a slug but its images are DICOM (needs `pydicom`); a
loader can be added on request.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from breastsynth.data.preprocess import preprocess_folder

IMAGE_EXTS = (".pgm", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _find_file(root: Path, names: tuple[str, ...]) -> Path | None:
    for p in root.rglob("*"):
        if p.name.lower() in names:
            return p
    return None


def _parse_mias_info(info_path: Path, abnormal_only: bool = True) -> dict[str, int]:
    """Parse MIAS Info.txt -> {refnum: label}. Malignant (M) = 1, Benign (B) = 0.

    Lines look like: 'mdb001 G CIRC B 535 425 197'; NORM lines have no severity.
    With `abnormal_only=True` (default) NORM films are EXCLUDED so the external
    set matches the CBIS-DDSM task (benign-lesion vs malignant-lesion) rather than
    mixing in normal, lesion-free mammograms. If a film has multiple
    abnormalities, malignant wins.
    """
    labels: dict[str, int] = {}
    for line in info_path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if not parts or not parts[0].lower().startswith("mdb"):
            continue
        ref = parts[0].lower()
        severity = parts[3].upper() if len(parts) > 3 else ""
        if severity not in ("B", "M"):
            if not abnormal_only:
                labels.setdefault(ref, 0)  # NORM -> benign/non-malignant
            continue
        label = 1 if severity == "M" else 0
        labels[ref] = max(labels.get(ref, 0), label)
    return labels


def build_mias_manifest(root: str | Path, work_dir: str | Path, size: int = 128) -> pd.DataFrame:
    root = Path(root)
    info = _find_file(root, ("info.txt",))
    if info is None:
        raise FileNotFoundError(f"MIAS Info.txt not found under {root}")
    labels = _parse_mias_info(info)

    # Locate the .pgm image directory.
    pgms = [p for p in root.rglob("*.pgm")]
    if not pgms:
        raise FileNotFoundError(f"No .pgm images found under {root}")
    src_dir = pgms[0].parent

    work = Path(work_dir)
    img_out = work / "images"
    report = preprocess_folder(src_dir, img_out, size=size, to_grayscale=True)
    print(f"MIAS preprocess: {report}")

    rows = []
    for p in sorted(img_out.glob("*.jpg")):
        ref = p.stem.lower()
        if ref not in labels:
            continue
        rows.append(
            {
                "image_id": ref,
                "patient_id": ref,  # one film == one subject in MIAS
                "view": None,
                "pathology": "MALIGNANT" if labels[ref] == 1 else "BENIGN",
                "label": labels[ref],
                "path": str(p),
                "is_duplicate": False,
            }
        )
    if not rows:
        raise ValueError("No MIAS images matched Info.txt labels.")
    df = pd.DataFrame(rows)
    (work / "manifest_external.csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(work / "manifest_external.csv", index=False)
    return df


def build_generic_manifest(
    images_dir: str | Path,
    labels_csv: str | Path,
    work_dir: str | Path,
    size: int = 128,
    image_col: str = "image",
    label_col: str = "label",
) -> pd.DataFrame:
    """Folder of images + a CSV with (image, label). Labels: 1=malignant, 0=other."""
    labels_df = pd.read_csv(labels_csv)
    labels_df.columns = [c.strip().lower() for c in labels_df.columns]
    lab = {Path(str(r[image_col])).stem: int(r[label_col]) for _, r in labels_df.iterrows()}

    work = Path(work_dir)
    img_out = work / "images"
    preprocess_folder(images_dir, img_out, size=size, to_grayscale=True)

    rows = []
    for p in sorted(img_out.glob("*.jpg")):
        if p.stem not in lab:
            continue
        rows.append(
            {
                "image_id": p.stem,
                "patient_id": p.stem,
                "view": None,
                "pathology": "MALIGNANT" if lab[p.stem] == 1 else "BENIGN",
                "label": lab[p.stem],
                "path": str(p),
                "is_duplicate": False,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(work / "manifest_external.csv", index=False)
    return df


def build_external_manifest(name: str, root: str | Path, work_dir: str | Path, size: int = 128):
    """Dispatch to the loader for `name` ('mias' supported; others -> hint)."""
    name = name.lower()
    if name == "mias":
        return build_mias_manifest(root, work_dir, size)
    if name == "inbreast":
        raise NotImplementedError(
            "INbreast images are DICOM (needs pydicom). Ask to add an INbreast "
            "loader, or preprocess to a folder of JPEGs + labels CSV and use "
            "build_generic_manifest()."
        )
    warnings.warn(f"Unknown dataset '{name}'; use build_generic_manifest() instead.", stacklevel=2)
    raise NotImplementedError(name)
