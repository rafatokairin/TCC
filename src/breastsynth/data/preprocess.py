"""Raw-image preprocessing: resize to NxN and orient every breast to the LEFT.

Reproduces the manual preprocessing step used to build `data/dataset128/`:
full mammograms are resized and horizontally flipped so the breast always points
left, which removes the left/right laterality cue so the models learn lesion
structure rather than anatomical side (important given the small dataset).

Laterality is taken from metadata when available (a `left or right breast` /
`laterality` column, or an `L`/`R` token in the filename); otherwise it is
inferred from the image itself (the breast is the bright tissue mass, so the
half-image with the larger foreground area is the breast side).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pgm", ".ppm", ".pbm")


def infer_laterality(img: Image.Image, fg_threshold: int = 20) -> str:
    """Return 'L' or 'R' — the side where the breast (bright foreground) sits."""
    g = np.asarray(img.convert("L"))
    w = g.shape[1]
    left = (g[:, : w // 2] > fg_threshold).sum()
    right = (g[:, w // 2 :] > fg_threshold).sum()
    return "L" if left >= right else "R"


def orient_left(img: Image.Image, laterality: str | None = None) -> Image.Image:
    """Flip horizontally if the breast is on the right, so it ends up on the left."""
    side = (laterality or infer_laterality(img)).strip().upper()[:1]
    if side == "R":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def preprocess_image(
    path: str | Path,
    size: int = 128,
    laterality: str | None = None,
    to_grayscale: bool = True,
) -> Image.Image:
    """Load -> orient-left -> resize to (size, size)."""
    img = Image.open(path)
    if to_grayscale:
        img = img.convert("L")
    img = orient_left(img, laterality)
    return img.resize((size, size), Image.BILINEAR)


def preprocess_folder(
    src: str | Path,
    dst: str | Path,
    size: int = 128,
    laterality_map: dict[str, str] | None = None,
    to_grayscale: bool = True,
) -> dict:
    """Preprocess every image in `src` into `dst`.

    Args:
        laterality_map: optional {image_id (stem) -> 'L'/'R'} from metadata; when
            an id is missing, laterality is inferred from the image.

    Returns a small report (counts and how many were flipped).
    """
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    laterality_map = laterality_map or {}
    n, n_flipped = 0, 0
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        lat = laterality_map.get(p.stem)
        img = Image.open(p)
        if to_grayscale:
            img = img.convert("L")
        side = (lat or infer_laterality(img)).strip().upper()[:1]
        if side == "R":
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            n_flipped += 1
        img.resize((size, size), Image.BILINEAR).save(dst / f"{p.stem}.jpg")
        n += 1
    return {"processed": n, "flipped_to_left": n_flipped, "size": size, "grayscale": to_grayscale}
