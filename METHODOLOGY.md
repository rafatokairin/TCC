# Methodology — Single Source of Truth

This document defines the corrected experimental methodology for the study
*"Using Generative Models to Create Synthetic Datasets for Breast Cancer
Classification"*. It is the authoritative reference for both the code
(`src/breastsynth`) and the papers (`paper/lncs`, `paper/tcc`). If code and
paper ever disagree, this file wins; update it first.

---

## 0. Framing (Reviewer 1 — clinical claims; Reviewer 2 — novelty)

This is an **exploratory feasibility study** on synthetic data augmentation for
mammography classification, **not** evidence of clinical robustness. We make no
diagnostic-support claims. The contribution is a **leakage-free, reproducible
protocol** for measuring the effect of GAN-generated, perceptually-filtered
mammograms on a downstream classifier, together with an honest analysis of when
augmentation helps and when it hurts, backed by significance testing.

We follow the **CLAIM** checklist (Checklist for AI in Medical Imaging) and the
**TRIPOD+AI** and **PROBAST+AI** reporting/risk-of-bias guidelines for dataset
description, validation design, and reproducibility.

---

## 1. The data-leakage problem being fixed (Reviewer 1 — major)

**Original flaw.** The GAN (StyleGAN2-ADA) was trained **once** on all 553 real
images. A single synthetic set was then concatenated into **every** training
fold of a 5-fold CV (`legacy/trainingCNN/efficientb0.py:178`). Because the GAN
had seen images that later appeared in validation folds, the synthetic images
carried information about the validation data. In addition, the LPIPS filter
compared synthetic images against the **full** real set, including the very
images used later for validation. Both paths leak validation information into
training → optimistic, invalid estimates.

**Two independent leakage sources, both eliminated:**
1. *Generator leakage* — the GAN must never see any image used to evaluate the
   classifier.
2. *Selection leakage* — the LPIPS filter must never reference any image used to
   evaluate the classifier.

---

## 2. Experimental protocols

Two protocols are implemented. The **hold-out** protocol is the one reported in
the paper (feasible on a single RTX 4060); the **fold-wise** protocol is the
stricter gold standard, provided for reproduction with more compute. Both are
leakage-free.

### 2.1 Primary: strict patient-level hold-out (reported)

Reviewer 1 explicitly endorsed this: *"the final evaluation should be done on a
strictly separated hold-out test set."*

```
553 real images
   └─ patient-level, stratified split (GroupShuffleSplit on patient_id)
        ├─ DEV  (≈80%)         ← everything the pipeline may see
        │     ├─ GAN trained ONLY on DEV
        │     ├─ LPIPS filter references ONLY DEV reals
        │     └─ classifier training pool
        └─ TEST (≈20%)         ← locked; touched only for final metric
```

- The GAN and LPIPS filter see **only DEV**. TEST is never seen by any
  generative or selection step.
- For each synthetic:real ratio (0:1 baseline, 1:1, 2:1, 3:1, 4:1) the
  EfficientNet-B0 classifier is trained on `DEV_real + synthetic` and evaluated
  on the **locked TEST set**.
- Model selection / early stopping uses an inner validation split **carved out
  of DEV only** (never TEST).
- The whole classifier-training step is repeated over `N_SEEDS` (default 10)
  independent seeds. This yields a distribution of TEST metrics per ratio →
  confidence intervals and paired significance tests. The GAN is trained once on
  DEV (the generator is part of the training procedure; DEV is training data).
- **Patient-level** splitting guarantees no patient contributes images to both
  DEV and TEST (Reviewer 1 — patient separation).

### 2.2 Optional: fold-wise generation (gold standard)

```
patient-level Stratified GroupKFold (K=5) over the 553 images
  for each fold k:
     train_k / val_k   (disjoint patients)
     GAN retrained on train_k ONLY
     LPIPS filter references train_k reals ONLY
     synthetic_k generated from the fold-k generator
     classifier trained on train_k_real + synthetic_k
     classifier evaluated on val_k
  aggregate across folds
```

This retrains the GAN K times (expensive). It removes leakage by construction
and is the reference implementation for high-compute reproduction.

---

## 3. Dataset construction (Reviewer 1 — objective description)

**Source.** CBIS-DDSM (Curated Breast Imaging Subset of DDSM). NOTE: the
original monograph incorrectly cited "MIAS"; the image identifiers are DICOM SOP
Instance UIDs from CBIS-DDSM. This is corrected everywhere.

**Selection pipeline (`scripts/00_build_manifest.py`):**
1. Start from CBIS-DDSM mass + calcification case CSVs (train+test splits merged),
   which provide `patient_id`, `image view`, `pathology`, `image file path`,
   etc.
2. Keep **full-mammogram** images only; discard cropped-lesion images and ROI
   masks (the GAN learns global breast structure).
3. Map pathology → binary label: `MALIGNANT` → 1; `BENIGN` and
   `BENIGN_WITHOUT_CALLBACK` → 0.
4. **Near-duplicate removal** (Reviewer 1): perceptual-hash (pHash) + Hamming
   distance; images with distance ≤ `DEDUP_THRESHOLD` (default 4) collapsed to a
   single representative. Duplicates are logged.
5. Retain `patient_id` on every row for patient-level splitting.
6. Resulting manifest: `data/manifest.csv` with columns
   `image_id, patient_id, view, pathology, label, split_source, is_duplicate`.

**Class balancing (Reviewer 1 — why 553 vs 490, what happened to the 63):**
- The full-image, de-duplicated set contains 308 benign + 245 malignant = **553**
  images. All 553 are available to the GAN **within DEV** (the GAN benefits from
  the extra benign examples; class imbalance is acceptable for a generator).
- For the **classifier**, the two classes are balanced by **downsampling benign
  to 245** → a balanced pool of 490. The 63 benign images removed by balancing
  are **excluded from the classifier's real pool** but, in the hold-out
  protocol, are still allowed in the GAN's DEV training data **only if their
  patient is in DEV** — never if their patient is in TEST. This is stated
  explicitly and the exact per-split counts are logged to
  `results/<run>/split_report.json`.
- Balancing is done with a fixed seed and is patient-aware (we downsample whole
  patients where possible to avoid splitting a patient across the balance cut).

All counts (per class, per split, generated / retained / discarded synthetic per
class) are emitted to JSON so the paper tables are reproducible.

---

## 4. Generative model (Reviewers — implementation details, FID/KID)

- **Architecture:** conditional StyleGAN2-ADA (style-based generator, ADA on the
  discriminator). A conditional WGAN-GP baseline is retained for comparison.
- **Resolution:** 128×128 grayscale. Treated explicitly as a **limitation**
  (fine detail — margins, microcalcifications — may be lost; Reviewer 1).
- **Key hyperparameters** (see `configs/gan_stylegan2ada.yaml`): z_dim=256,
  w_dim=256, batch=8, epochs=400, lr_g=lr_d=0.0025, betas=(0.0,0.99),
  R1 γ=10 every 32 steps, ADA target=0.6, mixed precision. Seed logged.
- **Fidelity/diversity metrics** (Reviewer 1 & 2): **FID** and **KID** between
  real (DEV) and synthetic distributions, per class, reported with the LPIPS
  distribution. FID/KID complement LPIPS (which alone cannot detect low
  diversity).

---

## 5. Synthetic selection & LPIPS threshold (Reviewer 1 — justify 0.2)

- Each generated image's similarity is its **minimum** LPIPS distance to the
  reference real set (DEV, same class). Minimum (nearest match) is used to keep
  visually plausible samples while allowing diversity.
- **Threshold justification.** Instead of asserting 0.2, we run a **threshold
  sweep** τ ∈ {0.1, 0.15, 0.2, 0.25, 0.3} and report downstream TEST performance
  vs τ. We show 0.2 is a defensible operating point and discuss the
  fidelity–diversity trade-off: too-low τ risks **memorisation** (near-copies of
  training images); too-high τ admits low-fidelity samples.
- **Memorisation / near-neighbour check** (Reviewer 1): for retained synthetic
  images we compute the nearest real neighbour (LPIPS + pixel L2) and report the
  distribution + show qualitative nearest-neighbour panels. Synthetic images
  whose nearest neighbour is a near-copy (distance below `MEMORIZATION_EPS`) are
  flagged and counted. This directly addresses "low perceptual distance can
  indicate memorisation."

---

## 6. Classifier (Reviewer 1 — EfficientNet citation; details)

- **EfficientNet-B0**, ImageNet-pretrained; cite the **original EfficientNet
  paper (Tan & Le, 2019)**, not an applied paper.
- Input 128×128 RGB, ImageNet normalisation; head replaced by a single logit;
  `BCEWithLogitsLoss`; Adam lr=1e-3; batch=32; up to 30 epochs with early
  stopping on the inner-DEV validation split.
- Augmentation (train only): random horizontal flip, ±5° rotation. Full settings
  and seeds in `configs/classifier_efficientnet.yaml`.

---

## 7. Statistical analysis (Reviewer 2 — significance)

For each ratio we obtain `N_SEEDS` TEST metrics. Reported per ratio:
- mean ± std **and** 95% bootstrap confidence interval;
- **paired Wilcoxon signed-rank test** vs the real-only baseline (same seeds),
  with Holm correction across ratios;
- effect size (Cliff's delta);
- **DeLong test** for AUC differences on the pooled TEST predictions.
A result is only called an improvement if it is both consistent and
statistically significant after correction.

---

## 8. Reproducibility (Reviewer 1 — seeds, libraries, code/model availability)

- Global seeding of `random`, `numpy`, `torch` (+ cuDNN deterministic) via
  `breastsynth.seed`. Every artifact records its seed and full config hash.
- Pinned dependencies in `requirements.txt` / `pyproject.toml`.
- Every stage writes a JSON run-report (config, seeds, git commit, counts,
  metrics) under `results/<run_id>/`.
- Code is released under this repository; trained GAN checkpoint via Git LFS.
- CLI scripts `scripts/00…05` reproduce the full pipeline; `scripts/run_all.sh`
  chains them.

---

## 9. Stated limitations (Reviewer 1 & 2)

1. External validation is performed on **one** additional dataset (MIAS, via
   `scripts/06_external_validation.py`): the CBIS-DDSM-trained model is evaluated
   on MIAS with no retraining. Broader multi-institution/multi-scanner validation
   (INbreast, VinDr-Mammo) remains future work, and MIAS is digitised film so the
   comparison is indicative.
2. 128×128 resolution loses fine diagnostic detail.
3. Small dataset; moderate AUC; **not** clinically validated.
4. No prospective evaluation; no expert qualitative reading (future work).
5. GAN trained once on DEV in the primary protocol (fold-wise provided but not
   the reported result due to compute).

---

## 10. Result placeholders

The numbers in the papers are **regenerated** by running the pipeline on the
corrected protocol. Until re-run, tables carry `\TODO{...}` macros keyed to the
CSVs the code emits (`results/<run>/classification_summary.csv`,
`fidelity.json`, `lpips_threshold_sweep.csv`, `stats_tests.json`). The old
(leaky) numbers — acc 0.651 / F1 0.658 / AUC 0.703 at 2:1 — are **not** reused.
