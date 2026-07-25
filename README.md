# Leakage-Free Synthetic Data Augmentation for Breast Cancer Mammography

Exploratory feasibility study on whether GAN-generated, perceptually-filtered
mammograms improve a downstream classifier — evaluated under a **leakage-free,
reproducible protocol**. The authoritative design lives in
[`METHODOLOGY.md`](METHODOLOGY.md).

> **Framing.** This is an *exploratory feasibility study*, not evidence of
> clinical robustness. No diagnostic-support claims are made. The contribution is
> methodological rigour: a leakage-free measurement protocol with statistical
> significance testing and full reproducibility.

---

## Why this rework exists (response to reviewers)

| Reviewer concern | What we changed | Where |
|---|---|---|
| **Data leakage** (GAN trained on all images; one synthetic set reused across all CV folds; LPIPS vs full real set) | Strict **patient-level hold-out**: GAN + LPIPS see **only DEV**; classifier evaluated on a **locked TEST** set. Optional **fold-wise** protocol retrains the GAN per fold. | [`METHODOLOGY.md` §1–2](METHODOLOGY.md), `src/breastsynth/pipeline.py`, `evaluation/protocols.py` |
| Dataset described poorly; patient separation; 553 vs 490; the 63 images; duplicates | Objective manifest with `patient_id`; **pHash near-duplicate removal**; documented balancing; all counts logged | `data/manifest.py`, `data/dedup.py`, `data/splits.py` |
| Overly strong clinical claims | Reframed as exploratory feasibility study; claims softened everywhere | both papers |
| LPIPS threshold 0.2 unjustified; memorisation risk | **Threshold sweep** + **nearest-neighbour memorisation audit** | `selection/lpips_filter.py`, `metrics/lpips_metrics.py` |
| LPIPS alone is limited | Added **FID** and **KID** | `metrics/fidelity.py` |
| 128×128 loses detail | Stated as an explicit limitation | both papers |
| Missing implementation details / reproducibility | Seeds, pinned deps, per-run JSON reports, config files, released checkpoint | `seed.py`, `config.py`, `runlog.py`, `configs/` |
| EfficientNet cited wrongly | Cite **Tan & Le (ICML 2019)**; add **CLAIM / TRIPOD+AI / PROBAST+AI** | `paper/*/references*.bib` |
| No statistical significance | **Wilcoxon** (Holm-corrected), **bootstrap CIs**, **Cliff's delta**, **DeLong** for AUC | `evaluation/stats.py` |
| Single dataset / no external validation | **External validation on MIAS** (auto-download, no retraining) + remaining multi-scanner work documented | `data/external.py`, `data/download.py`, `scripts/06_external_validation.py` |

---

## Repository layout

```
src/breastsynth/          installable package
  config.py  seed.py  runlog.py  pipeline.py
  data/        manifest · dedup · patient-level splits · dataset
  models/      stylegan2ada · wgan (baseline) · classifier (EfficientNet-B0)
  generative/  GAN training · sampling
  selection/   LPIPS filter · threshold sweep · memorisation
  metrics/     classification (+CIs) · FID/KID · LPIPS
  evaluation/  leakage-free protocols · statistical tests
  viz/         paper figures
configs/                  default.yaml (hold-out) · foldwise.yaml
scripts/                  00_build_manifest … 05_make_tables · run_all.sh · run_foldwise.py
tests/                    leakage guarantees + statistics sanity
paper/lncs/               Springer LNCS paper (IBERAMIA resubmission)
paper/tcc/                ABNTeX2 monograph (Portuguese), revised
data/                     dataset128/ (553 sample images) · dataset.csv · ckpt400.pth (LFS)
legacy/                   original prototype scripts (kept for reference)
```

---

## Install

```bash
git lfs install && git lfs pull        # fetch the trained checkpoint
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # or: pip install -r requirements.txt
```
Requires a CUDA GPU for training (developed on an NVIDIA RTX 4060, 8 GB).

## Reproduce the pipeline

```bash
# 0a) (Optional) Reproduce preprocessing from RAW full mammograms:
#     resize to 128 and orient every breast to the LEFT (removes laterality cue).
#     Skip this if you already have data/dataset128/.
python scripts/prepare_images.py --src <raw_full_mammograms> --dst data/dataset128 \
    --size 128 --laterality-csv <cbis>/csv/dicom_info.csv

# 0) Build the manifest (patient-level, de-duplicated).
#    Preferred — recover patient_id from the CBIS-DDSM case CSVs:
python scripts/00_build_manifest.py --image-root data/dataset128 \
    --cbis-csv mass_case_description_train_set.csv \
    --cbis-csv calc_case_description_train_set.csv   # ...+ test CSVs
#    Or keep labels from the legacy CSV and recover patient_id from dicom_info.csv
#    (awsaf49 Kaggle layout) — the easiest patient-level route:
python scripts/00_build_manifest.py --image-root data/dataset128 \
    --legacy-csv data/dataset.csv --dicom-info <cbis>/csv/dicom_info.csv
#    Fallback (no patient ids; warns — patient separation NOT guaranteed):
python scripts/00_build_manifest.py --image-root data/dataset128 --legacy-csv data/dataset.csv

# 1)–5) Full leakage-free run:
bash scripts/run_all.sh configs/default.yaml
```

This trains the GAN on DEV only, runs the LPIPS sweep + filter, computes FID/KID,
trains/evaluates the classifier on the locked TEST set across seeds with
significance tests, and renders LaTeX tables into `paper/generated/`.

External validation on a second dataset (auto-downloads via kagglehub):
```bash
python scripts/06_external_validation.py --config configs/default.yaml \
    --dataset mias --download --synthetic results/synthetic --out results/external_mias
# then fold the external table into the papers:
python scripts/05_make_tables.py --summary results/classification/summary.json \
    --external results/external_mias/external_summary.json --external-name MIAS
```
Known datasets: `cbis-ddsm`, `mias`, `inbreast` (see `data/download.py`); or pass a
raw `owner/dataset` slug. Requires Kaggle credentials (`~/.kaggle/kaggle.json`).

Gold-standard (expensive) fold-wise protocol:
```bash
python scripts/run_foldwise.py --config configs/foldwise.yaml --out results/foldwise
```

## Tests

```bash
pytest -q          # proves no patient overlap / no TEST leakage; validates stats
```

---

## Important notes on the numbers

The previously reported figures (accuracy 0.651 / F1 0.658 / AUC 0.703 at 2:1)
were produced under the leaky setup and are **not** reused. All tables in the
papers are regenerated from `results/` after a leakage-free run; until then they
show `\TODO{}` placeholders. Under the corrected protocol, expect gains to be
smaller and better calibrated.

## Data & privacy

`data/dataset128/` holds the 553 resized (128×128) full-mammogram images used
here; the full CBIS-DDSM is available from TCIA. The trained generator checkpoint
is tracked via Git LFS. No patient-identifiable data is included.

## License

Academic and research use only. See [LICENSE](LICENSE).

## Author
Rafael Palheta Tokairin — Undergraduate Thesis (TCC), State University of Londrina (UEL).
