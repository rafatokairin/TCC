# LNCS conference paper (IBERAMIA resubmission)

`main.tex` is a Springer **LNCS** paper. The class files are not redistributable,
so fetch them once:

```bash
# from paper/lncs/
wget https://ftp.springer.de/pub/tex/latex/llncs/latex2e/llncs2e.zip
unzip llncs2e.zip 'llncs.cls' 'splncs04.bst'
```

(or copy `llncs.cls` and `splncs04.bst` from an Overleaf "Springer LNCS" template).

## Build

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## Filling in the numbers

The results tables and figures carry `\TODO{...}` placeholders and
`\IfFileExists{../generated/*.tex}` guards. After running the leakage-free
pipeline on your GPU:

```bash
# from repo root
bash scripts/run_all.sh            # produces results/ and paper/generated/*.tex
```

`scripts/05_make_tables.py` writes `paper/generated/{results_table.tex,
stats_table.tex, fidelity_table.tex}`, which the paper `\input`s automatically.
Copy the figures produced in `results/synthetic/` (`lpips_distribution.png`,
`threshold_sweep.png`) into `figures/` and uncomment the `\includegraphics`
lines in Fig. 1. Replace the remaining inline `\TODO{}` numbers (e.g. retained
fraction, memorisation rate) from `results/`.

## Quick local preview without LNCS

Temporarily change the first line to `\documentclass[11pt]{article}` and comment
out `\authorrunning`/`\titlerunning`/`\institute`/`\orcidID`; the body compiles
with only minor spacing differences.
