#!/usr/bin/env python
"""Render LaTeX table snippets from results so the papers \\input real numbers.

    python scripts/05_make_tables.py --summary results/classification/summary.json \
        --fidelity results/synthetic/fidelity.json --out paper/generated

Writes paper/generated/{results_table.tex, fidelity_table.tex, stats_table.tex}.
Both paper/lncs/main.tex and paper/tcc use \\input on these files, replacing the
placeholder \\TODO macros once you have re-run the pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_ci(m):
    return f"{m['mean']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]"


def _ratio_label(ratio: str) -> str:
    return "Real only" if float(ratio) == 0 else f"{int(float(ratio))}:1"


def results_table(summary: dict) -> str:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Synthetic:Real & Accuracy [95\% CI] & F1 [95\% CI] & AUC [95\% CI] \\",
        r"\midrule",
    ]
    for ratio in sorted(summary["per_ratio"], key=lambda r: float(r)):
        m = summary["per_ratio"][ratio]
        lines.append(
            f"{_ratio_label(ratio)} & {_fmt_ci(m['accuracy'])} & {_fmt_ci(m['f1'])} & {_fmt_ci(m['auc'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def stats_table(summary: dict) -> str:
    w = summary.get("stats", {}).get("wilcoxon_auc", {})
    dl = summary.get("stats", {}).get("delong_auc_vs_baseline", {})
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Ratio & Wilcoxon $p_{\text{adj}}$ (AUC) & Cliff's $\delta$ & DeLong $p$ & Signif. \\",
        r"\midrule",
    ]
    for ratio in sorted(w, key=lambda r: float(r)):
        row = w[ratio]
        d = dl.get(ratio, {})
        sig = "yes" if row.get("reject") else "no"
        lines.append(
            f"{_ratio_label(ratio)} & {row.get('p_adjusted', float('nan')):.3f} & "
            f"{row.get('cliffs_delta', float('nan')):.2f} & "
            f"{d.get('p_value', float('nan')):.3f} & {sig} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def fidelity_table(fidelity: dict) -> str:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Class & FID $\downarrow$ & KID $\downarrow$ & n (real/fake) \\",
        r"\midrule",
    ]
    for name, m in fidelity.items():
        lines.append(
            f"{name.title()} & {m['fid']:.1f} & {m['kid_mean']:.3f} $\\pm$ {m['kid_std']:.3f} "
            f"& {m['n_real']}/{m['n_fake']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def external_table(summary: dict, dataset: str = "external") -> str:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        rf"Synthetic:Real & Accuracy [95\% CI] & F1 [95\% CI] & AUC [95\% CI]"
        rf" ({dataset}) \\",
        r"\midrule",
    ]
    for ratio in sorted(summary["per_ratio"], key=lambda r: float(r)):
        m = summary["per_ratio"][ratio]
        lines.append(
            f"{_ratio_label(ratio)} & {_fmt_ci(m['accuracy'])} & {_fmt_ci(m['f1'])} & {_fmt_ci(m['auc'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="results/classification/summary.json")
    ap.add_argument("--fidelity", default="results/synthetic/fidelity.json")
    ap.add_argument("--external", default=None, help="external_summary.json (validation)")
    ap.add_argument("--external-name", default="MIAS")
    ap.add_argument("--out", default="paper/generated")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = json.loads(Path(args.summary).read_text())
    (out / "results_table.tex").write_text(results_table(summary))
    (out / "stats_table.tex").write_text(stats_table(summary))
    if Path(args.fidelity).exists():
        fidelity = json.loads(Path(args.fidelity).read_text())
        (out / "fidelity_table.tex").write_text(fidelity_table(fidelity))
    if args.external and Path(args.external).exists():
        ext = json.loads(Path(args.external).read_text())
        (out / "external_table.tex").write_text(external_table(ext, args.external_name))
    print(f"Wrote LaTeX tables to {out}/")


if __name__ == "__main__":
    main()
