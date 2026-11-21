"""Figure 6 of the paper: anticipating where species are recorded next.

Drawn from the saved per-species scores, with the rows and blocks of Table 8:
per_species.jsonl (forecast.py) for the models with collecting effort and the rules,
per_species_review2.jsonl (forecast_review2.py) for the system's model, records alone,
kernel density and MaxEnt. Nothing is refitted.

Drawn at the journal's full width (180 mm) with 8 pt text, so it prints at 100%.

Run:  python -m benchmark.fig_forecast [--out paper2/figs/fig_forecast_metrics.pdf]
"""
from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory as blended

DATA = Path(__file__).resolve().parent / "data" / "forecast"
TOP_K = 10
INK, MUTED, ACCENT = "#222222", "#8a8a8a", "#1F4E46"

# (file, score name, label); None starts a block with the given title
ROWS = [
    (None, None, "One model for all species, without collecting effort"),
    ("review2", "system's model", "System's model (Manual, map, records)"),
    ("review2", "records only", "Records alone"),
    (None, None, "The same, with collecting effort"),
    ("main", "gradient boosting, all evidence", "Manual, map, records and effort"),
    ("main", "gradient boosting, without the Manual", "Records and effort (no Manual)"),
    ("main", "gradient boosting, without records", "Manual and effort (no records)"),
    (None, None, "Rules and per-species models"),
    ("main", "nearest record", "Nearest known record"),
    ("review2", "kernel density", "Kernel density of known records"),
    ("review2", "maxent", "MaxEnt per species (2,835 species)"),
    ("main", "system map (rule)", "System's map, as a rule"),
    ("main", "prevalence", "Prevalence"),
]


def load() -> dict:
    out = defaultdict(list)
    for tag, name in (("main", "per_species.jsonl"), ("review2", "per_species_review2.jsonl")):
        for line in io.open(DATA / name, encoding="utf-8"):
            r = json.loads(line)
            out[(tag, r["score"])].append(r)
    return out


def pooled_lift(rows: list[dict]) -> float:
    return sum(r["hits"] for r in rows) / sum(TOP_K * r["new"] / r["n"] for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper2/figs/fig_forecast_metrics.pdf")
    a = ap.parse_args()
    S = load()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 8,
                         "axes.edgecolor": MUTED, "xtick.color": INK, "ytick.color": INK,
                         "font.family": "DejaVu Sans"})
    # y positions: a block title takes a slot of its own
    ypos, labels, data, lifts, heads = [], [], [], [], []
    y = 0.0
    for tag, score, label in ROWS:
        if tag is None:
            y += 0.35 if y else 0
            heads.append((y, label))
            y += 0.85
            continue
        rows = S[(tag, score)]
        ypos.append(y)
        labels.append(label)
        data.append([r["auc"] for r in rows])
        lifts.append(pooled_lift(rows))
        y += 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(180 / 25.4, 4.6), sharey=True,
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    colors = [ACCENT if i == 0 else "#7f9c95" for i in range(len(labels))]
    bp = a1.boxplot(data, positions=ypos, vert=False, showfliers=False, widths=0.6,
                    patch_artist=True, medianprops={"color": INK, "lw": 1.6},
                    whiskerprops={"color": MUTED, "lw": 0.8}, capprops={"color": MUTED, "lw": 0.8},
                    boxprops={"lw": 0})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    a1.axvline(0.5, color=MUTED, lw=0.8, ls=":")
    a1.set_xlim(0.2, 1.0)
    a1.set_title("AUC per species (0.5 is chance)", loc="left")
    a2.barh(ypos, lifts, color=colors, height=0.6)
    a2.axvline(1, color=MUTED, lw=0.8, ls=":")
    a2.set_title(f"Top-{TOP_K} lift (1 is chance)", loc="left")
    for yy, v in zip(ypos, lifts):
        a2.text(v + 0.15, yy, f"{v:.1f}", va="center", fontsize=7.5, color=INK)
    a2.set_xlim(0, max(lifts) * 1.15)
    a2.set_xticks(range(0, int(max(lifts) * 1.15) + 1, 2))
    for ax in (a1, a2):
        ax.grid(axis="x", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    a1.set_yticks(ypos, labels)
    for yy, title in heads:
        fig.text(0.005, yy, title, transform=blended(fig.transFigure, a1.transData), ha="left", va="center",
                fontsize=8, fontstyle="italic", color=MUTED)
    a1.set_ylim(y - 0.4, -0.6)
    fig.subplots_adjust(left=0.345, right=0.97, top=0.93, bottom=0.07, wspace=0.14)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
