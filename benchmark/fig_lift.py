"""Figure 4 of the paper: per-species lift of maps and climate envelopes.

Drawn from null_models_scores.jsonl (null_models.py), which holds, per species and
source, the held-out capture and the share of the country covered. Sources that give no
map or envelope (area 0) are left out, as in the tables; the count is in parentheses.

Run:  python -m benchmark.fig_lift [--out paper2/figs/fig_rq3.png]
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent / "data"
MAPS = [("heldout_random", "map", "System, random split"),
        ("heldout_spatial", "map", "System, spatial split"),
        ("map_all", "manual", "Manual alone"),
        ("map_all", "paid", "Mistral, asked freely"),
        ("map_all", "open", "Gemma, asked freely")]
CLIM = [("climate_worldclim", "ours", "System profile, random split"),
        ("heldout_spatial", "clim_records", "System profile, spatial split"),
        ("heldout_spatial", "clim_map", "Profile from the map, spatial"),
        ("climate_worldclim", "paid_manual", "Mistral, given the Manual"),
        ("climate_worldclim", "open_manual", "Gemma, given the Manual"),
        ("climate_worldclim", "paid", "Mistral, asked freely"),
        ("climate_worldclim", "open", "Gemma, asked freely")]


def lifts(rows, analysis, source, null="area"):
    out = [r["capture"] / r[null] for r in rows
           if r["analysis"] == analysis and r["source"] == source and r["area"] > 0
           and r[null] > 0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper2/figs/fig_rq3.pdf")
    ap.add_argument("--null", default="p_sites", choices=("area", "p_sites"),
                    help="share of the country (area) or of collection sites (p_sites)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in io.open(DATA / "null_models_scores.jsonl", encoding="utf-8")]
    # drawn at the journal's full width (180 mm), panels stacked, 8 pt text
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 8})
    fig, axes = plt.subplots(2, 1, figsize=(180 / 25.4, 6.4),
                             sharex=True, gridspec_kw={"height_ratios": [5, 7]})
    for ax, spec, title in ((axes[0], MAPS, "Range maps"),
                            (axes[1], CLIM, "Climate envelopes (WorldClim 2.1)")):
        data = [lifts(rows, an, src, a.null) for an, src, _ in spec]
        labels = [f"{lab} ({len(d)})" for (_, _, lab), d in zip(spec, data)]
        ax.boxplot(data[::-1], vert=False, showfliers=False, widths=0.55,
                   medianprops={"color": "#1f5c4a", "linewidth": 1.8},
                   boxprops={"linewidth": 0.8}, whiskerprops={"linewidth": 0.8},
                   capprops={"linewidth": 0.8})
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_yticks(range(1, len(data) + 1), labels[::-1])
        ax.axvline(1, color="0.4", linestyle=":", linewidth=1)
        ax.grid(axis="x", color="0.9")
        ax.set_title(title, loc="left")
        if ax is axes[1]:
            ax.set_xlabel("lift: share of records captured over share of " +
                          ("collection sites covered" if a.null == "p_sites"
                           else "the country covered"))
    fig.tight_layout()
    fig.savefig(a.out, dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
