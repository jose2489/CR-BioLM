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
           if r["analysis"] == analysis and r["source"] == source and r["area"] > 0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper2/figs/fig_rq3.png")
    a = ap.parse_args()
    rows = [json.loads(l) for l in io.open(DATA / "null_models_scores.jsonl", encoding="utf-8")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, spec, title in ((axes[0], MAPS, "Range maps"),
                            (axes[1], CLIM, "Climate envelopes (WorldClim 2.1)")):
        data = [lifts(rows, an, src) for an, src, _ in spec]
        labels = [f"{lab} ({len(d)})" for (_, _, lab), d in zip(spec, data)]
        ax.boxplot(data[::-1], vert=False, showfliers=False, widths=0.55,
                   medianprops={"color": "#1f5c4a", "linewidth": 2.2})
        ax.set_yticks(range(1, len(data) + 1), labels[::-1])
        ax.axvline(1, color="0.4", linestyle=":", linewidth=1)
        ax.grid(axis="x", color="0.9")
        ax.set_title(title)
        ax.set_xlabel("lift: share of records captured\nover share of the country covered")
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
