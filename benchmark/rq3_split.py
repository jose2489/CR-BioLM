"""RQ3 held out at random and in space: the system's map and climate profile.

The map comparison (mapcompare.py) and the climate comparison (climate_compare.py)
score the system's map and profile on one random half of a species' records, built from
the other half. Records of one collecting trip then fall on both sides, which flatters
anything built from records. Here both are also scored with a spatial split: blocks of
about 20 km (BLOCK degrees) go to one half or the other at random, so no test record has
a training record in its block (Roberts et al. 2017).

Variants, per split, both halves in turn:
  map             the system's map: the Manual's region-slope pairs plus those holding
                  >= 5 training records, inside the Manual's elevation band
  clim_records    5th-95th percentile envelope of annual mean temperature and rainfall
                  over the training records inside the Manual's band (the profile)
  clim_map        the same envelope over the ground the training map covers
  clim_5records   the profile from 5 training records only

Writes data/rq3_split_scores.jsonl and data/rq3_figure.png (per-species lift of every
source of the map and climate comparisons, with the two splits for the system).

Run:  python -m benchmark.rq3_split            (score, report, figure)
      python -m benchmark.rq3_split --figure   (figure from the saved scores)
"""
from __future__ import annotations

import argparse
import io
import json
import random
import statistics as st

import numpy as np

from . import climate_profile as C
from . import map_refine as R
from . import mapcompare as M
from .climate_compare import env_mask

DATA = M.DATA
OUT = DATA / "rq3_split_scores.jsonl"
BLOCK = 0.18                    # degrees, about 20 km


def folds(sp: str, lonlat: list[tuple], split: str) -> list[tuple[list[int], list[int]]]:
    n = len(lonlat)
    if split == "random":                 # mapcompare's halves, as published
        rng = random.Random(f"{M.SEED}-{sp}")
        order = list(range(n))
        rng.shuffle(order)
        half = n // 2
        return [(order[:half], order[half:]), (order[half:], order[:half])]
    block = [(int(lo // BLOCK), int(la // BLOCK)) for lo, la in lonlat]
    rng = random.Random(f"{M.SEED}-{sp}-blocks")
    side = {b: rng.random() < .5 for b in sorted(set(block))}
    a = [i for i, b in enumerate(block) if side[b]]
    b = [i for i, b in enumerate(block) if not side[b]]
    return [(a, b), (b, a)]


def _env(values: dict, pix: np.ndarray):
    env = {}
    for v in ("bio1", "bio12"):
        x = values[v][pix]
        x = x[~np.isnan(x)]
        if not len(x):
            return None
        env[v] = (float(np.quantile(x, .05)), float(np.quantile(x, .95)))
    return env


def score() -> list[dict]:
    from mpcr_rag.schema import Ficha
    from mpcr_rag.store import pg_store
    species = [r["species"] for r in json.loads(
        (DATA / "maps_sample.json").read_text(encoding="utf-8"))["sample"]]
    clim = C.climate()
    land_index = R.grid()["index"]
    n_land = int((land_index >= 0).sum())
    rows = []
    conn = pg_store.connect()
    with conn.cursor() as cur:
        for k, sp in enumerate(species, 1):
            cur.execute("SELECT ficha FROM mpcr.fichas WHERE species = %s", (sp,))
            manual = M.from_ficha(Ficha(**cur.fetchone()[0]))
            cur.execute("""SELECT o.lon, o.lat, o.region, o.vertiente, o.elev_dem
                           FROM mpcr.occurrences o
                           JOIN mpcr.species_evidence e ON e.taxon_key = o.species_key
                           WHERE e.species = %s""", (sp,))
            got = cur.fetchall()
            lonlat = [(r[0], r[1]) for r in got]
            recs = [(r[2], None if r[3] is None else ("Caribe" if "arib" in r[3] else "Pacífico"), r[4])
                    for r in got]
            pix = R.record_pixels(lonlat)
            band = np.zeros(len(pix), bool)
            band[pix >= 0] = C.in_band(cur, sp, pix[pix >= 0])
            for split in ("random", "spatial"):
                fs = folds(sp, lonlat, split)
                if any(not tr or not te for tr, te in fs):
                    continue
                acc = {v: [0, [], 0] for v in ("map", "clim_records", "clim_map", "clim_5records")}
                for tr, te in fs:
                    ing = M.with_records(manual, [recs[i] for i in tr])
                    mk = M.mask(ing)
                    acc["map"][0] += M.captured(ing, [recs[i] for i in te])
                    acc["map"][1].append(float(mk.mean()) if mk is not None else 0.0)
                    acc["map"][2] += len(te)
                    trp = [i for i in tr if pix[i] >= 0]
                    tep = np.array([pix[i] for i in te if pix[i] >= 0], dtype=int)
                    inb = [i for i in trp if band[i]] or trp
                    five = random.Random(f"{M.SEED}-{sp}-5").sample(inb, min(5, len(inb)))
                    envs = {"clim_records": _env(clim, pix[inb]) if inb else None,
                            "clim_5records": _env(clim, pix[five]) if five else None,
                            "clim_map": (_env(clim, np.flatnonzero(mk))
                                         if mk is not None and mk.any() else None)}
                    for v, e in envs.items():
                        em = env_mask(e) if e else np.zeros(n_land, bool)
                        acc[v][0] += int(em[tep].sum())
                        acc[v][1].append(float(em.mean()))
                        acc[v][2] += len(tep)
                for v, (cap, area, n) in acc.items():
                    rows.append({"species": sp, "split": split, "variant": v,
                                 "capture": cap / max(1, n), "area": float(np.mean(area)), "n": n})
            if k % 50 == 0:
                print(f"  {k}/{len(species)}", flush=True)
    conn.close()
    with io.open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def report(rows: list[dict]) -> None:
    from scipy.stats import wilcoxon
    print(f"\n{'variant':15s} {'split':8s} {'n':>4s} {'capture':>8s} {'area':>6s} {'lift':>5s}")
    by = {}
    for v in ("map", "clim_records", "clim_map", "clim_5records"):
        for s in ("random", "spatial"):
            sub = [r for r in rows if r["variant"] == v and r["split"] == s]
            lift = [r["capture"] / r["area"] for r in sub if r["area"] > 0]
            by[(v, s)] = {r["species"]: r for r in sub}
            print(f"{v:15s} {s:8s} {len(sub):4d} {st.median(r['capture'] for r in sub):8.0%} "
                  f"{st.median(r['area'] for r in sub):6.0%} {st.median(lift):5.2f}")
    for v in ("map", "clim_records", "clim_map"):
        a, b = by[(v, "random")], by[(v, "spatial")]
        common = sorted(set(a) & set(b))
        d = [a[s]["capture"] - b[s]["capture"] for s in common]
        print(f"  {v}: random minus spatial capture, median {st.median(d):+.1%}, "
              f"Wilcoxon p = {wilcoxon(d).pvalue:.2g} ({len(common)} species)")
    a, b = by[("clim_records", "spatial")], by[("clim_map", "spatial")]
    common = sorted(set(a) & set(b))
    d = [b[s]["capture"] - a[s]["capture"] for s in common]
    print(f"  spatial: map profile minus records profile, median {st.median(d):+.1%}, "
          f"Wilcoxon p = {wilcoxon(d).pvalue:.2g}")


def figure(rows: list[dict]) -> None:
    """Per-species lift, one box per source: maps (left), climate (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def lift(cap, area):
        return cap / area if area > 0 else None

    sysrows = {(v, s): [lift(r["capture"], r["area"]) for r in rows
                        if r["variant"] == v and r["split"] == s]
               for v in ("map", "clim_records", "clim_map") for s in ("random", "spatial")}
    maps = [json.loads(l) for l in io.open(DATA / "mapcompare_scores_maps.jsonl", encoding="utf-8")]
    mlift = {s: [lift(r["captured"] / r["n_records"], r["area"]) for r in maps
                 if r["source"] == s and r["has_map"] and r["n_records"]]
             for s in ("manual", "paid", "open")}
    clim = [json.loads(l) for l in io.open(DATA / "climate_compare_scores.jsonl", encoding="utf-8")]
    clift = {s: [lift(r["capture"], r["area"]) for r in clim if r["source"] == s and r["has_env"]]
             for s in ("paid_manual", "open_manual", "paid", "open")}
    panels = [
        ("Range maps", [
            ("System, random split", sysrows[("map", "random")]),
            ("System, spatial split", sysrows[("map", "spatial")]),
            ("Manual alone", mlift["manual"]),
            ("Mistral, asked freely", mlift["paid"]),
            ("Gemma, asked freely", mlift["open"])]),
        ("Climate envelopes (WorldClim)", [
            ("System profile, random split", sysrows[("clim_records", "random")]),
            ("System profile, spatial split", sysrows[("clim_records", "spatial")]),
            ("Profile from the map, spatial", sysrows[("clim_map", "spatial")]),
            ("Mistral, given the Manual", clift["paid_manual"]),
            ("Gemma, given the Manual", clift["open_manual"]),
            ("Mistral, asked freely", clift["paid"]),
            ("Gemma, asked freely", clift["open"])]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (title, items) in zip(axes, panels):
        data = [[x for x in v if x is not None] for _, v in items]
        ax.boxplot(data, vert=False, showfliers=False, widths=.6,
                   medianprops={"color": "#1F4E46", "lw": 2})
        ax.axvline(1, color="grey", lw=.8, ls=":")
        ax.set_yticks(range(1, len(items) + 1), [f"{n} ({len(d)})" for (n, _), d in zip(items, data)])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("lift (share of records captured / share of the country covered)", fontsize=9)
        ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    path = DATA / "rq3_figure.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true", help="figure from the saved scores only")
    args = ap.parse_args()
    if args.figure:
        rows = [json.loads(l) for l in io.open(OUT, encoding="utf-8")]
    else:
        rows = score()
        report(rows)
    figure(rows)


if __name__ == "__main__":
    main()
