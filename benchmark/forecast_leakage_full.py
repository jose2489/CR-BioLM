"""The leakage test on every channel through which the Manual enters the forecast.

forecast_leakage.py and forecast_review2.py prune only the Manual's region-slope pairs
to those holding a record by the cut. The Manual also enters through its elevation band
and its forest types, which the pruning left untouched. Here, at each cut (1999 for the
training table, 2009 for the test table), every species' Manual is reduced to what the
records known by that cut could have told its authors:

  region-slope pairs   only those holding a record by the cut
  elevation band       clipped to the elevation span (DEM) of the records known by the
                       cut; empty when they do not overlap it
  forest types         removed (they cannot be tied to dated records)

The Manual's features, the system's map and the map's climate profile are recomputed
from the reduced Manual, the system's model is retrained on the reduced training table
and scored on the reduced test table, and compared per species with the published
model and with the same model on the records alone (data/forecast/per_species_review2.jsonl).
A Manual reduced this way cannot carry any information collected after the cut, so
whatever it still adds over the records alone is not leakage.

Writes data/forecast/leakage_full.json and per_species_leakage_full.jsonl.

Run:  python -m benchmark.forecast_leakage_full
"""
from __future__ import annotations

import collections
import io
import json
import time

import numpy as np

from . import forecast as F
from . import map_refine as R


def reduce_all(tab: dict, G, D: dict, cut: int) -> np.ndarray:
    pre = D["year"] <= cut
    attrs_pre, elev_pre = collections.defaultdict(list), collections.defaultdict(list)
    for s, a, p, k in zip(D["sp"].tolist(), D["attrs"], D["pix"].tolist(), pre.tolist()):
        if k:
            attrs_pre[s].append(a)
            elev_pre[s].append(float(G.elev[p]))
    X = tab["X"].copy()
    fi = {n: F.FEATURES.index(n) for n in ("manual_pairs", "manual_band", "manual_map",
                                            "manual_moist", "system_map", "clim_map_dist")}
    X[:, fi["manual_moist"]] = np.nan
    order = np.argsort(tab["s"], kind="stable")
    g = tab["s"][order]
    cuts = np.flatnonzero(np.diff(g)) + 1
    empty = np.zeros(len(G.pc), bool)
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(g)]):
        s = int(g[a])
        ing = D["ings"][s]
        seen = {(r, sl) for r, sl in attrs_pre.get(s, []) if r and sl}
        keep = [p for p in ing["pairs"] if tuple(p) in seen]
        ev = elev_pre.get(s, [])
        if ing["elev"] and ev:
            lo, hi = max(ing["elev"][0], min(ev)), min(ing["elev"][1], max(ev))
            band = G.band_mask([lo, hi]) if lo <= hi else empty
        elif ing["elev"]:
            band = empty
        else:
            band = G.band_mask(None)
        rows = order[a:b]
        cc = tab["c"][rows]
        pm = G.pairs_mask(keep)
        X[rows, fi["manual_pairs"]] = G.frac(pm)[cc]
        X[rows, fi["manual_band"]] = G.frac(band)[cc] if ing["elev"] else np.nan
        X[rows, fi["manual_map"]] = G.frac(pm & band)[cc]
        wide = R.widen({**ing, "pairs": keep}, attrs_pre.get(s, []))
        mk = G.pairs_mask(wide["pairs"]) & band
        X[rows, fi["system_map"]] = G.frac(mk)[cc]
        envs = {v: (F._env(G.clim[v + "_px"][mk]) if mk.any() else None) for v in ("bio1", "bio12")}
        X[rows, fi["clim_map_dist"]] = F._clim_dist(G, envs)[cc]
    return X


def main() -> None:
    from scipy.stats import mannwhitneyu, wilcoxon
    from mpcr_rag.store import pg_store
    from .forecast_leakage import PUBLISHED, volumes
    from .forecast_tertiles import known_records
    t0 = time.time()
    species, sets, _rec = F.build()
    tr, te = sets["train"], sets["test"]
    conn = pg_store.connect()
    with conn.cursor() as cur:
        D = F.load(cur)
    conn.close()
    assert D["species"] == species
    G = F.Grid()
    Xtr = reduce_all(tr, G, D, F.CUTS["train"][0])
    Xte = reduce_all(te, G, D, F.CUTS["test"][0])
    # forest types are removed by leaving their feature out of the model altogether
    cols = [c for c in F._cols(F.SUBSETS["without effort"])
            if F.FEATURES[c] != "manual_moist"]
    gb, _l, _i = F.train({**tr, "X": Xtr}, cols, 3_000_000)
    name = "Manual reduced on every channel"
    ps = F.per_species(te, {name: gb.predict_proba(Xte[:, cols])[:, 1]})
    print(f"  model ({time.time()-t0:.0f}s)", flush=True)
    with io.open(F.OUT / "per_species_leakage_full.jsonl", "w", encoding="utf-8") as fh:
        for r in ps:
            fh.write(json.dumps({**r, "species": species[r["species"]]}, ensure_ascii=False) + "\n")

    by = collections.defaultdict(dict)
    for r in ps:
        by[r["score"]][species[r["species"]]] = r
    for l in io.open(F.OUT / "per_species_review2.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["score"] in ("system's model", "records only"):
            by[r["score"]][r["species"]] = r
    vol = dict(zip(species, volumes(species)))
    nk = known_records()
    rng = np.random.default_rng(20261116)
    out = {"median_auc": {k: float(np.median([r["auc"] for r in v.values()])) for k, v in by.items()},
           "pooled_top10": {}, "paired": {}}
    for k, v in by.items():
        exp = sum(F.TOP_K * r["new"] / r["n"] for r in v.values())
        out["pooled_top10"][k] = sum(r["hits"] for r in v.values()) / exp
    for x, y in ((name, "records only"), ("system's model", name)):
        sp = sorted(set(by[x]) & set(by[y]))
        d = np.array([by[x][s]["auc"] - by[y][s]["auc"] for s in sp])
        lg = np.log10(1 + np.array([nk.get(s, 0) for s in sp]))
        slope, icpt = np.polyfit(lg, d, 1)
        boot = [np.polyfit(lg[i], d[i], 1)[0]
                for i in (rng.integers(len(sp), size=len(sp)) for _ in range(2000))]
        late = np.array([PUBLISHED.get(vol.get(s, ""), 0) > F.CUTS["test"][0] for s in sp])
        known = np.array([vol.get(s, "") in PUBLISHED for s in sp])
        out["paired"][f"{x} vs {y}"] = {
            "species": len(sp), "median": float(np.median(d)), "mean": float(d.mean()),
            "wilcoxon_p": float(wilcoxon(d).pvalue),
            "slope_per_log10": float(slope),
            "slope_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "fitted_at_3": float(icpt + slope * np.log10(4)),
            "fitted_at_100": float(icpt + slope * np.log10(101)),
            "early_mean": float(d[known & ~late].mean()), "late_mean": float(d[known & late].mean()),
            "late_vs_early_p": float(mannwhitneyu(d[known & late], d[known & ~late]).pvalue)}
    print(json.dumps(out, indent=1))
    (F.OUT / "leakage_full.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
