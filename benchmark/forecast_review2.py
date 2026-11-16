"""Forecast checks asked for by the second review: per-species baselines, and leakage in
the training window.

1. Per-species baselines fitted to what was known by the test cut (2009), scored on the
   same candidate cells and with the same per-species AUC and top-10 lift as the
   forecast (forecast.per_species):

     kernel density   sum over the species' known record pixels of a Gaussian kernel
                      of KDE_KM, at each candidate cell's centre
     maxent           MaxEnt (elapid, multiplier 1 fixed in advance) on the nine
                      bioclimatic variables and elevation of sdm_layer, background drawn
                      from pixels collected by the cut (target group); a cell's score is
                      the mean suitability of its pixels. Species with at least
                      sdm_layer.MIN_TRAIN known pixels.

2. Leakage in the training window. Every volume appeared after the training cut (1999),
   so the Manual's features could encode 2000-2009 finds that the model then learns
   from. The model is retrained with the Manual reduced, in training, to the
   region-slope pairs holding a record by 1999, and tested with it reduced to the pairs
   holding a record by 2009 (forecast_leakage.py does the test side only). If it still
   beats the same model fitted to the records alone, the Manual's contribution is not leakage
   through the region-slope pairs. The elevation band and forest types are not reduced:
   neither can be tied to dated records.

3. The Manual's contribution against the number of known records, as a continuous
   relation: the per-species AUC gain regressed on log10(1 + records known by 2009),
   with a species bootstrap interval on the slope, for both learners.

Writes data/forecast/review2.json and data/forecast/per_species_review2.jsonl.

Run:  python -m benchmark.forecast_review2 [--workers 8]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import io
import json
import time
import zlib

import numpy as np

from . import forecast as F
from . import map_refine as R
from . import sdm_layer as S

KDE_KM = 10.0
SEED = 20261115


# ------------------------------------------------------------------- 1. baselines

def kde_scores(G, te: dict, rec: dict, cut: int) -> np.ndarray:
    known = rec["year"] <= cut
    cells_by_sp = collections.defaultdict(list)
    for s, c in zip(rec["sp"][known].tolist(), G.pc[rec["pix"][known]].tolist()):
        cells_by_sp[s].append(c)
    out = np.zeros(len(te["s"]), np.float32)
    order = np.argsort(te["s"], kind="stable")
    g = te["s"][order]
    cuts = np.flatnonzero(np.diff(g)) + 1
    h = KDE_KM / F.PX_KM
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(g)]):
        s = int(g[a])
        kc = np.array(cells_by_sp.get(s, []))
        if not len(kc):
            continue
        rows = order[a:b]
        cc = te["c"][rows]
        dr = G.cr[cc][:, None] - G.cr[kc][None, :]
        dc = G.cc[cc][:, None] - G.cc[kc][None, :]
        out[rows] = np.exp(-(dr ** 2 + dc ** 2) / (2 * h * h)).sum(1)
    return out


_W: dict = {}


def _init(bg_pool: np.ndarray, pc: np.ndarray, n_cells: int):
    _W.update(bg=bg_pool, pc=pc, n=n_cells, X=S.features())


def _maxent_cells(args):
    s, pres = args
    import elapid
    X, bg_pool = _W["X"], _W["bg"]
    pres = np.unique(pres)
    if len(pres) < S.MIN_TRAIN:
        return s, None
    rng = np.random.default_rng(zlib.crc32(f"{SEED}-{s}".encode()))
    bg = rng.choice(bg_pool, size=min(S.N_BACKGROUND, len(bg_pool)), replace=False)
    bg = bg[~np.isin(bg, pres)]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    try:
        m = elapid.MaxentModel(feature_types=["linear", "quadratic", "hinge", "product"],
                               beta_multiplier=1.0, n_cpus=1, random_state=0)
        m.fit(np.vstack([Z[pres], Z[bg]]), np.r_[np.ones(len(pres)), np.zeros(len(bg))])
        suit = m.predict(Z)
    except Exception:
        return s, None
    pc, n = _W["pc"], _W["n"]
    cell = np.bincount(pc, suit, n) / np.maximum(np.bincount(pc, minlength=n), 1)
    return s, cell.astype(np.float32)


def maxent_scores(G, te: dict, rec: dict, cut: int, workers: int) -> np.ndarray:
    known = rec["year"] <= cut
    bg_pool = np.unique(rec["pix"][known])
    pix_by_sp = collections.defaultdict(list)
    for s, p in zip(rec["sp"][known].tolist(), rec["pix"][known].tolist()):
        pix_by_sp[s].append(p)
    want = sorted(set(te["s"].tolist()))
    jobs = [(s, np.array(pix_by_sp.get(s, []), dtype=int)) for s in want
            if len(set(pix_by_sp.get(s, []))) >= S.MIN_TRAIN]
    out = np.full(len(te["s"]), np.nan, np.float32)
    order = np.argsort(te["s"], kind="stable")
    g = te["s"][order]
    cuts = np.flatnonzero(np.diff(g)) + 1
    span = {int(g[a]): order[a:b] for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(g)])}
    t0 = time.time()
    with cf.ProcessPoolExecutor(workers, initializer=_init,
                                initargs=(bg_pool, G.pc, G.n)) as pool:
        for i, (s, cell) in enumerate(pool.map(_maxent_cells, jobs, chunksize=4), 1):
            if cell is not None:
                rows = span[s]
                out[rows] = cell[te["c"][rows]]
            if i % 250 == 0:
                print(f"    maxent {i}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------- 2. training-window leakage

def reduce_manual(tab: dict, G, D: dict, cut: int) -> np.ndarray:
    """The pair table's features with the Manual reduced to the region-slope pairs
    holding a record by `cut` (as forecast_leakage.py does for the test side)."""
    pre = D["year"] <= cut
    attrs_pre = collections.defaultdict(list)
    for s, a, k in zip(D["sp"].tolist(), D["attrs"], pre.tolist()):
        if k:
            attrs_pre[s].append(a)
    X = tab["X"].copy()
    fi = {n: F.FEATURES.index(n) for n in ("manual_pairs", "manual_map", "system_map",
                                            "clim_map_dist")}
    order = np.argsort(tab["s"], kind="stable")
    g = tab["s"][order]
    cuts = np.flatnonzero(np.diff(g)) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(g)]):
        s = int(g[a])
        ing = D["ings"][s]
        seen = {(r, sl) for r, sl in attrs_pre.get(s, []) if r and sl}
        keep = [p for p in ing["pairs"] if tuple(p) in seen]
        if len(keep) == len(ing["pairs"]):
            continue
        rows = order[a:b]
        cc = tab["c"][rows]
        pm = G.pairs_mask(keep)
        bm = G.band_mask(ing["elev"])
        X[rows, fi["manual_pairs"]] = G.frac(pm)[cc]
        X[rows, fi["manual_map"]] = G.frac(pm & bm)[cc]
        wide = R.widen({**ing, "pairs": keep}, attrs_pre.get(s, []))
        mk = G.pairs_mask(wide["pairs"]) & bm
        X[rows, fi["system_map"]] = G.frac(mk)[cc]
        envs = {v: (F._env(G.clim[v + "_px"][mk]) if mk.any() else None)
                for v in ("bio1", "bio12")}
        X[rows, fi["clim_map_dist"]] = F._clim_dist(G, envs)[cc]
    return X


# --------------------------------------------------------------------------- main

def main() -> None:
    from scipy.stats import wilcoxon
    from mpcr_rag.store import pg_store
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    species, sets, rec = F.build()
    tr, te = sets["train"], sets["test"]
    conn = pg_store.connect()
    with conn.cursor() as cur:
        D = F.load(cur)
    conn.close()
    assert D["species"] == species, "cached tables and catalogue disagree"
    G = F.Grid()
    cut_tr, cut_te = F.CUTS["train"][0], F.CUTS["test"][0]
    print(f"tables ready ({time.time()-t0:.0f}s)", flush=True)

    scores = {}
    cols = F._cols(F.SUBSETS["without effort"])
    gb, _lr, _imp = F.train(tr, cols, 3_000_000)
    scores["system's model"] = gb.predict_proba(te["X"][:, cols])[:, 1]
    # the system's model leaves effort out, so its Manual-free counterpart is records only
    cols_nm = F._cols(("records",))
    gb_nm, _l, _i = F.train(tr, cols_nm, 3_000_000)
    scores["records only"] = gb_nm.predict_proba(te["X"][:, cols_nm])[:, 1]
    print(f"  reference models ({time.time()-t0:.0f}s)", flush=True)

    Xtr = reduce_manual(tr, G, D, cut_tr)
    Xte = reduce_manual(te, G, D, cut_te)
    gb_r, _l, _i = F.train({**tr, "X": Xtr}, cols, 3_000_000)
    scores["Manual reduced in training and test"] = gb_r.predict_proba(Xte[:, cols])[:, 1]
    print(f"  reduced-Manual model ({time.time()-t0:.0f}s)", flush=True)

    scores["kernel density"] = kde_scores(G, te, rec, cut_te)
    print(f"  kernel density ({time.time()-t0:.0f}s)", flush=True)
    scores["maxent"] = maxent_scores(G, te, rec, cut_te, a.workers)
    print(f"  maxent ({time.time()-t0:.0f}s)", flush=True)

    ps = F.per_species(te, {k: v for k, v in scores.items() if k != "maxent"})
    # MaxEnt only where it was fitted: per_species on the pairs of fitted species
    fitted = ~np.isnan(scores["maxent"])
    sub = {k: te[k][fitted] for k in ("s", "c", "y")}
    ps_m = F.per_species(sub, {"maxent": scores["maxent"][fitted],
                               "system's model, MaxEnt species": scores["system's model"][fitted]})
    rows = ps + ps_m
    with io.open(F.OUT / "per_species_review2.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({**r, "species": species[r["species"]]}, ensure_ascii=False) + "\n")

    by = collections.defaultdict(dict)
    for r in rows:
        by[r["score"]][r["species"]] = r
    summ = F.summarize_species(rows)
    out = {"per_species": summ, "kde_km": KDE_KM, "paired": {}}
    for x, y in (("system's model", "kernel density"),
                 ("system's model, MaxEnt species", "maxent"),
                 ("Manual reduced in training and test", "records only"),
                 ("system's model", "Manual reduced in training and test")):
        sp = sorted(set(by[x]) & set(by[y]))
        d = np.array([by[x][s]["auc"] - by[y][s]["auc"] for s in sp])
        out["paired"][f"{x} vs {y}"] = {"species": len(sp), "median": float(np.median(d)),
                                         "mean": float(d.mean()),
                                         "wilcoxon_p": float(wilcoxon(d).pvalue) if d.any() else 1.0}
    for k, v in summ.items():
        print(f"  {k:42s} AUC {v['median_auc']:.3f}  top-10 lift {v['pooled_lift']:5.1f}  "
              f"({v['species']} species)")
    for k, v in out["paired"].items():
        print(f"  {k}: {v['species']} species, median {v['median']:+.4f}, mean {v['mean']:+.4f}, "
              f"p = {v['wilcoxon_p']:.2g}")

    # 3. the Manual's gain against known records, both learners, from the main run
    from .forecast_tertiles import known_records
    main_rows = [json.loads(l) for l in io.open(F.OUT / "per_species.jsonl", encoding="utf-8")]
    mb = collections.defaultdict(dict)
    for r in main_rows:
        mb[r["score"]][r["species"]] = r
    nk = known_records()
    rng = np.random.default_rng(SEED)
    out["gain_vs_records"] = {}
    for learner in ("gradient boosting", "logistic regression"):
        A, B = mb[f"{learner}, all evidence"], mb[f"{learner}, without the Manual"]
        sp = sorted(set(A) & set(B))
        x = np.log10(1 + np.array([nk.get(s, 0) for s in sp]))
        d = np.array([A[s]["auc"] - B[s]["auc"] for s in sp])
        slope, icpt = np.polyfit(x, d, 1)
        boot = []
        for _ in range(2000):
            i = rng.integers(len(sp), size=len(sp))
            boot.append(np.polyfit(x[i], d[i], 1)[0])
        lo, hi = np.quantile(boot, [.025, .975])
        out["gain_vs_records"][learner] = {
            "species": len(sp), "median_gain": float(np.median(d)), "mean_gain": float(d.mean()),
            "slope_per_log10": float(slope), "slope_ci": [float(lo), float(hi)],
            "fitted_at_3": float(icpt + slope * np.log10(4)),
            "fitted_at_100": float(icpt + slope * np.log10(101))}
        print(f"  {learner}: gain {np.median(d):+.4f} median; slope per log10 records "
              f"{slope:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    (F.OUT / "review2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {F.OUT / 'review2.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
