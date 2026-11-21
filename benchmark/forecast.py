"""Beyond the record: predicting where species were found next.

Two questions the record cannot answer by lookup, asked of what was known at a date
and checked against what was collected afterwards:

  1. Where will this species turn up next?   (per species, rank the cells)
  2. What may I find here that nobody has recorded here yet?   (per cell, rank species)

The unit is a (species, 5 km cell) pair. A model is trained on what was known up to
1999, to predict the records of 2000-2009, and tested on what was known up to 2009,
to predict the records of 2010 onwards. Every feature is computed from data before
the cut. Only cells someone collected in during the window count, and in each of them
the candidate species that were not collected are the negatives (target-group
background, Phillips et al. 2009), which takes much of the collecting bias out of the
comparison. Pairs already recorded in the cell before the cut are left out: the task
is new occurrences, not repeat visits.

The features are the system's own evidence: the Manual (regions and slopes, elevation
band, forest moisture), the system's map, distance to and density of earlier records,
the climate profile (from records and from the map), companions (species that share
cells with this one) and collecting effort. One gradient-boosting model serves every
species; it is compared with the system's map as a rule, with nearness to a known
record, with prevalence and with a logistic regression on the same features.

Run:  python -m benchmark.forecast            (builds the tables once, trains, reports)
      python -m benchmark.forecast --rebuild  (rebuilds the cached tables)
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from . import climate_profile as C
from . import map_refine as R

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "forecast"
CELL = 5                      # land pixels per side of a cell (about 5 km)
PX_KM = 0.92                  # side of a DEM pixel, km
NEAR_KM = 50                  # a species is a candidate in cells this close to a record
N_COMPANIONS = 20
MIN_WIDTH = {"bio1": 1.0, "bio12": 100.0}   # degC, mm: the narrowest envelope width
CUTS = {"train": (1999, 2009), "test": (2009, 2026)}   # known up to a, window (a, b]
FEATURES = ("manual_pairs", "manual_band", "manual_map", "manual_moist", "system_map",
            "near_km", "recs_10km", "log_prevalence", "clim_rec_dist", "clim_map_dist",
            "companions", "log_effort", "log_richness")


# ---------------------------------------------------------------------------- grid

class Grid:
    def __init__(self):
        g = R.grid()
        rows, cols = np.nonzero(g["index"] >= 0)          # the land index runs in row order
        key = (rows // CELL) * 100_000 + cols // CELL
        _, self.pc = np.unique(key, return_inverse=True)  # land pixel -> cell
        self.n = int(self.pc.max()) + 1
        self.npix = np.bincount(self.pc, minlength=self.n).astype(np.float32)
        self.cr = np.bincount(self.pc, rows, self.n) / self.npix
        self.cc = np.bincount(self.pc, cols, self.n) / self.npix
        self.rows, self.cols = rows, cols
        self.elev = np.asarray(g["elev"], dtype=np.float32)
        self.moist = g["moist"]
        names, slopes = g["names"].astype(str), g["slopes"].astype(str)
        pair = np.char.add(np.char.add(names, "|"), slopes)
        self.pair_codes, self.pair_id = np.unique(pair, return_inverse=True)
        self.pair_index = {p: i for i, p in enumerate(self.pair_codes)}
        clim = C.climate()
        self.clim = {}
        for v in ("bio1", "bio12"):
            x = clim[v]
            ok = ~np.isnan(x)
            s = np.bincount(self.pc[ok], x[ok], self.n)
            k = np.bincount(self.pc[ok], minlength=self.n)
            self.clim[v] = np.where(k > 0, s / np.maximum(k, 1), np.nan)
            self.clim[v + "_px"] = x
        # neighbours: the 8 cells around, as a sparse 0/1 matrix with the diagonal
        from scipy.sparse import coo_matrix
        rc = {(int(r // CELL), int(c // CELL)): i for i, (r, c) in
              enumerate(zip(self.cr, self.cc))}
        a, b = [], []
        for (r, c), i in rc.items():
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    j = rc.get((r + dr, c + dc))
                    if j is not None:
                        a.append(i), b.append(j)
        self.adj = coo_matrix((np.ones(len(a), np.float32), (a, b)), (self.n, self.n)).tocsr()

    def frac(self, mask: np.ndarray) -> np.ndarray:
        return (np.bincount(self.pc, mask.astype(np.float32), self.n) / self.npix).astype(np.float32)

    def pairs_mask(self, pairs) -> np.ndarray:
        ids = [self.pair_index[f"{r}|{s}"] for r, s in pairs if f"{r}|{s}" in self.pair_index]
        return np.isin(self.pair_id, ids)

    def band_mask(self, elev) -> np.ndarray:
        if not elev:
            return np.ones(len(self.pc), dtype=bool)
        return (self.elev >= elev[0]) & (self.elev <= elev[1])


# ------------------------------------------------------------------------ the data

def load(cur):
    """Species with a Manual entry and records; their ingredients; all records."""
    from mpcr_rag.schema import Ficha
    cur.execute("""SELECT e.species, e.taxon_key, f.ficha FROM mpcr.species_evidence e
                   JOIN mpcr.fichas f ON f.species = e.species
                   WHERE e.taxon_key IS NOT NULL AND e.n_records > 0 ORDER BY e.species""")
    species, keys, ings = [], [], []
    for sp, key, ficha in cur.fetchall():
        try:
            ing = R.ingredients(Ficha(**ficha))
        except Exception:                     # an entry the parser cannot read
            continue
        species.append(sp), keys.append(key), ings.append(ing)
    index = {k: i for i, k in enumerate(keys)}
    cur.execute("""SELECT species_key, lon, lat, year, region, vertiente
                   FROM mpcr.occurrences WHERE year IS NOT NULL""")
    rows = [r for r in cur.fetchall() if r[0] in index]
    sp = np.array([index[r[0]] for r in rows], dtype=np.int32)
    pix = R.record_pixels([(r[1], r[2]) for r in rows])
    year = np.array([r[3] for r in rows], dtype=np.int32)
    attrs = [(r[4], None if r[5] is None else ("Caribe" if "arib" in r[5] else "Pacífico"))
             for r in rows]
    ok = pix >= 0
    return {"species": species, "ings": ings, "sp": sp[ok], "pix": pix[ok], "year": year[ok],
            "attrs": [a for a, k in zip(attrs, ok) if k]}


def _env(x: np.ndarray):
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return None
    return float(np.quantile(x, .05)), float(np.quantile(x, .95))


def _clim_dist(G: Grid, envs: dict) -> np.ndarray:
    """Per cell, how far its mean climate lies outside the envelope, in envelope
    widths (0 inside); the larger of temperature and rain. NaN without an envelope."""
    if any(e is None for e in envs.values()):
        return np.full(G.n, np.nan, np.float32)
    d = np.zeros(G.n, np.float32)
    for v, (lo, hi) in envs.items():
        x = G.clim[v]
        w = max(hi - lo, MIN_WIDTH[v])     # a one-site envelope would divide by ~0
        d = np.maximum(d, np.maximum(0, np.maximum(lo - x, x - hi)) / w)
    return np.minimum(d, 20)


def static_features(G: Grid, D: dict) -> dict:
    """The Manual's part, which does not depend on the cut."""
    S = len(D["species"])
    out = {k: np.zeros((S, G.n), np.float16) for k in
           ("manual_pairs", "manual_band", "manual_map", "manual_moist")}
    for i, ing in enumerate(D["ings"]):
        pm, bm = G.pairs_mask(ing["pairs"]), G.band_mask(ing["elev"])
        out["manual_pairs"][i] = G.frac(pm)
        out["manual_band"][i] = G.frac(bm) if ing["elev"] else np.nan
        out["manual_map"][i] = G.frac(pm & bm)
        if ing["moist"]:
            want = sum(1 << j for j, c in enumerate(R.MOIST) if c in ing["moist"])
            out["manual_moist"][i] = G.frac((G.moist & want) != 0)
        else:
            out["manual_moist"][i] = np.nan
    return out


def cut_features(G: Grid, D: dict, cut: int) -> dict:
    """Everything that depends on the records known up to ``cut``."""
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree
    S = len(D["species"])
    pre = D["year"] <= cut
    sp, pix = D["sp"][pre], D["pix"][pre]
    attrs = [a for a, k in zip(D["attrs"], pre) if k]
    by_sp = collections.defaultdict(list)
    for j, s in enumerate(sp):
        by_sp[int(s)].append(j)
    cell = G.pc[pix]
    f = {k: np.full((S, G.n), np.nan, np.float16) for k in
         ("system_map", "near_km", "recs_10km", "clim_rec_dist", "clim_map_dist")}
    prevalence = np.zeros(S, np.float32)
    cent = np.c_[G.cr, G.cc]
    for s in range(S):
        js = by_sp.get(s, [])
        ing = D["ings"][s]
        mk = G.pairs_mask(R.widen(ing, [attrs[j] for j in js])["pairs"]) & G.band_mask(ing["elev"])
        f["system_map"][s] = G.frac(mk)
        envs = {v: (_env(G.clim[v + "_px"][mk]) if mk.any() else None) for v in ("bio1", "bio12")}
        f["clim_map_dist"][s] = _clim_dist(G, envs)
        prevalence[s] = len(js)
        if not js:
            continue
        p = pix[js]
        t = cKDTree(np.c_[G.rows[p], G.cols[p]])
        d, _ = t.query(cent)
        f["near_km"][s] = np.minimum(d * PX_KM, 500)
        f["recs_10km"][s] = np.minimum(t.query_ball_point(cent, 10 / PX_KM, return_length=True), 60000)
        if ing["elev"]:
            lo, hi = ing["elev"]
            band = (G.elev[p] >= lo - 200) & (G.elev[p] <= hi + 200)
            p = p[band] if band.any() else p
        f["clim_rec_dist"][s] = _clim_dist(G, {v: _env(G.clim[v + "_px"][p]) for v in ("bio1", "bio12")})
    f["log_prevalence"] = np.log1p(prevalence)
    # presence (species x cell) before the cut, companions and effort
    P = csr_matrix((np.ones(len(sp), np.float32), (sp, cell)), (S, G.n))
    P.data[:] = 1
    n = np.asarray(P.sum(1)).ravel()
    co = (P @ P.T).tocsr()
    T_rows, T_cols = [], []
    for s in range(S):
        row = co.getrow(s)
        if row.nnz <= 1:
            continue
        j, c = row.indices, row.data
        jac = c / (n[s] + n[j] - c)
        jac[j == s] = -1
        jac[c < 3] = -1
        top = j[np.argsort(-jac)[:N_COMPANIONS]]
        top = top[jac[np.argsort(-jac)[:N_COMPANIONS]] > 0]
        T_rows += [s] * len(top)
        T_cols += list(top)
    Tm = csr_matrix((np.ones(len(T_rows), np.float32), (T_rows, T_cols)), (S, S))
    near = (P @ G.adj).tocsr()
    near.data[:] = 1
    f["companions"] = (Tm @ near).toarray().astype(np.float16)
    effort = np.bincount(cell, minlength=G.n)
    richness = np.asarray(P.sum(0)).ravel()
    f["log_effort"] = np.log1p(effort).astype(np.float32)
    f["log_richness"] = np.log1p(richness).astype(np.float32)
    f["_known"] = P
    return f


def rows_for(G: Grid, D: dict, st: dict, cf: dict, cut: int, end: int):
    """(species, cell) pairs in cells collected in the window, with features and label."""
    win = (D["year"] > cut) & (D["year"] <= end)
    wsp, wcell = D["sp"][win], G.pc[D["pix"][win]]
    sampled = np.unique(wcell)
    found = set(zip(wsp.tolist(), wcell.tolist()))
    known = cf["_known"].tocsc()
    S = len(D["species"])
    blocks = []
    for c in sampled:
        cand = ((st["manual_pairs"][:, c] > 0) | (cf["system_map"][:, c] > 0)
                | (np.nan_to_num(cf["near_km"][:, c].astype(np.float32), nan=1e9) <= NEAR_KM))
        kn = known[:, c].toarray().ravel() > 0
        s = np.nonzero(cand & ~kn)[0]
        if len(s):
            blocks.append(np.c_[s, np.full(len(s), c)])
    idx = np.vstack(blocks)
    s, c = idx[:, 0], idx[:, 1]
    X = np.empty((len(idx), len(FEATURES)), np.float32)
    for k, name in enumerate(FEATURES):
        if name in st:
            X[:, k] = st[name][s, c]
        elif name in ("log_prevalence",):
            X[:, k] = cf[name][s]
        elif name in ("log_effort", "log_richness"):
            X[:, k] = cf[name][c]
        else:
            X[:, k] = cf[name][s, c]
    y = np.fromiter(((int(a), int(b)) in found for a, b in zip(s, c)), bool, len(s))
    return {"s": s.astype(np.int32), "c": c.astype(np.int32), "X": X, "y": y}


# ------------------------------------------------------------------------ evaluation

def _per_group_auc(group, score, y):
    from sklearn.metrics import roc_auc_score
    out = []
    order = np.argsort(group, kind="stable")
    g, sc, yy = group[order], score[order], y[order]
    cuts = np.flatnonzero(np.diff(g)) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(g)]):
        if 0 < yy[a:b].sum() < b - a:
            out.append(roc_auc_score(yy[a:b], sc[a:b]))
    return out


def _recall_at(group, score, y, k=20):
    order = np.lexsort((-score, group))
    g, yy = group[order], y[order]
    cuts = np.flatnonzero(np.diff(g)) + 1
    rec = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(g)]):
        pos = yy[a:b].sum()
        if pos:
            rec.append(yy[a:min(b, a + k)].sum() / min(pos, k))
    return rec


def evaluate(test: dict, scores: dict, sample_idx: set[int]) -> list[dict]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = test["y"]
    res = []
    for name, sc in scores.items():
        sc = np.nan_to_num(sc, nan=-1e9)
        sp_auc = _per_group_auc(test["s"], sc, y)
        in300 = np.isin(test["s"], list(sample_idx))
        sp300 = _per_group_auc(test["s"][in300], sc[in300], y[in300])
        cell_rec = _recall_at(test["c"], sc, y, 20)
        res.append({"score": name, "auc": float(roc_auc_score(y, sc)),
                    "ap": float(average_precision_score(y, sc)),
                    "species_auc_median": float(np.median(sp_auc)), "n_species": len(sp_auc),
                    "species300_auc_median": float(np.median(sp300)) if sp300 else None,
                    "n_species300": len(sp300),
                    "cell_recall20_mean": float(np.mean(cell_rec)), "n_cells": len(cell_rec)})
    return res


CACHE = Path(tempfile.gettempdir()) / "crbiolm_forecast.npz"   # ~600 MB, kept off Drive
GROUPS = {
    "Manual": ("manual_pairs", "manual_band", "manual_map", "manual_moist"),
    "system map": ("system_map", "clim_map_dist"),
    "records": ("near_km", "recs_10km", "log_prevalence", "clim_rec_dist", "companions"),
    "effort": ("log_effort", "log_richness"),
}
SUBSETS = {                     # name -> groups the model may use
    "all evidence": ("Manual", "system map", "records", "effort"),
    "without the Manual": ("records", "effort"),
    "without records": ("Manual", "effort"),
    "without effort": ("Manual", "system map", "records"),
}
MISSING = ("near_km", "clim_rec_dist", "clim_map_dist")


def build(rebuild: bool = False):
    """The train and test pair tables and the dated records, from the cache if any."""
    from mpcr_rag.store import pg_store
    if CACHE.exists() and not rebuild:
        z = np.load(CACHE)
        sets = {n: {k: z[f"{n}_{k}"] for k in ("s", "c", "X", "y")} for n in CUTS}
        rec = {k: z[f"rec_{k}"] for k in ("sp", "pix", "year")}
        species = json.loads((OUT / "species.json").read_text(encoding="utf-8"))
        return species, sets, rec
    t0 = time.time()
    conn = pg_store.connect()
    with conn.cursor() as cur:
        D = load(cur)
    conn.close()
    G = Grid()
    st = static_features(G, D)
    sets = {}
    for name, (cut, end) in CUTS.items():
        sets[name] = rows_for(G, D, st, cut_features(G, D, cut), cut, end)
        r = sets[name]
        print(f"{name}: known to {cut}, window {cut+1}-{end}: {len(r['y']):,} pairs in "
              f"{len(np.unique(r['c']))} cells, {int(r['y'].sum()):,} new occurrences "
              f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(CACHE, **{f"{n}_{k}": v for n, d in sets.items() for k, v in d.items()},
             rec_sp=D["sp"], rec_pix=D["pix"], rec_year=D["year"])
    OUT.mkdir(exist_ok=True)
    with io.open(OUT / "species.json", "w", encoding="utf-8") as fh:
        json.dump(D["species"], fh, ensure_ascii=False)
    return D["species"], sets, {"sp": D["sp"], "pix": D["pix"], "year": D["year"]}


def _cols(groups) -> list[int]:
    return [FEATURES.index(f) for g in groups for f in GROUPS[g]]


class Imputed:
    """Median fill plus a missing flag for the features that can be missing, for the
    logistic regression (the boosted trees take NaN as it is)."""
    def __init__(self, X, cols):
        self.cols = cols
        self.fill = np.nanmedian(X[:, cols], axis=0)
        self.flag = [i for i, c in enumerate(cols) if FEATURES[c] in MISSING]

    def __call__(self, X):
        X = X[:, self.cols].copy()
        miss = np.isnan(X)
        X[miss] = np.take(self.fill, np.nonzero(miss)[1])
        return np.c_[X, miss[:, self.flag]]


def train(tr: dict, cols: list[int], max_train: int, seed: int = 20261102):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    keep = np.arange(len(tr["y"]))
    if len(keep) > max_train:               # every positive, a sample of negatives
        pos, neg = keep[tr["y"]], keep[~tr["y"]]
        keep = np.r_[pos, rng.choice(neg, max_train - len(pos), replace=False)]
    X, y = tr["X"][keep], tr["y"][keep]
    gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                        min_samples_leaf=200, l2_regularization=1.0,
                                        early_stopping=True, validation_fraction=0.1,
                                        random_state=0).fit(X[:, cols], y)
    imp = Imputed(X, cols)
    sub = rng.choice(len(y), min(len(y), 1_000_000), replace=False)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(imp(X[sub]), y[sub])
    return gb, lr, imp


def _metrics(te: dict, sc: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = te["y"]
    sc = np.nan_to_num(sc, nan=-1e9)
    return {"auc": float(roc_auc_score(y, sc)), "ap": float(average_precision_score(y, sc)),
            "species_auc": float(np.median(_per_group_auc(te["s"], sc, y))),
            "cell_r20": float(np.mean(_recall_at(te["c"], sc, y, 20))),
            "cell_r100": float(np.mean(_recall_at(te["c"], sc, y, 100)))}


TOP_K = 10


def _hits_at(score: np.ndarray, y: np.ndarray, k: int) -> float:
    """Expected new occurrences among the k best-scored cells, ties at the cut shared
    evenly (a rule that scores many cells alike is not credited with a lucky order)."""
    k = min(k, len(score))
    t = np.sort(score)[::-1][k - 1]
    above = score > t
    tied = score == t
    return float(y[above].sum() + (k - above.sum()) * y[tied].mean())


def per_species(te: dict, scores: dict) -> list[dict]:
    """Per species: AUC (does the score rank the cells where it turned up above the
    others?) and the precision of its TOP_K best cells as a multiple of what TOP_K cells
    drawn at random would give (prevalence differs between species, so raw precision
    or PR-AUC would not compare). Species with at least TOP_K candidates, one new cell
    and one cell without."""
    from sklearn.metrics import roc_auc_score
    order = np.argsort(te["s"], kind="stable")
    g, y = te["s"][order], te["y"][order]
    cuts = np.flatnonzero(np.diff(g)) + 1
    spans = [(a, b) for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(g)])
             if 0 < y[a:b].sum() < b - a and b - a >= TOP_K]
    rows = []
    for name, sc in scores.items():
        sc = np.nan_to_num(sc, nan=-1e9)[order]
        for a, b in spans:
            yy, ss = y[a:b], sc[a:b]
            hits = _hits_at(ss, yy, TOP_K)
            rows.append({"score": name, "species": int(g[a]), "n": int(b - a),
                         "new": int(yy.sum()), "auc": float(roc_auc_score(yy, ss)),
                         "hits": hits, "lift": hits / (TOP_K * yy.mean())})
    return rows


def summarize_species(rows: list[dict]) -> dict:
    """Per score: median AUC; for the top cells, pooled lift (all hits over all hits
    expected at random), the share of species with a hit, and the median lift."""
    out = {}
    for name in dict.fromkeys(r["score"] for r in rows):
        sub = [r for r in rows if r["score"] == name]
        exp = sum(TOP_K * r["new"] / r["n"] for r in sub)
        out[name] = {"species": len(sub), "median_auc": float(np.median([r["auc"] for r in sub])),
                     "pooled_lift": sum(r["hits"] for r in sub) / exp,
                     "share_with_hit": float(np.mean([r["hits"] > 0 for r in sub])),
                     "median_lift": float(np.median([r["lift"] for r in sub]))}
    return out


def plot_per_species(rows: list[dict], path: Path) -> None:
    """AUC per species (boxes) and the pooled lift of the top cells (bars), one row
    per score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(r["score"] for r in rows))
    summ = summarize_species(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 0.45 * len(names) + 1.5), sharey=True)
    a1.boxplot([[r["auc"] for r in rows if r["score"] == n] for n in names], vert=False,
               showfliers=False, widths=.6, medianprops={"color": "#1F4E46", "lw": 2})
    a1.axvline(.5, color="grey", lw=.8, ls=":")
    a1.set_title("AUC per species")
    pos = list(range(1, len(names) + 1))
    a2.barh(pos, [summ[n]["pooled_lift"] for n in names], color="#1F4E46", height=.6)
    a2.axvline(1, color="grey", lw=.8, ls=":")
    a2.set_title(f"Precision of the {TOP_K} best cells, × random")
    for ax in (a1, a2):
        ax.grid(axis="x", alpha=.3)
    a1.set_yticks(pos, names)
    a1.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _rules(te: dict) -> dict:
    F = {k: te["X"][:, i] for i, k in enumerate(FEATURES)}
    return {"system map (rule)": F["system_map"] - 1e-3 * np.nan_to_num(F["clim_map_dist"], nan=10),
            "nearest record": -np.nan_to_num(F["near_km"], nan=1e4),
            "prevalence": F["log_prevalence"]}


# ------------------------------------------------------------------------ parks

def park_of_cell() -> tuple[np.ndarray, list[str]]:
    """Per cell, the protected area covering at least half of it (-1 if none). Areas
    of the same name are merged; areas under 20 km2 hold too few cells to score."""
    import geopandas as gpd
    import rasterio
    from rasterio import features as rf
    from utils.distribution_map.geodata import PATHS
    G = Grid()
    with rasterio.open(PATHS["dem"]) as dem:
        shape, transform, crs = dem.shape, dem.transform, dem.crs
    asp = gpd.read_file(PATHS["protected_areas"]).to_crs(crs)
    asp["geometry"] = asp.geometry.make_valid()       # one polygon self-intersects
    asp = asp.dissolve(by="nombre_asp").reset_index()
    asp = asp[asp.geometry.area > 0]
    names = asp["nombre_asp"].tolist()
    ras = rf.rasterize(((g, i + 1) for i, g in enumerate(asp.geometry)), out_shape=shape,
                       transform=transform, fill=0, dtype="int32")
    land = R.grid()["index"] >= 0
    px = ras[land] - 1
    out = -np.ones(G.n, np.int32)
    for c in range(G.n):
        v = px[G.pc == c]
        v = v[v >= 0]
        if len(v) and np.bincount(v).max() >= 0.5 * G.npix[c]:
            out[c] = int(np.bincount(v).argmax())
    return out, names


def park_eval(te: dict, scores: dict, rec: dict) -> dict:
    """What may I find in this park? Per park, species ranked by their best cell in it;
    the new species are those recorded in the park from 2010 and in none of its cells
    before. Parks with at least five new species."""
    G = Grid()
    pcell, names = park_of_cell()
    park = pcell[te["c"]]
    inpark = park >= 0
    known = set(zip(rec["sp"][rec["year"] <= 2009].tolist(),
                    pcell[G.pc[rec["pix"][rec["year"] <= 2009]]].tolist()))
    out = {}
    for name, sc in scores.items():
        sc = np.nan_to_num(sc, nan=-1e9)
        best = collections.defaultdict(lambda: -np.inf)
        lab = collections.defaultdict(bool)
        for s, p, v, y in zip(te["s"][inpark], park[inpark], sc[inpark], te["y"][inpark]):
            if (int(s), int(p)) in known:
                continue
            k = (int(p), int(s))
            best[k] = max(best[k], v)
            lab[k] = lab[k] or bool(y)
        by_park = collections.defaultdict(list)
        for (p, s), v in best.items():
            by_park[p].append((v, lab[(p, s)]))
        rec_k = {20: [], 50: [], 100: []}
        n_new, kept = [], []
        for p, lst in by_park.items():
            pos = sum(l for _, l in lst)
            if pos < 5:
                continue
            lst.sort(key=lambda t: -t[0])
            for k in rec_k:
                rec_k[k].append(sum(l for _, l in lst[:k]) / min(pos, k))
            n_new.append(pos)
            kept.append(names[p])
        out[name] = {f"recall@{k}": float(np.median(v)) for k, v in rec_k.items()}
        out[name]["parks"] = len(kept)
        out[name]["median new species"] = float(np.median(n_new))
        out[name]["median candidates"] = float(np.median([len(by_park[names.index(n)]) for n in kept]))
    return out


# ------------------------------------------------------------------------ examples

REASON = {
    "system_map": lambda v: f"{v:.0%} de la celda está en el mapa del sistema",
    "manual_band": lambda v: f"{v:.0%} de la celda está en la banda altitudinal del Manual",
    "manual_pairs": lambda v: f"{v:.0%} de la celda está en regiones y vertientes que cita el Manual",
    "manual_moist": lambda v: f"{v:.0%} de la celda tiene el tipo de bosque del Manual",
    "manual_map": lambda v: f"{v:.0%} de la celda está en el mapa del Manual",
    "near_km": lambda v: f"el registro conocido más cercano está a {v:.0f} km",
    "recs_10km": lambda v: f"{v:.0f} registros conocidos a menos de 10 km",
    "companions": lambda v: f"{v:.0f} de sus 20 especies compañeras están registradas en la zona",
    "clim_rec_dist": lambda v: ("el clima de la celda está dentro de su perfil" if v == 0
                                else f"el clima de la celda se aleja {v:.1f} anchos de su perfil"),
    "clim_map_dist": lambda v: ("el clima está dentro del perfil de su mapa" if v == 0
                                else f"el clima se aleja {v:.1f} anchos del perfil de su mapa"),
    "log_effort": lambda v: f"la celda tiene {np.expm1(v):.0f} registros previos de cualquier especie",
    "log_richness": lambda v: f"en la celda se han registrado {np.expm1(v):.0f} especies",
    "log_prevalence": lambda v: f"la especie tiene {np.expm1(v):.0f} registros previos",
}


def examples(species: list[str], te: dict, gb, cols: list[int], rec: dict, k: int = 4) -> list[dict]:
    """Species of the 300-species sample with the most new cells after 2009: a map of
    the model's score per cell with the records before and after, and the reasons
    (largest SHAP contributions) for its three best-scored new cells."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    G = Grid()
    sample = {r["species"] for r in json.loads(
        (DATA / "maps_sample.json").read_text(encoding="utf-8"))["sample"]}
    # typical cases, not the extremes: sample species with 10-40 new cells whose AUC is
    # closest to the median species AUC (the species with most new cells are weeds and
    # cultivated plants, recorded everywhere, where the model does worst)
    pos_cells = collections.Counter(te["s"][te["y"]].tolist())
    cand = [s for s, n in pos_cells.items() if species[s] in sample and 10 <= n <= 40]
    allp = gb.predict_proba(te["X"][:, cols])[:, 1]
    auc = {}
    for s in cand:
        m = te["s"] == s
        a = _per_group_auc(np.zeros(m.sum(), int), allp[m], te["y"][m])
        if a:
            auc[s] = a[0]
    med = float(np.median(list(auc.values())))
    pick = sorted(auc, key=lambda s: abs(auc[s] - med))[:k]
    expl = shap.TreeExplainer(gb)
    folder = OUT / "examples"
    folder.mkdir(parents=True, exist_ok=True)
    rows_n, cols_n = int(G.cr.max() // CELL) + 1, int(G.cc.max() // CELL) + 1
    out = []
    for s in pick:
        m = te["s"] == s
        X, c, y = te["X"][m], te["c"][m], te["y"][m]
        p = gb.predict_proba(X[:, cols])[:, 1]
        img = np.full((rows_n, cols_n), np.nan)
        land = np.full((rows_n, cols_n), np.nan)
        land[(G.cr // CELL).astype(int), (G.cc // CELL).astype(int)] = 0
        img[(G.cr[c] // CELL).astype(int), (G.cc[c] // CELL).astype(int)] = p
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.imshow(land, cmap="Greys", vmin=-1, vmax=3)
        im = ax.imshow(img, cmap="YlGn", vmin=0, vmax=np.nanquantile(p, .99))
        for sel, col, lab in (((rec["sp"] == s) & (rec["year"] <= 2009), "#222222", "hasta 2009"),
                              ((rec["sp"] == s) & (rec["year"] > 2009), "#d62728", "desde 2010")):
            q = rec["pix"][sel]
            ax.scatter(G.cols[q] / CELL, G.rows[q] / CELL, s=7, c=col, label=lab, lw=0)
        ax.set_title(species[s], style="italic")
        ax.legend(loc="lower left", fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=.6, label="probabilidad estimada (celdas recolectadas desde 2010)")
        path = folder / (species[s].replace(" ", "_") + ".png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        top = np.argsort(-p)[:3]
        sv = expl.shap_values(X[top][:, cols])
        sv = sv[1] if isinstance(sv, list) else sv
        cells = []
        for j, t in enumerate(top):
            order = np.argsort(-sv[j])[:3]
            why = [REASON[FEATURES[cols[o]]](float(X[t, cols[o]])) for o in order
                   if not np.isnan(X[t, cols[o]])]
            cells.append({"cell": int(c[t]), "p": float(p[t]), "found_after_2009": bool(y[t]),
                          "reasons": why})
        auc = _per_group_auc(np.zeros(len(y), int), p, y)
        out.append({"species": species[s], "new_cells": int(y.sum()), "candidate_cells": int(len(y)),
                    "auc": auc[0] if auc else None, "map": str(path.name), "top_cells": cells})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=3_000_000)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    species, sets, rec = build(args.rebuild)
    tr, te = sets["train"], sets["test"]
    print(f"tables ready ({time.time()-t0:.0f}s): train {len(tr['y']):,} pairs, test "
          f"{len(te['y']):,} pairs, base rate {te['y'].mean():.4f}", flush=True)

    # 1. the comparison, and what each kind of evidence adds
    results, models, kept, lrs = {}, {}, {}, {}
    for name, groups in SUBSETS.items():
        cols = _cols(groups)
        gb, lr, imp = train(tr, cols, args.max_train)
        models[name] = (gb, cols)
        lrs[name] = (lr, imp)
        for label, sc in ((f"gradient boosting, {name}", gb.predict_proba(te["X"][:, cols])[:, 1]),
                          (f"logistic regression, {name}", lr.predict_proba(imp(te["X"]))[:, 1])):
            results[label] = _metrics(te, sc)
            kept[label] = sc
        print(f"  {name} ({time.time()-t0:.0f}s)", flush=True)
    for name, sc in _rules(te).items():
        results[name] = _metrics(te, sc)
        kept[name] = sc

    # per species, AUC and top-cell precision for the scores the paper compares
    shown = [f"{m}, {name}" for m in ("gradient boosting", "logistic regression")
             for name in SUBSETS] + ["system map (rule)", "nearest record", "prevalence"]
    ps = per_species(te, {k: kept[k] for k in shown})
    with io.open(OUT / "per_species.jsonl", "w", encoding="utf-8") as fh:
        for r in ps:
            fh.write(json.dumps({**r, "species": species[r["species"]]}, ensure_ascii=False) + "\n")
    plot_per_species(ps, OUT / "per_species_metrics.png")
    summ = summarize_species(ps)
    print(f"\nper species: AUC, and the {TOP_K} best cells against {TOP_K} at random")
    for k, v in summ.items():
        print(f"  {k:40s} AUC {v['median_auc']:.3f}  pooled lift {v['pooled_lift']:5.1f}  "
              f"species with a hit {v['share_with_hit']:.0%}  median lift {v['median_lift']:.1f}"
              f"  ({v['species']} species)")
    del kept
    print(f"\ntest: known to 2009, new occurrences 2010-2026")
    print(f"{'score':48s} {'AUC':>6s} {'AP':>6s} {'sp.AUC':>7s} {'R@20':>6s} {'R@100':>6s}")
    for k, r in results.items():
        print(f"{k:48s} {r['auc']:6.3f} {r['ap']:6.3f} {r['species_auc']:7.3f} "
              f"{r['cell_r20']:6.3f} {r['cell_r100']:6.3f}")

    # 2. what may I find in this park?
    # the system's model leaves collecting effort out: it adds nothing measurable and its
    # reasons would describe where collectors work, not where the plant grows
    gb, cols = models["without effort"]
    lr, imp = lrs["without effort"]
    pscores = {"gradient boosting": gb.predict_proba(te["X"][:, cols])[:, 1],
               "logistic regression": lr.predict_proba(imp(te["X"]))[:, 1], **_rules(te)}
    parks = park_eval(te, pscores, rec)
    print("\nparks (median over parks with >= 5 new species since 2010)")
    for k, r in parks.items():
        print(f"  {k:22s} " + "  ".join(f"{m} {v:.3f}" if isinstance(v, float) and m.startswith("recall")
                                        else f"{m} {v}" for m, v in r.items()))

    # 3. examples with reasons
    ex = examples(species, te, gb, cols, rec)
    print("\nexamples")
    for e in ex:
        print(f"  {e['species']}: {e['new_cells']} new cells of {e['candidate_cells']} candidates, "
              f"AUC {e['auc'] or 0:.2f}, map {e['map']}")
        for c in e["top_cells"]:
            print(f"    p={c['p']:.2f} found after 2009: {'sí' if c['found_after_2009'] else 'no'}; "
                  + "; ".join(c["reasons"]))

    import shap
    rng = np.random.default_rng(1)
    pick = rng.choice(len(te["y"]), min(20_000, len(te["y"])), replace=False)
    sv = shap.TreeExplainer(gb).shap_values(te["X"][pick][:, cols])
    sv = sv[1] if isinstance(sv, list) else sv
    imp_ = np.abs(sv).mean(0)
    (OUT / "results.json").write_text(json.dumps(
        {"evaluation": results, "per_species": summ, "parks": parks, "examples": ex,
         "shap": {FEATURES[c]: float(v) for c, v in zip(cols, imp_)},
         "n_train": int(len(tr["y"])), "n_test": int(len(te["y"])),
         "base_rate_test": float(te["y"].mean())}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
