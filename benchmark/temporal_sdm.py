"""Record-fitted models against the Manual's range, held out in time, on equal terms.

A second review found the comparison of sdm_compare.py unfair in three ways: the models
drew background from the whole country while the evaluation corrects for where plants
are collected; the MaxEnt multiplier shown was chosen on the test data; and no model
combined the records with the Manual's range. It also found that held-out records may
have informed the Manual. Here every source sees only what was collected up to the
year the species' volume was published (temporal_holdout.py) and is scored on what came
after, and:

  rf            the down-sampled random forest of sdm_layer, background drawn from
                collection sites (target group, Phillips et al. 2009)
  maxent        MaxEnt (elapid) at the default multiplier 1, fixed in advance, same
                target-group background
  rf+manual     the forest with the Manual's range (region-slope pairs inside its
                elevation band) as one more predictor: records and expert range
                combined (Merow et al. 2017 use the range as an offset; a covariate is
                the simplest form of the same idea)

Each model is cut at the area of the system's map for that species, so capture alone
compares them, and at the 10th percentile of training presences. Test sets as in
temporal_holdout.py: every later record, and later records on new ground (20 km blocks
holding no earlier record).

Writes data/temporal_sdm.jsonl.

Run:  python -m benchmark.temporal_sdm --workers 8
      python -m benchmark.temporal_sdm --report
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import statistics
import zlib

import numpy as np

from . import map_refine as R
from . import sdm_layer as S

DATA = S.DATA
OUT = DATA / "temporal_sdm.jsonl"
MIN_SIDE = 5


def _bg(seed: str, pres: np.ndarray) -> np.ndarray:
    from .null_models import background
    sites = np.flatnonzero(background()[0])
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    bg = rng.choice(sites, size=min(S.N_BACKGROUND, len(sites)), replace=False)
    return bg[~np.isin(bg, pres)]


def fit_rf(X: np.ndarray, pres: np.ndarray, seed: str):
    from sklearn.tree import DecisionTreeClassifier
    bg = _bg(seed, pres)
    rng = np.random.default_rng(zlib.crc32((seed + "rf").encode()))
    trees, oob_sum, oob_n = [], np.zeros(len(pres)), np.zeros(len(pres))
    for _ in range(S.N_TREES):
        ip = rng.integers(len(pres), size=len(pres))
        ib = rng.choice(bg, size=len(pres), replace=False)
        t = DecisionTreeClassifier(max_features="sqrt", random_state=int(rng.integers(2**31)))
        t.fit(np.vstack([X[pres[ip]], X[ib]]), np.r_[np.ones(len(ip)), np.zeros(len(ib))])
        trees.append(t)
        out = np.setdiff1d(np.arange(len(pres)), ip)
        if len(out):
            oob_sum[out] += t.predict_proba(X[pres[out]])[:, 1]
            oob_n[out] += 1
    suit = np.mean([t.predict_proba(X)[:, 1] for t in trees], axis=0).astype(np.float32)
    return suit, oob_sum[oob_n > 0] / oob_n[oob_n > 0]


def fit_maxent(X: np.ndarray, pres: np.ndarray, seed: str):
    import elapid
    bg = _bg(seed, pres)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    m = elapid.MaxentModel(feature_types=["linear", "quadratic", "hinge", "product"],
                           beta_multiplier=1.0, n_cpus=1, random_state=0)
    m.fit(np.vstack([Z[pres], Z[bg]]), np.r_[np.ones(len(pres)), np.zeros(len(bg))])
    suit = m.predict(Z).astype(np.float32)
    return suit, suit[pres]


def score_species(species: str) -> dict | None:
    from mpcr_rag.schema import Ficha
    from mpcr_rag.store import pg_store
    from . import mapcompare as M
    from .forecast_leakage import PUBLISHED
    from .null_models import _blocks, expected
    conn = pg_store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT ficha, volume FROM mpcr.fichas WHERE species = %s", (species,))
        fj, vol = cur.fetchone()
        cur.execute("""SELECT o.lon, o.lat, o.region, o.vertiente, o.elev_dem, o.year
                       FROM mpcr.occurrences o
                       JOIN mpcr.species_evidence e ON e.taxon_key = o.species_key
                       WHERE e.species = %s AND o.year IS NOT NULL""", (species,))
        got = cur.fetchall()
    conn.close()
    pub = PUBLISHED[vol]
    manual = M.from_ficha(Ficha(**fj))
    lonlat = [(r[0], r[1]) for r in got]
    pix = R.record_pixels(lonlat)
    keep = pix >= 0
    got = [g for g, k in zip(got, keep) if k]
    lonlat = [p for p, k in zip(lonlat, keep) if k]
    pix = pix[keep]
    recs = [(r[2], None if r[3] is None else ("Caribe" if "arib" in r[3] else "Pacífico"), r[4])
            for r in got]
    tr = [i for i, r in enumerate(got) if r[5] <= pub]
    te = [i for i, r in enumerate(got) if r[5] > pub]
    pres = np.unique(pix[tr])
    if len(te) < MIN_SIDE or len(pres) < S.MIN_TRAIN:
        return None
    blocks = _blocks(lonlat)
    seen = {blocks[i] for i in tr}
    tests = {"temporal": np.array(te),
             "temporal_spatial": np.array([i for i in te if blocks[i] not in seen])}

    X = S.features()
    man = M.mask(manual)
    man = np.zeros(len(X), bool) if man is None else man
    base = M.mask(M.with_records(manual, [recs[i] for i in tr]))
    base = np.zeros(len(X), bool) if base is None else base
    Xm = np.hstack([X, man[:, None].astype(X.dtype)])
    seed = f"{species}-temporal"
    models = {"rf": fit_rf(X, pres, seed), "maxent": fit_maxent(X, pres, seed),
              "rf+manual": fit_rf(Xm, pres, seed)}

    maps = {"manual": man, "system": base}
    for name, (suit, pv) in models.items():
        if base.any():
            maps[f"{name}@matched"] = suit >= float(np.quantile(suit, 1 - base.mean()))
        maps[f"{name}@10"] = suit >= float(np.quantile(pv, 0.10))
    out = {"species": species, "volume": vol, "published": pub, "n_train": len(tr)}
    for tname, idx in tests.items():
        if len(idx) < MIN_SIDE:
            continue
        out[tname] = {"n": int(len(idx))}
        for mname, mk in maps.items():
            e = expected(mk)
            out[tname][mname] = {"capture": float(mk[pix[idx]].mean()), "area": e["area"],
                                 "p_sites": e["p_sites"]}
    return out


def report() -> None:
    from scipy.stats import wilcoxon
    rows = [json.loads(l) for l in io.open(OUT, encoding="utf-8")]
    for t in ("temporal", "temporal_spatial"):
        sub = [r[t] for r in rows if t in r]
        print(f"\n{t}: {len(sub)} species")
        print(f"  {'map':18s} {'capture':>8s} {'area':>6s} {'lift':>5s} {'site lift':>9s}  vs system (paired)")
        for name in sub[0]:
            if name == "n":
                continue
            s = [r for r in sub if name in r]
            cap = [r[name]["capture"] for r in s]
            line = (f"  {name:18s} {statistics.median(cap):8.1%} "
                    f"{statistics.median(r[name]['area'] for r in s):6.1%} "
                    f"{statistics.median(r[name]['capture'] / r[name]['area'] for r in s if r[name]['area'] > 0):5.2f} "
                    f"{statistics.median(r[name]['capture'] / r[name]['p_sites'] for r in s if r[name]['p_sites'] > 0):9.2f}")
            if name.endswith("@matched"):
                d = np.array([r[name]["capture"] - r["system"]["capture"] for r in s])
                p = wilcoxon(d).pvalue if np.any(d) else 1.0
                line += f"  median diff {100*np.median(d):+.1f} pts, mean {100*d.mean():+.1f}, p={p:.3g}"
            print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    species = [r["species"] for r in json.loads(
        (DATA / "maps_sample.json").read_text(encoding="utf-8"))["sample"]]
    if a.n:
        species = species[: a.n]
    rows = []
    with cf.ProcessPoolExecutor(a.workers) as pool:
        futs = [pool.submit(score_species, s) for s in species]
        for i, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            if r:
                rows.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(species)}", flush=True)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda r: r["species"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    report()


if __name__ == "__main__":
    main()
