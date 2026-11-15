"""Record-fitted distribution models against the system's map, more than one of them.

sdm_layer.py tested one model, a down-sampled random forest, at one threshold. A
reviewer asked whether the result holds for other methods and thresholds. This module
scores, on the same species, splits, features and background:

  rf                 the down-sampled random forest of sdm_layer.fit (unchanged)
  maxent_b{1,2,4}    MaxEnt (elapid, Anderson 2023, which follows maxnet: linear,
                     quadratic, hinge and product features), at regularisation
                     multipliers 1, 2 and 4. "maxent tuned" is the multiplier with the
                     best median held-out capture-over-area on the spatial split.

Each model's suitability is cut at the 5th, 10th and 20th percentile of the training
presences' suitability (out of bag for the forest, in sample for MaxEnt, as is usual
for it), and also at the system map's own area ("@matched"), since the threshold sets
area and area is the denominator of lift: at equal area, capture alone compares them. The system's map, and the system's map cut by each model, are scored too.
Capture is counted on the held-out half (random halves, or ~20 km blocks), area is the
share of the country covered, and the expected capture under the collection-site null
of null_models.py is stored as well.

Writes data/sdm_compare_{split}.jsonl.

Run:  python -m benchmark.sdm_compare --split spatial --workers 8
      python -m benchmark.sdm_compare --report
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
THRESHOLDS = (0.05, 0.10, 0.20)
BETAS = (1.0, 2.0, 4.0)


def fit_maxent(train_pix: np.ndarray, seed: str, beta: float):
    """MaxEnt suitability over every land pixel and the training presences' values,
    or None with too few presences. Same background draw as the forest."""
    import elapid
    X = S.features()
    pres = np.unique(train_pix)
    if len(pres) < S.MIN_TRAIN:
        return None
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    bg = rng.choice(len(X), size=S.N_BACKGROUND, replace=False)
    bg = bg[~np.isin(bg, pres)]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    m = elapid.MaxentModel(feature_types=["linear", "quadratic", "hinge", "product"],
                           beta_multiplier=beta, n_cpus=1, random_state=0)
    m.fit(np.vstack([Z[pres], Z[bg]]), np.r_[np.ones(len(pres)), np.zeros(len(bg))])
    suit = m.predict(Z).astype(np.float32)
    return suit, suit[pres]


def fit_rf(train_pix: np.ndarray, seed: str):
    """The forest of sdm_layer, with its out-of-bag presence values kept so that every
    threshold can be applied, not only the 10th percentile."""
    from sklearn.tree import DecisionTreeClassifier
    X = S.features()
    pres = np.unique(train_pix)
    if len(pres) < S.MIN_TRAIN:
        return None
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    bg = rng.choice(len(X), size=S.N_BACKGROUND, replace=False)
    bg = bg[~np.isin(bg, pres)]
    trees, oob_sum, oob_n = [], np.zeros(len(pres)), np.zeros(len(pres))
    for _ in range(S.N_TREES):
        ip = rng.integers(len(pres), size=len(pres))
        ib = rng.choice(bg, size=len(pres), replace=False)
        Xt = np.vstack([X[pres[ip]], X[ib]])
        yt = np.r_[np.ones(len(ip)), np.zeros(len(ib))]
        t = DecisionTreeClassifier(max_features="sqrt", random_state=int(rng.integers(2**31)))
        t.fit(Xt, yt)
        trees.append(t)
        out = np.setdiff1d(np.arange(len(pres)), ip)
        if len(out):
            oob_sum[out] += t.predict_proba(X[pres[out]])[:, 1]
            oob_n[out] += 1
    oob = oob_sum[oob_n > 0] / oob_n[oob_n > 0]
    return S.suitability(trees), oob


def score_species(species: str, split: str) -> dict:
    from mpcr_rag.schema import Ficha
    from mpcr_rag.store import pg_store
    from . import mapcompare as M
    from .null_models import expected
    from .rq3_split import folds as split_folds
    conn = pg_store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT ficha FROM mpcr.fichas WHERE species = %s", (species,))
        manual = M.from_ficha(Ficha(**cur.fetchone()[0]))
        cur.execute("""SELECT o.lon, o.lat, o.region, o.vertiente, o.elev_dem
                       FROM mpcr.occurrences o
                       JOIN mpcr.species_evidence e ON e.taxon_key = o.species_key
                       WHERE e.species = %s""", (species,))
        got = cur.fetchall()
    conn.close()
    lonlat = [(r[0], r[1]) for r in got]
    recs = [(r[2], None if r[3] is None else ("Caribe" if "arib" in r[3] else "Pacífico"), r[4])
            for r in got]
    pix = R.record_pixels(lonlat)
    keep = pix >= 0
    pix, recs = pix[keep], [a for a, k in zip(recs, keep) if k]
    lonlat = [p for p, k in zip(lonlat, keep) if k]
    acc: dict[str, dict] = {}
    fitted = 0

    def add(name, mk, test):
        a = acc.setdefault(name, {"cap": 0, "area": [], "p_sites": []})
        a["cap"] += int(mk[pix[test]].sum())
        e = expected(mk)
        a["area"].append(e["area"])
        a["p_sites"].append(e["p_sites"])

    for k, (train, test) in enumerate(split_folds(species, lonlat, split)):
        if not train or not test:
            continue
        base = M.mask(M.with_records(manual, [recs[i] for i in train]))
        if base is None:
            base = np.zeros(len(S.features()), dtype=bool)
        add("system", base, test)
        models = {"rf": fit_rf(pix[train], f"{species}-{k}")}
        for b in BETAS:
            models[f"maxent_b{b:g}"] = fit_maxent(pix[train], f"{species}-{k}", b)
        if any(v is None for v in models.values()):
            continue
        fitted += 1
        for name, (suit, pres_vals) in models.items():
            # Cut at the system map's own area: the threshold sets area, and area is
            # the denominator of lift, so models are compared with the map like for like.
            if base.any():
                t = float(np.quantile(suit, 1 - base.mean()))
                add(f"{name}@matched", suit >= t, test)
            for q in THRESHOLDS:
                hab = suit >= float(np.quantile(pres_vals, q))
                add(f"{name}@{int(q * 100)}", hab, test)
                add(f"system cut by {name}@{int(q * 100)}", base & hab, test)
    out = {"species": species, "split": split, "n": len(pix), "fitted": fitted}
    for name, a in acc.items():
        out[name] = {"held_capture": a["cap"] / max(1, len(pix)),
                     "area": float(np.mean(a["area"])), "p_sites": float(np.mean(a["p_sites"]))}
    return out


def report(split: str) -> None:
    rows = [json.loads(l) for l in io.open(DATA / f"sdm_compare_{split}.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["fitted"] == 2]
    print(f"\n{split} split: {len(rows)} species with every model fitted on both halves")
    names = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    print(f"{'map':32s} {'capture':>8s} {'area':>6s} {'lift':>5s} {'lift_sites':>10s}")
    for name in names:
        cap = [r[name]["held_capture"] for r in rows]
        area = [r[name]["area"] for r in rows]
        lift = [r[name]["held_capture"] / r[name]["area"] for r in rows if r[name]["area"] > 0]
        ls = [r[name]["held_capture"] / r[name]["p_sites"] for r in rows if r[name]["p_sites"] > 0]
        print(f"{name:32s} {statistics.median(cap):8.1%} {statistics.median(area):6.1%} "
              f"{statistics.median(lift):5.2f} {statistics.median(ls):10.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("random", "spatial"), default="spatial")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=0, help="first n species only (a smoke test)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        for split in ("random", "spatial"):
            if (DATA / f"sdm_compare_{split}.jsonl").exists():
                report(split)
        return
    species = [r["species"] for r in json.loads(
        (DATA / "maps_sample.json").read_text(encoding="utf-8"))["sample"]]
    if args.n:
        species = species[: args.n]
    out = DATA / f"sdm_compare_{args.split}.jsonl"
    done = set()
    if out.exists() and not args.n:
        done = {json.loads(l)["species"] for l in io.open(out, encoding="utf-8")}
    todo = [s for s in species if s not in done]
    print(f"{len(todo)} species to score ({len(done)} on disk)", flush=True)
    with io.open(out, "a", encoding="utf-8") as fh, \
            cf.ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(score_species, s, args.split): s for s in todo}
        for i, f in enumerate(cf.as_completed(futs), 1):
            fh.write(json.dumps(f.result(), ensure_ascii=False) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)
    report(args.split)


if __name__ == "__main__":
    main()
