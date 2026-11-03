"""A modelled-habitat layer per species, as an input to the system, and its test.

The proposal rules out generating species distribution models as a product, but
counts existing models as an input. This module builds one per species offline, the
way the climate profile is built, and asks whether it improves the system's map:

  * the model alone, as a fourth map source;
  * the system's map cut by the model (Manual regions and slopes, kept only where the
    model finds habitat), which is the change the two botanists asked for when they
    judged some maps too broad or too high.

Model: a random forest on presence pixels against a random background of land
pixels, on the WorldClim 2.1 variables we hold at 30 arc-seconds plus elevation. The
nine variables are a subset of the nineteen, which keeps the collinear ones out.
Presences are deduplicated to one per pixel. Each tree sees as many background pixels as presences (a down-sampled forest). The
threshold that turns suitability into habitat is the 10th percentile of the
out-of-bag suitability of the training presences, so it is not set on the records
the map is scored on.

Scoring is map_refine's and climate_profile's: the same random halves, the model and
the map built from one half and capture counted on the other, both halves in turn.

Run:  python -m benchmark.sdm_layer --n 20 --split spatial   (stratified sample of the 300)
      python -m benchmark.sdm_layer --species "Hyptis brachiata"
"""
from __future__ import annotations

import argparse
import io
import json
import random
import statistics
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import map_refine as R

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
CLIMATE = ROOT / "data_raw" / "climate_rasters" / "clipped"
BIOCLIM = ("bio1", "bio2", "bio3", "bio4", "bio12", "bio15", "bio17", "bio18", "bio19")
FEATURES = BIOCLIM + ("elevation",)
N_BACKGROUND = 10_000
N_TREES = 500
THRESHOLD_Q = 0.10            # of the training presences' out-of-bag suitability
MIN_TRAIN = 15                # presence pixels needed to fit a model on one half
BLOCK = 20                    # pixels per side of a spatial block (about 20 km)


@lru_cache(maxsize=None)
def features() -> np.ndarray:
    """(land pixels x FEATURES). Coast pixels the climate rasters miss take the value
    of the nearest pixel that has one, so every land pixel can be predicted."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    from scipy import ndimage
    from utils.distribution_map.geodata import PATHS
    g = R.grid()
    land = g["index"] >= 0
    with rasterio.open(PATHS["dem"]) as dem:
        shape, transform, crs = dem.shape, dem.transform, dem.crs
    cols = []
    for v in BIOCLIM:
        with rasterio.open(CLIMATE / f"cr_30s_{v.replace('bio', 'bio_')}.tif") as src:
            dst = np.full(shape, np.nan, dtype=np.float32)
            reproject(source=rasterio.band(src, 1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                      dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                      resampling=Resampling.nearest)
        miss = np.isnan(dst)
        if miss.any():
            _, (ri, ci) = ndimage.distance_transform_edt(miss, return_indices=True)
            dst = dst[ri, ci]
        cols.append(dst[land])
    elev = np.asarray(g["elev"], dtype=np.float32)
    elev[np.isnan(elev)] = np.nanmedian(elev)      # 28 coast pixels the DEM misses
    cols.append(elev)
    return np.column_stack(cols)


def fit(train_pix: np.ndarray, seed: str):
    """Model and threshold from presence pixels, or None if there are too few.

    A down-sampled random forest (Valavi et al. 2021): each tree sees a bootstrap of
    the presences and as many background pixels, drawn afresh. With a few dozen
    presences against ten thousand background pixels an ordinary forest scores almost
    every presence near zero out of bag, and the threshold collapses to zero."""
    from sklearn.tree import DecisionTreeClassifier
    X = features()
    pres = np.unique(train_pix)
    if len(pres) < MIN_TRAIN:
        return None
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    bg = rng.choice(len(X), size=N_BACKGROUND, replace=False)
    bg = bg[~np.isin(bg, pres)]
    trees, oob_sum, oob_n = [], np.zeros(len(pres)), np.zeros(len(pres))
    for _ in range(N_TREES):
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
    return trees, float(np.quantile(oob, THRESHOLD_Q))


def suitability(trees) -> np.ndarray:
    X = features()
    return np.mean([t.predict_proba(X)[:, 1] for t in trees], axis=0).astype(np.float32)


def _auc(suit: np.ndarray, test_pix: np.ndarray, seed: str) -> float:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(zlib.crc32((seed + "-auc").encode()))
    bg = rng.choice(len(suit), size=N_BACKGROUND, replace=False)
    pres = np.unique(test_pix)
    y = np.r_[np.ones(len(pres)), np.zeros(len(bg))]
    return float(roc_auc_score(y, np.r_[suit[pres], suit[bg]]))


@lru_cache(maxsize=None)
def _rowcol() -> tuple[np.ndarray, np.ndarray]:
    """Grid row and column of each land pixel (the land index runs in row order)."""
    return np.nonzero(R.grid()["index"] >= 0)


def folds(pix: np.ndarray, species: str, split: str) -> list[tuple[list[int], list[int]]]:
    """Two train/test halves of the records.

    random:  map_refine's halves, so the system's numbers match the published ones.
             Half the test records share a 1 km pixel with a training record, which
             favours a model at that resolution over a map drawn by region.
    spatial: blocks of BLOCK x BLOCK pixels (about 20 km) go to one half or the other
             at random, so no test record has a training record in its block
             (block cross-validation, Roberts et al. 2017)."""
    if split == "random":
        rng = random.Random(f"{R.SEED}-{species}")
        order = list(range(len(pix)))
        rng.shuffle(order)
        half = len(order) // 2
        return [(order[:half], order[half:]), (order[half:], order[:half])]
    rows, cols = _rowcol()
    block = (rows[pix] // BLOCK) * 10_000 + cols[pix] // BLOCK
    rng = random.Random(f"{R.SEED}-{species}-blocks")
    side = {b: rng.random() < 0.5 for b in sorted(set(block.tolist()))}
    a = [i for i, b in enumerate(block) if side[b]]
    b = [i for i, b in enumerate(block) if not side[b]]
    return [(a, b), (b, a)]


def score_species(cur, species: str, split: str = "random") -> dict:
    from mpcr_rag.schema import Ficha
    cur.execute("SELECT ficha FROM mpcr.fichas WHERE species = %s", (species,))
    f = Ficha(**cur.fetchone()[0])
    lonlat, attrs = R._records(cur, species)
    pix = R.record_pixels(lonlat)
    keep = pix >= 0
    pix, attrs = pix[keep], [a for a, k in zip(attrs, keep) if k]
    ing = R.ingredients(f)
    folds_ = folds(pix, species, split)
    names = ("system", "model", "system cut by model")
    acc = {k: {"cap": 0, "area": []} for k in names}
    aucs, fitted = [], 0
    for k, (train, test) in enumerate(folds_):
        base = R.mask(R.widen(ing, [attrs[i] for i in train]), True, False)
        model = fit(pix[train], f"{species}-{k}") if test else None
        if model is None:
            hab = np.ones(len(base), dtype=bool)      # no model: nothing is cut
        else:
            fitted += 1
            suit = suitability(model[0])
            hab = suit >= model[1]
            aucs.append(_auc(suit, pix[test], f"{species}-{k}"))
        for name, mk in (("system", base), ("model", hab), ("system cut by model", base & hab)):
            acc[name]["cap"] += int(mk[pix[test]].sum())
            acc[name]["area"].append(float(mk.mean()))
    out = {"species": species, "split": split, "n": len(pix), "fitted": fitted,
           "auc": float(np.mean(aucs)) if aucs else None}
    for name in names:
        out[name] = {"held_capture": acc[name]["cap"] / max(1, len(pix)),
                     "area": float(np.mean(acc[name]["area"]))}
    return out


def _sample(n: int) -> list[str]:
    sample = json.loads((DATA / "maps_sample.json").read_text(encoding="utf-8"))["sample"]
    rng = random.Random("sdm-check")
    out = []
    for i, g in enumerate(("G1", "G2", "G3")):
        pool = sorted(r["species"] for r in sample if r["stratum"] == g)
        rng.shuffle(pool)
        out += pool[: n // 3 + (1 if i < n % 3 else 0)]
    return out


def report(rows: list[dict]) -> None:
    rows = [r for r in rows if r["fitted"] == 2]
    print(f"\n{len(rows)} species with a model on both halves; median per species")
    print(f"{'map':22s} {'capture':>8s} {'area':>7s} {'lift':>6s}")
    for name in ("system", "model", "system cut by model"):
        cap = [r[name]["held_capture"] for r in rows]
        area = [r[name]["area"] for r in rows]
        lift = [c / a for c, a in zip(cap, area) if a > 0]
        print(f"{name:22s} {statistics.median(cap):8.1%} {statistics.median(area):7.1%} "
              f"{statistics.median(lift):6.1f}")
    print(f"held-out AUC: median {statistics.median(r['auc'] for r in rows):.3f}")
    lost = [r["system"]["held_capture"] - r["system cut by model"]["held_capture"] for r in rows]
    saved = [1 - r["system cut by model"]["area"] / r["system"]["area"] for r in rows
             if r["system"]["area"] > 0]
    print(f"cutting the system's map by the model: median area removed {statistics.median(saved):.0%},"
          f" median capture lost {statistics.median(lost):.1%} points;"
          f" {sum(l > .10 for l in lost)} species lose more than 10 points")


def main() -> None:
    from mpcr_rag.store import pg_store
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="")
    ap.add_argument("--n", type=int, default=0, help="stratified sample of the 300")
    ap.add_argument("--split", choices=("random", "spatial"), default="random")
    args = ap.parse_args()
    if args.species:
        species = [s.strip() for s in args.species.split(",") if s.strip()]
    elif args.n:
        species = _sample(args.n)
    else:
        species = [r["species"] for r in json.loads(
            (DATA / "maps_sample.json").read_text(encoding="utf-8"))["sample"]]
    conn = pg_store.connect()
    rows = []
    with conn.cursor() as cur:
        for i, sp in enumerate(species, 1):
            rows.append(score_species(cur, sp, args.split))
            r = rows[-1]
            print(f"  {i:3d}/{len(species)} {sp:34s} n={r['n']:5d} auc={r['auc'] or 0:.2f}  "
                  + "  ".join(f"{k}: {r[k]['held_capture']:.0%}/{r[k]['area']:.0%}"
                              for k in ("system", "model", "system cut by model")), flush=True)
    conn.close()
    stem = "sdm_scores" if not (args.species or args.n) else "sdm_check"
    out = DATA / f"{stem}_{args.split}.jsonl"
    with io.open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(rows)


if __name__ == "__main__":
    main()
