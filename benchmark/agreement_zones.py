"""Where the expert range and a record-fitted model agree, and where they do not.

Under the time split of temporal_holdout.py (built from records up to the year the
species' volume was published, scored on records collected after it), each species'
country is cut into four zones by the Manual's range and a model fitted to the earlier
records (MaxEnt or the random forest of temporal_sdm.py, cut at the area of the
system's map):

  both          inside the Manual's range and selected by the model
  model only    selected by the model, outside the Manual's range
  Manual only   inside the Manual's range, not selected by the model
  neither

For each zone: its share of the country, its share of the collection sites (the null
of null_models.py), and its share of the later records. Lift over sites = records
share / sites share. Pooled over species (every later record counts once) and as the
median of per-species values where the zone is non-empty.

Run:  python -m benchmark.agreement_zones --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json

import numpy as np

from . import map_refine as R
from . import sdm_layer as S
from . import temporal_sdm as T

DATA = S.DATA
OUT = DATA / "agreement_zones.jsonl"
ZONES = ("both", "model only", "Manual only", "neither")


def score_species(species: str) -> dict | None:
    from mpcr_rag.schema import Ficha
    from mpcr_rag.store import pg_store
    from . import mapcompare as M
    from .forecast_leakage import PUBLISHED
    from .null_models import background
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
    pix = R.record_pixels([(r[0], r[1]) for r in got])
    keep = pix >= 0
    got = [g for g, k in zip(got, keep) if k]
    pix = pix[keep]
    recs = [(r[2], None if r[3] is None else ("Caribe" if "arib" in r[3] else "Pacífico"), r[4])
            for r in got]
    tr = [i for i, r in enumerate(got) if r[5] <= pub]
    te = [i for i, r in enumerate(got) if r[5] > pub]
    pres = np.unique(pix[tr])
    if len(te) < T.MIN_SIDE or len(pres) < S.MIN_TRAIN:
        return None
    X = S.features()
    manual = M.from_ficha(Ficha(**fj))
    man = M.mask(manual)
    base = M.mask(M.with_records(manual, [recs[i] for i in tr]))
    if man is None or base is None or not base.any():
        return None
    sites = background()[0]
    seed = f"{species}-temporal"
    out = {"species": species, "n_test": len(te), "models": {}}
    tep = pix[te]
    for name, fit in (("maxent", T.fit_maxent), ("rf", T.fit_rf)):
        suit, _ = fit(X, pres, seed)
        mod = suit >= float(np.quantile(suit, 1 - base.mean()))
        zones = {"both": man & mod, "model only": mod & ~man,
                 "Manual only": man & ~mod, "neither": ~man & ~mod}
        out["models"][name] = {z: {"area": float(m.mean()), "sites": int(m[sites].sum()),
                                   "records": int(m[tep].sum())}
                               for z, m in zones.items()}
    out["n_sites"] = int(sites.sum())
    return out


def report(rows: list[dict]) -> None:
    for name in ("maxent", "rf"):
        print(f"\n{name}: {len(rows)} species, {sum(r['n_test'] for r in rows)} later records")
        print(f"  {'zone':12s} {'area':>6s} {'sites':>6s} {'records':>8s} {'pooled lift':>11s} "
              f"{'median lift':>11s} {'species with records here':>26s}")
        for z in ZONES:
            area = np.mean([r["models"][name][z]["area"] for r in rows])
            s = sum(r["models"][name][z]["sites"] for r in rows) / sum(r["n_sites"] for r in rows)
            rec = sum(r["models"][name][z]["records"] for r in rows) / sum(r["n_test"] for r in rows)
            per = [(r["models"][name][z]["records"] / r["n_test"]) /
                   (r["models"][name][z]["sites"] / r["n_sites"])
                   for r in rows if r["models"][name][z]["sites"] > 0]
            withrec = sum(r["models"][name][z]["records"] > 0 for r in rows)
            print(f"  {z:12s} {100*area:5.1f}% {100*s:5.1f}% {100*rec:7.1f}% {rec/s:11.2f} "
                  f"{np.median(per):11.2f} {withrec:26d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report([json.loads(l) for l in io.open(OUT, encoding="utf-8")])
        return
    species = [r["species"] for r in json.loads(
        (DATA / "maps_sample.json").read_text(encoding="utf-8"))["sample"]]
    rows = []
    with cf.ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(score_species, species):
            if r:
                rows.append(r)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(rows)


if __name__ == "__main__":
    main()
