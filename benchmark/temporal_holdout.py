"""The map and the climate profiles held out in time: built from what was collected up to
the year the species' volume of the Manual was published, scored on what came after.

The random and spatial splits (rq3_split.py) hold records out from the records-built
part of the map, but not from the Manual: its authors wrote the distributions from
herbarium specimens, many of which are in the GBIF download, so a held-out record may
be one the Manual's authors saw. Records collected after the volume was published
cannot have informed it. Here every source gets the same information horizon:

  manual          the Manual's region-slope pairs inside its elevation band, no records
  map             the system's map: the Manual's pairs plus those holding >= 5 records
                  collected up to the publication year
  clim_records    the 5th-95th percentile envelope over those records, inside the band
  clim_map        the same envelope over the ground the map covers

Two test sets, per species with >= 5 records on each side of the publication year:

  temporal          every record collected after the publication year
  temporal_spatial  those of them in a block of about 20 km (rq3_split.BLOCK) holding
                    no record from up to the publication year: new ground, new time

Rows carry the same nulls and block bootstrap as null_models.py. No model is called.

Writes data/temporal_holdout_scores.jsonl.

Run:  python -m benchmark.temporal_holdout            (score and report)
      python -m benchmark.temporal_holdout --report   (report from the saved scores)
"""
from __future__ import annotations

import argparse
import io
import json
import statistics as st

import numpy as np

from . import climate_profile as C
from . import map_refine as R
from . import mapcompare as M
from . import null_models as N
from . import rq3_split as Q
from .climate_compare import env_mask
from .forecast_leakage import PUBLISHED

DATA = M.DATA
OUT = DATA / "temporal_holdout_scores.jsonl"
MIN_SIDE = 5


def score() -> list[dict]:
    from mpcr_rag.schema import Ficha
    from mpcr_rag.store import pg_store
    clim = C.climate()
    n_land = int((R.grid()["index"] >= 0).sum())
    rows = []
    conn = pg_store.connect()
    with conn.cursor() as cur:
        for k, sp in enumerate(N._species(), 1):
            cur.execute("SELECT ficha, volume FROM mpcr.fichas WHERE species = %s", (sp,))
            fj, vol = cur.fetchone()
            pub = PUBLISHED[vol]
            manual = M.from_ficha(Ficha(**fj))
            cur.execute("""SELECT o.lon, o.lat, o.region, o.vertiente, o.elev_dem, o.year,
                                  o.basis
                           FROM mpcr.occurrences o
                           JOIN mpcr.species_evidence e ON e.taxon_key = o.species_key
                           WHERE e.species = %s AND o.year IS NOT NULL""", (sp,))
            got = cur.fetchall()
            tr = [i for i, r in enumerate(got) if r[5] <= pub]
            te = [i for i, r in enumerate(got) if r[5] > pub]
            if len(tr) < MIN_SIDE or len(te) < MIN_SIDE:
                continue
            lonlat = [(r[0], r[1]) for r in got]
            recs = [(r[2], None if r[3] is None else
                     ("Caribe" if "arib" in r[3] else "Pacífico"), r[4]) for r in got]
            pix = R.record_pixels(lonlat)
            band = np.zeros(len(pix), bool)
            band[pix >= 0] = C.in_band(cur, sp, pix[pix >= 0])

            ing = M.with_records(manual, [recs[i] for i in tr])
            mk_map, mk_man = M.mask(ing), M.mask(manual)
            trp = [i for i in tr if pix[i] >= 0]
            inb = [i for i in trp if band[i]] or trp
            e_rec = Q._env(clim, pix[inb]) if inb else None
            e_map = Q._env(clim, np.flatnonzero(mk_map)) if mk_map is not None and mk_map.any() else None
            m_rec = env_mask(e_rec) if e_rec else np.zeros(n_land, bool)
            m_cm = env_mask(e_map) if e_map else np.zeros(n_land, bool)

            seen = {b for i, b in enumerate(N._blocks(lonlat)) if i in set(tr)}
            blocks = N._blocks(lonlat)
            sets = {"temporal": te,
                    "temporal_spatial": [i for i in te if blocks[i] not in seen]}
            for analysis, idx in sets.items():
                if len(idx) < MIN_SIDE:
                    continue
                ll = [lonlat[i] for i in idx]
                px = pix[idx]
                for source, inside, mk in (
                        ("manual", N._inside_map(manual, [recs[i] for i in idx]), mk_man),
                        ("map", N._inside_map(ing, [recs[i] for i in idx]), mk_map)):
                    r = N.row(sp, analysis, source, inside, ll, px, [mk])
                    r.update(volume=vol, published=pub, n_train=len(tr),
                             n_obs=sum(got[i][6] == "HUMAN_OBSERVATION" for i in idx))
                    rows.append(r)
                ok = px >= 0
                for source, em in (("clim_records", m_rec), ("clim_map", m_cm)):
                    inside = np.zeros(len(idx), bool)
                    inside[ok] = em[px[ok]]
                    keep = np.flatnonzero(ok)
                    r = N.row(sp, analysis, source, inside[keep], [ll[i] for i in keep],
                              px[keep], [em])
                    r.update(volume=vol, published=pub, n_train=len(tr))
                    rows.append(r)
            if k % 50 == 0:
                print(f"  {k}/300", flush=True)
    conn.close()
    with io.open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def report(rows: list[dict]) -> None:
    for analysis in ("temporal", "temporal_spatial"):
        sub = [r for r in rows if r["analysis"] == analysis]
        sp = {r["species"] for r in sub}
        print(f"\n{analysis}: {len(sp)} species, "
              f"{sum(r['n'] for r in sub if r['source'] == 'map')} test records")
        print(f"  {'source':14s} {'capture':>8s} {'area':>6s} {'lift':>6s} {'site lift':>9s} "
              f"{'beats (blocks)':>14s}")
        for source in ("manual", "map", "clim_records", "clim_map"):
            s = [r for r in sub if r["source"] == source and r["area"] > 0]
            cap = st.median(r["capture"] for r in s)
            area = st.median(r["area"] for r in s)
            lift = st.median(r["capture"] / r["area"] for r in s)
            sl = st.median(r["capture"] / r["p_sites"] for r in s if r["p_sites"] > 0)
            beats = sum(r["boot_q05"] > r["p_sites"] for r in s)
            print(f"  {source:14s} {100*cap:7.1f}% {100*area:5.1f}% {lift:6.2f} {sl:9.2f} "
                  f"{beats:6d} of {len(s)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    rows = ([json.loads(l) for l in io.open(OUT, encoding="utf-8")] if a.report
            else score())
    report(rows)


if __name__ == "__main__":
    main()
