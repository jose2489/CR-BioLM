"""Are the map results carried by records with suspect coordinates?

Occurrence cleaning (mpcr_rag ingest) filters by taxon key, bounding box and coordinate
uncertainty, but runs no test for centroids or institutional coordinates. A record
georeferenced to a province's centroid, or to a herbarium or garden, could fall inside
or outside a map for reasons unrelated to where the plant grows. Two tests in the spirit
of CoordinateCleaner (Zizka et al. 2019), written here because that package is not in
this environment:

  centroid       within 1 km of the centroid of Costa Rica or of one of its seven
                 provinces (CoordinateCleaner::cc_cen, default buffer 1 km)
  institution    a coordinate (rounded to 4 decimals, about 10 m) shared by records of
                 at least 50 species: herbaria, gardens and stations collect everything
                 at one point (a proxy for cc_inst, whose gazetteer is not available)

The system's map is then rescored on the 300-species sample under the random split
(rq3_split.folds) with the flagged records removed from the test halves, and compared
with the published score on the same species.

Run:  python -m benchmark.coord_check
"""
from __future__ import annotations

import collections
import json
import statistics as st

import numpy as np

from . import mapcompare as M
from . import null_models as N
from . import rq3_split as Q

INST_SPECIES = 50
CENT_KM = 1.0


def _centroids() -> list[tuple[float, float]]:
    import geopandas as gpd
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "data_raw" / "Cartografia" / "IGN_5_limite_Provincial.shp"
    g = gpd.read_file(p).to_crs(5367)              # CRTM05, so centroids are planar
    pts = [(c.x, c.y) for c in g.geometry.centroid.to_crs(4326)]
    w = gpd.GeoSeries([g.geometry.union_all().centroid], crs=5367).to_crs(4326).iloc[0]
    return pts + [(w.x, w.y)]


def main() -> None:
    from mpcr_rag.schema import Ficha
    from mpcr_rag.store import pg_store
    cents = np.array(_centroids())
    conn = pg_store.connect()
    with conn.cursor() as cur:
        cur.execute("""SELECT round(lon::numeric, 4), round(lat::numeric, 4),
                              count(DISTINCT species_key)
                       FROM mpcr.occurrences GROUP BY 1, 2 HAVING count(DISTINCT species_key) >= %s""",
                    (INST_SPECIES,))
        inst = {(float(a), float(b)) for a, b, _ in cur.fetchall()}
        rows = {"all": [], "clean": []}
        flagged = collections.Counter()
        n_rec = 0
        for sp in N._species():
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
            ll = np.array(lonlat)
            dkm = np.sqrt(((ll[:, None, 0] - cents[None, :, 0]) * 108.8) ** 2
                          + ((ll[:, None, 1] - cents[None, :, 1]) * 110.6) ** 2).min(1)
            cen = dkm <= CENT_KM
            ins = np.array([(round(a, 4), round(b, 4)) in inst for a, b in lonlat])
            bad = cen | ins
            flagged["centroid"] += int(cen.sum())
            flagged["institution"] += int(ins.sum())
            flagged["either"] += int(bad.sum())
            n_rec += len(got)
            cap = {"all": [0, 0], "clean": [0, 0]}
            for tr, te in Q.folds(sp, lonlat, "random"):
                ing = M.with_records(manual, [recs[i] for i in tr])
                inside = N._inside_map(ing, [recs[i] for i in te])
                ok = ~bad[te]
                cap["all"][0] += int(inside.sum()); cap["all"][1] += len(te)
                cap["clean"][0] += int(inside[ok].sum()); cap["clean"][1] += int(ok.sum())
            for k, (c, n) in cap.items():
                if n:
                    rows[k].append((sp, c / n))
    conn.close()
    a, b = dict(rows["all"]), dict(rows["clean"])
    common = sorted(set(a) & set(b))
    d = np.array([b[s] - a[s] for s in common])
    out = {"records": n_rec, "flagged": dict(flagged),
           "institution_points": len(inst),
           "median_capture_all": st.median(a[s] for s in common),
           "median_capture_clean": st.median(b[s] for s in common),
           "median_change": float(np.median(d)), "mean_change": float(d.mean()),
           "species_changed_over_2pts": int((np.abs(d) > .02).sum()), "species": len(common)}
    print(json.dumps(out, indent=1))
    (M.DATA / "coord_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
