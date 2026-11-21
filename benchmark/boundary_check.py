"""How much do the map results depend on where the region boundaries were drawn?

The botanical-region layer was drawn by hand from a printed map; another hand would
place some boundaries differently. Shifting every boundary at once is approximated by
growing or shrinking the system's map by 1 and 2 pixels (about 1 and 2 km) and
rescoring it, by pixel, on the held-out records of the random split (rq3_split.folds),
for the 300-species sample. The unshifted map is scored by pixel too, so the four
versions differ only in their boundaries.

Run:  python -m benchmark.boundary_check
"""
from __future__ import annotations

import json
import statistics as st

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from . import map_refine as R
from . import mapcompare as M
from . import null_models as N
from . import rq3_split as Q

SHIFTS = (-2, -1, 0, 1, 2)


def main() -> None:
    from mpcr_rag.schema import Ficha
    from mpcr_rag.store import pg_store
    idx = R.grid()["index"]
    land = idx >= 0
    rows = {k: [] for k in SHIFTS}
    conn = pg_store.connect()
    with conn.cursor() as cur:
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
            pix = R.record_pixels(lonlat)
            acc = {k: [0, 0, []] for k in SHIFTS}
            for tr, te in Q.folds(sp, lonlat, "random"):
                mk = M.mask(M.with_records(manual, [recs[i] for i in tr]))
                if mk is None:
                    continue
                grid = np.zeros(idx.shape, bool)
                grid[land] = mk[idx[land]]
                tp = pix[te]
                tp = tp[tp >= 0]
                for k in SHIFTS:
                    g = (grid if k == 0 else
                         binary_dilation(grid, iterations=k) if k > 0 else
                         binary_erosion(grid, iterations=-k)) & land
                    m = np.zeros(int(land.sum()), bool)
                    m[idx[g]] = True
                    acc[k][0] += int(m[tp].sum())
                    acc[k][1] += len(tp)
                    acc[k][2].append(float(m.mean()))
            for k, (c, n, a) in acc.items():
                if n:
                    rows[k].append({"species": sp, "capture": c / n, "area": float(np.mean(a))})
    conn.close()
    out = {}
    base = {r["species"]: r for r in rows[0]}
    for k in SHIFTS:
        r = rows[k]
        d = [x["capture"] - base[x["species"]]["capture"] for x in r]
        out[k] = {"median_capture": st.median(x["capture"] for x in r),
                  "median_area": st.median(x["area"] for x in r),
                  "median_lift": st.median(x["capture"] / x["area"] for x in r if x["area"] > 0),
                  "median_capture_change": st.median(d)}
        print(f"shift {k:+d} km: capture {100*out[k]['median_capture']:.1f}%  area "
              f"{100*out[k]['median_area']:.1f}%  lift {out[k]['median_lift']:.2f}  "
              f"capture change {100*out[k]['median_capture_change']:+.1f} pts")
    (M.DATA / "boundary_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
