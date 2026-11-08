"""Data for a page on which the forecast's predictions can be read one by one.

For each species of the 300-species map sample that has a new cell in the test window,
the model's score in every candidate cell (what was known by 2009), the cells where it
turned up from 2010, its records before and after, and for its ten best-scored cells
the reasons (largest SHAP contributions) and where the cell is. For each protected
area with at least five new species since 2010, the twenty species the model ranks
first. The model is the system's: gradient boosting without collecting effort.

Each species also gets the system's distribution map (the validated renderer), drawn
from the records known by 2009, with its ten best-scored cells numbered (green where it
turned up after 2009) and the records from 2010 as white crosses.

Run:  python -m benchmark.forecast_viewer          (writes data/forecast/viewer.json)
      python -m benchmark.forecast_viewer --maps   (also data/forecast/viewer_maps/*.jpg)
"""
from __future__ import annotations

import collections
import json

import numpy as np

from . import forecast as F
from . import map_refine as R

TOP = 10
PARK_TOP = 20


def cell_places(G: F.Grid) -> list[dict]:
    from pyproj import Transformer
    g = R.grid()
    xs, ys = g["transform"] * (G.cc + .5, G.cr + .5)
    lon, lat = Transformer.from_crs(g["crs"], 4326, always_xy=True).transform(xs, ys)
    out = []
    order = np.argsort(G.pc, kind="stable")
    cuts = np.flatnonzero(np.diff(G.pc[order])) + 1
    for c, px in enumerate(np.split(order, cuts)):
        el = G.elev[px]
        el = el[~np.isnan(el)]
        region, slope = G.pair_codes[np.bincount(G.pair_id[px]).argmax()].split("|")
        out.append({"lat": round(float(lat[c]), 3), "lon": round(float(lon[c]), 3),
                    "elev": [int(np.quantile(el, .1)), int(np.quantile(el, .9))] if len(el) else None,
                    "region": region, "slope": None if slope in ("", "None", "nan") else slope})
    return out


def pixel_lonlat(G: F.Grid, g: dict, pix: np.ndarray) -> np.ndarray:
    from pyproj import Transformer
    xs, ys = g["transform"] * (G.cols[pix] + .5, G.rows[pix] + .5)
    lon, lat = Transformer.from_crs(g["crs"], 4326, always_xy=True).transform(xs, ys)
    return np.c_[lon, lat]


def render_maps(species: list[dict], rec: dict, names: list[str]) -> None:
    import io as _io
    import tempfile
    from pathlib import Path
    from PIL import Image
    from mpcr_rag.store import pg_store
    from . import expert_round3 as E
    G = F.Grid()
    g = R.grid()
    index = {n: i for i, n in enumerate(names)}
    folder = F.OUT / "viewer_maps"
    folder.mkdir(exist_ok=True)
    conn = pg_store.connect()
    with conn.cursor() as cur, tempfile.TemporaryDirectory() as tmp:
        for k, sp in enumerate(species, 1):
            out = folder / (sp["name"].replace(" ", "_") + ".jpg")
            if out.exists():
                continue
            s = index[sp["name"]]
            m = rec["sp"] == s
            before = pixel_lonlat(G, g, rec["pix"][m & (rec["year"] <= 2009)])
            after = pixel_lonlat(G, g, rec["pix"][m & (rec["year"] > 2009)])
            cells = [t["cell"] for t in sp["top"]]
            boxes = E.cell_boxes(G, cells)
            draw_top = E.draw_cells(boxes, list(range(1, len(cells) + 1)))
            found = [i for i, t in enumerate(sp["top"]) if t["found"]]

            def overlay(ax, crs, draw_top=draw_top, boxes=boxes, found=found, after=after):
                import geopandas as gpd
                draw_top(ax, crs)
                if found:
                    boxes.iloc[found].to_crs(crs).plot(ax=ax, facecolor="none",
                                                       edgecolor="#39ff14", linewidth=2.6, zorder=22)
                if len(after):
                    a = gpd.GeoSeries(gpd.points_from_xy(after[:, 0], after[:, 1]), crs=4326).to_crs(crs)
                    ax.scatter(a.x, a.y, marker="x", s=16, c="white", linewidths=1.1, zorder=19)

            png = Path(tmp) / "m.png"
            E.system_map(cur, sp["name"], png, lonlat=[tuple(p) for p in before], overlay=overlay)
            im = Image.open(png).convert("RGB")
            im.thumbnail((900, 1100))
            im.save(out, "JPEG", quality=80, optimize=True)
            if k % 20 == 0:
                print(f"  {k}/{len(species)} maps", flush=True)
    conn.close()


def main() -> None:
    import argparse
    import shap
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", action="store_true")
    args = ap.parse_args()
    names, sets, rec = F.build()
    tr, te = sets["train"], sets["test"]
    cols = F._cols(F.SUBSETS["without effort"])
    gb, _, _ = F.train(tr, cols, 3_000_000)
    p = gb.predict_proba(te["X"][:, cols])[:, 1].astype(np.float32)
    G = F.Grid()
    places = cell_places(G)
    expl = shap.TreeExplainer(gb)

    sample = {r["species"] for r in json.loads(
        (F.DATA / "maps_sample.json").read_text(encoding="utf-8"))["sample"]}
    index = {n: i for i, n in enumerate(names)}
    rec_cell = G.pc[rec["pix"]]
    species = []
    for sp in sorted(sample):
        s = index.get(sp)
        if s is None:
            continue
        m = np.flatnonzero(te["s"] == s)
        y = te["y"][m]
        if len(m) < TOP or not y.any():
            continue
        ps, cs, X = p[m], te["c"][m], te["X"][m]
        top = np.argsort(-ps)[:TOP]
        sv = expl.shap_values(X[top][:, cols])
        sv = sv[1] if isinstance(sv, list) else sv
        cells = []
        for j, t in enumerate(top):
            order = np.argsort(-sv[j])[:3]
            why = [F.REASON[F.FEATURES[cols[o]]](float(X[t, cols[o]])) for o in order
                   if not np.isnan(X[t, cols[o]]) and sv[j][o] > 0]
            cells.append({"cell": int(cs[t]), "p": round(float(ps[t]), 3),
                          "found": bool(y[t]), "why": why})
        from sklearn.metrics import roc_auc_score
        hits = F._hits_at(ps, y, TOP)
        r = rec["sp"] == s
        species.append({
            "name": sp, "auc": round(float(roc_auc_score(y, ps)), 3),
            "n_cand": int(len(m)), "n_new": int(y.sum()),
            "hits": round(hits, 2), "expected": round(TOP * float(y.mean()), 2),
            "cand": cs.tolist(), "p": [int(round(1000 * v)) for v in ps.tolist()],
            "new": cs[y].tolist(),
            "before": sorted(set(rec_cell[r & (rec["year"] <= 2009)].tolist())),
            "after": sorted(set(rec_cell[r & (rec["year"] > 2009)].tolist())),
            "top": cells})
    print(f"{len(species)} species")

    # parks: the species the model ranks first in each, and which turned up since 2010
    pcell, pnames = F.park_of_cell()
    park = pcell[te["c"]]
    known = set(zip(rec["sp"][rec["year"] <= 2009].tolist(),
                    pcell[G.pc[rec["pix"][rec["year"] <= 2009]]].tolist()))
    best = collections.defaultdict(lambda: -1.0)
    lab = collections.defaultdict(bool)
    for s, k, v, yy in zip(te["s"][park >= 0], park[park >= 0], p[park >= 0], te["y"][park >= 0]):
        s, k = int(s), int(k)
        if (s, k) in known:
            continue
        best[(k, s)] = max(best[(k, s)], float(v))
        lab[(k, s)] = lab[(k, s)] or bool(yy)
    by_park = collections.defaultdict(list)
    for (k, s), v in best.items():
        by_park[k].append((v, s, lab[(k, s)]))
    parks = []
    for k, lst in by_park.items():
        new = sum(l for _, _, l in lst)
        if new < 5:
            continue
        lst.sort(key=lambda t: -t[0])
        parks.append({"name": pnames[k], "n_cand": len(lst), "n_new": int(new),
                      "known": sum(1 for (s_, k_) in known if k_ == k),
                      "top": [{"species": names[s], "p": round(v, 3), "found": l}
                              for v, s, l in lst[:PARK_TOP]]})
    parks.sort(key=lambda d: d["name"])
    print(f"{len(parks)} parks")

    rows_n, cols_n = int(G.cr.max() // F.CELL) + 1, int(G.cc.max() // F.CELL) + 1
    elev = np.bincount(G.pc, np.nan_to_num(G.elev), G.n) / G.npix
    out = {"grid": {"rows": rows_n, "cols": cols_n,
                    "r": (G.cr // F.CELL).astype(int).tolist(),
                    "c": (G.cc // F.CELL).astype(int).tolist(),
                    "elev": [int(v) for v in elev]},
           "places": places, "species": species, "parks": parks, "top": TOP}
    path = F.OUT / "viewer.json"
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
    if args.maps:
        render_maps(species, rec, names)


if __name__ == "__main__":
    main()
