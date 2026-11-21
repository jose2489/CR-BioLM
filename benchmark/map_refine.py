"""Two refinements of the system's map, measured before adopting either.

The expert map review (revision_mapas_Armando.xlsx) judged the system's map too broad
for species of the seasonal Pacific (Hyptis brachiata, Eriosema crinitum). Two causes
were traced, each a piece of the Manual's statement the map ignores:

  slope per region   "ambas verts. Cord. de Guanacaste, vert. Pac. Cord. de Talamanca"
                     says Pacific side only for Talamanca. The map applied the
                     species' slope set (both) to every region. The region layer is
                     already split by slope, so a map can be a set of (region, slope)
                     pairs; the slope of each comes from the parser's slope state
                     machine (geo_parser), which already tracks it per feature.
  forest type        "Bosque seco, húmedo" says nothing grows in pluvial forest. The
                     Holdridge life zones give each pixel a moisture class, and a
                     species with stated moisture classes is drawn only where one of
                     them holds (a transition zone counts as both of its classes).

A third fix rides along: four catalogue region names ("Valle Central", "Valle Central
(Occidental)", "Fila Costeña (Norte)", "Fila Costeña (Sur)") are the gazetteer's
canonical names, not the layer's, and the map comparison had been matching them against
the layer directly, so those regions were never drawn. They are mapped through the
gazetteer here, in every variant, so the refinements are measured on their own.

Every variant is scored the same way: capture is whether a record's PIXEL is inside the
map (one rule for all variants), area is the share of land pixels, and the system's
variants are scored held out, regions and pairs from one random half of the records and
capture on the other half, as in mapcompare.

Run:  python -m benchmark.map_refine            (300-species sample, prints a table)
      python -m benchmark.map_refine --species "Hyptis brachiata,Eriosema crinitum"
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import random
import statistics
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import mapcompare as M

DATA = Path(__file__).resolve().parent / "data"
SEED = 20261012
MIN = 5                       # records needed to add a region (or a region-slope pair)
MOIST = ("seco", "húmedo", "muy húmedo", "pluvial")
LIFE_ZONES = Path(__file__).resolve().parent.parent / "data_raw" / "ecoregions" / "zonas_vida_cr.shp"


# ----------------------------------------------------------------------- the grid

def _moisture_classes(nombre: str) -> set[str]:
    n = nombre.upper()
    out = set()
    for part in n.split("TRANSICION A"):
        if "PARAMO" in part or "PLUVIAL" in part:
            out.add("pluvial")
        elif "MUY HUMEDO" in part or "PERHUMEDO" in part:
            out.add("muy húmedo")
        elif "HUMEDO" in part:
            out.add("húmedo")
        elif "SECO" in part:
            out.add("seco")
    return out


@lru_cache(maxsize=1)
def grid():
    """Per land pixel: region, slope, elevation (from mapcompare) and a moisture
    bitmask; and the transform to find a record's pixel."""
    import geopandas as gpd
    import rasterio
    from rasterio import features
    from utils.distribution_map.geodata import PATHS
    names, slopes, elev = M._grid()
    with rasterio.open(PATHS["dem"]) as src:
        shape, transform, crs = src.shape, src.transform, src.crs
    land = M._GEO["land"]
    zones = gpd.read_file(LIFE_ZONES).to_crs(crs)
    bits = np.zeros(shape, dtype=np.uint8)
    for i, cls in enumerate(MOIST):
        sel = zones[zones["nombre"].map(lambda n: cls in _moisture_classes(n))]
        if len(sel):
            r = features.rasterize(((g, 1) for g in sel.geometry), out_shape=shape,
                                   transform=transform, fill=0)
            bits |= (r.astype(np.uint8) << i)
    flat_index = -np.ones(shape, dtype=np.int64)
    flat_index[land] = np.arange(land.sum())
    return {"names": names, "slopes": slopes, "elev": elev, "moist": bits[land],
            "transform": transform, "crs": crs, "index": flat_index}


def record_pixels(lonlat: list[tuple]) -> np.ndarray:
    """Land-pixel index of each record (-1 when it falls outside land)."""
    from pyproj import Transformer
    from rasterio.transform import rowcol
    g = grid()
    if not lonlat:
        return np.array([], dtype=np.int64)
    tr = Transformer.from_crs(4326, g["crs"], always_xy=True)
    xs, ys = tr.transform([p[0] for p in lonlat], [p[1] for p in lonlat])
    rows, cols = rowcol(g["transform"], xs, ys)
    rows, cols = np.asarray(rows), np.asarray(cols)
    ok = (rows >= 0) & (rows < g["index"].shape[0]) & (cols >= 0) & (cols < g["index"].shape[1])
    out = -np.ones(len(rows), dtype=np.int64)
    out[ok] = g["index"][rows[ok], cols[ok]]
    return out


# ------------------------------------------------------------------ the ingredients

@lru_cache(maxsize=None)
def _layer(canonical: str) -> tuple[str, ...]:
    """Gazetteer canonical region name -> region layer name(s)."""
    from utils.distribution_map.gazetteer import lookup
    hits = [e for v in lookup(canonical).values() for e in v
            if e.get("target_shapefile") == "regiones_botanicas"]
    exact = [e["target_value"] for e in hits if e["canonical_name"] == canonical]
    return tuple(exact or [e["target_value"] for e in hits] or [canonical])


def layer_regions(names) -> set[str]:
    return {x for n in names or [] for x in _layer(n)}


def _pairs_from_paragraph(paragraph: str, default_slopes: list[str]) -> set[tuple]:
    from utils.distribution_map.gazetteer import lookup
    from utils.distribution_map.geo_parser import parse_distribution_block
    pairs = set()
    both = ["Caribe", "Pacífico"]
    for occ in parse_distribution_block(paragraph or ""):
        v = occ.get("vertiente")
        slopes = both if v == "ambas" else [v] if v in both else (default_slopes or both)
        hits = [e["target_value"] for lv in lookup(occ.get("feature_name") or "").values()
                for e in lv if e.get("target_shapefile") == "regiones_botanicas"]
        for reg in hits:
            for s in slopes:
                pairs.add((reg, s))
    return pairs


def ingredients(f) -> dict:
    """The Manual's statement in all its parts: regions (layer names), slopes, the
    region-slope pairs the paragraph states, moisture classes and elevation."""
    base = M.from_ficha(f)
    regions = layer_regions(f.regions)
    slopes = list(f.vertientes or [])
    pairs = _pairs_from_paragraph(f.distribution_paragraph, slopes)
    # Regions the catalogue holds but the paragraph's parse did not pair (directional
    # expansions, regions named outside the slope clauses) keep the species' slopes.
    paired = {r for r, _ in pairs}
    for r in regions - paired:
        for s in (slopes or ["Caribe", "Pacífico"]):
            pairs.add((r, s))
    moist = {m for m in (f.forest_types or []) if m in MOIST}
    return {"regions": sorted(regions), "slopes": slopes, "pairs": sorted(pairs),
            "moist": sorted(moist), "elev": base["elev"]}


def widen(ing: dict, recs: list[tuple]) -> dict:
    """The system's widening: regions (and region-slope pairs) holding >= MIN records."""
    reg = collections.Counter(r for r, _s in recs if r)
    pair = collections.Counter((r, s) for r, s in recs if r and s)
    return {**ing,
            "regions": sorted(set(ing["regions"]) | {r for r, n in reg.items() if n >= MIN}),
            "pairs": sorted(set(map(tuple, ing["pairs"])) | {p for p, n in pair.items() if n >= MIN})}


VARIANTS = {               # name -> (use pairs, use moisture)
    "current": (False, False),
    "slope per region": (True, False),
    "forest type": (False, True),
    "both": (True, True),
}


def mask(ing: dict, use_pairs: bool, use_moist: bool) -> np.ndarray:
    g = grid()
    if use_pairs:
        keys = np.char.add(np.char.add(g["names"].astype(str), "|"), g["slopes"].astype(str))
        m = np.isin(keys, [f"{r}|{s}" for r, s in ing["pairs"]])
    else:
        m = np.isin(g["names"], ing["regions"])
        if ing["slopes"]:
            m &= np.isin(g["slopes"], ing["slopes"])
    if ing["elev"]:
        m &= (g["elev"] >= ing["elev"][0]) & (g["elev"] <= ing["elev"][1])
    if use_moist and ing["moist"]:
        want = 0
        for i, c in enumerate(MOIST):
            if c in ing["moist"]:
                want |= 1 << i
        m &= (g["moist"] & want) != 0
    return m


# ------------------------------------------------------------------------ scoring

def _records(cur, species: str):
    cur.execute("""SELECT o.lon, o.lat, o.region, o.vertiente FROM mpcr.occurrences o
                   JOIN mpcr.species_evidence e ON e.taxon_key = o.species_key
                   WHERE e.species = %s""", (species,))
    rows = cur.fetchall()
    return ([(lon, lat) for lon, lat, _r, _v in rows],
            [(r, None if v is None else ("Caribe" if "arib" in v else "Pacífico"))
             for _lon, _lat, r, v in rows])


def score_species(cur, species: str) -> dict:
    from mpcr_rag.schema import Ficha
    cur.execute("SELECT ficha FROM mpcr.fichas WHERE species = %s", (species,))
    f = Ficha(**cur.fetchone()[0])
    lonlat, attrs = _records(cur, species)
    pix = record_pixels(lonlat)
    keep = pix >= 0
    pix, attrs = pix[keep], [a for a, k in zip(attrs, keep) if k]
    ing = ingredients(f)
    rng = random.Random(f"{SEED}-{species}")
    order = list(range(len(pix)))
    rng.shuffle(order)
    half = len(order) // 2
    folds = [(order[:half], order[half:]), (order[half:], order[:half])]
    out = {"species": species, "n": len(pix), "moist": ing["moist"]}
    for name, (up, um) in VARIANTS.items():
        m_manual = mask(ing, up, um)
        cap, area = 0, []
        for train, test in folds:
            mk = mask(widen(ing, [attrs[i] for i in train]), up, um)
            cap += int(mk[pix[test]].sum())
            area.append(float(mk.mean()))
        out[name] = {"held_capture": cap / max(1, len(pix)), "area": float(np.mean(area)),
                     "manual_capture": float(m_manual[pix].mean()) if len(pix) else 0.0,
                     "manual_area": float(m_manual.mean())}
    return out


def main() -> None:
    from mpcr_rag.store import pg_store
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="")
    args = ap.parse_args()
    if args.species:
        species = [s.strip() for s in args.species.split(",") if s.strip()]
    else:
        species = [r["species"] for r in json.loads(
            (DATA / "maps_sample.json").read_text(encoding="utf-8"))["sample"]]
    conn = pg_store.connect()
    rows = []
    with conn.cursor() as cur:
        for i, sp in enumerate(species, 1):
            rows.append(score_species(cur, sp))
            if i % 50 == 0:
                print(f"  {i}/{len(species)}", flush=True)
    conn.close()
    out = DATA / ("map_refine_scores.jsonl" if not args.species else "map_refine_check.jsonl")
    with io.open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{len(rows)} species; system map scored held out (median per species)")
    print(f"{'variant':18s} {'capture':>8s} {'area':>6s} {'lift':>5s} {'>=half':>7s}"
          f"   Manual alone: capture  area")
    for name in VARIANTS:
        cap = [r[name]["held_capture"] for r in rows]
        area = [r[name]["area"] for r in rows]
        lift = [c / a for c, a in zip(cap, area) if a > 0]
        print(f"{name:18s} {100 * statistics.median(cap):7.0f}% {100 * statistics.median(area):5.1f}%"
              f" {statistics.median(lift):5.1f} {sum(c >= .5 for c in cap):4d}/{len(rows)}"
              f"   {100 * statistics.median(r[name]['manual_capture'] for r in rows):11.0f}%"
              f" {100 * statistics.median(r[name]['manual_area'] for r in rows):5.1f}%")
    if args.species:
        for r in rows:
            print(f"\n{r['species']} ({r['n']} records; moisture {r['moist']})")
            for name in VARIANTS:
                v = r[name]
                print(f"   {name:18s} capture {100 * v['held_capture']:3.0f}%  area "
                      f"{100 * v['area']:4.1f}%")


if __name__ == "__main__":
    main()
