"""A climate profile per species, from where it has been collected, and a test of
whether it sharpens the system's map.

The expert map review found two failures the Manual's regions cannot fix: species of
the seasonal Pacific drawn over wet forest (Hyptis brachiata, Eriosema crinitum) and
narrow species drawn over whole regions (Racinaea spiculosa, Kearnemalvastrum
subtriflorum). Both are differences of climate inside a region. WorldClim 2.1 at 30
arc-seconds is on the same 1 km grid as the DEM, so each land pixel has a climate.

The profile is the range of four variables over the species' records:

  bio1    annual mean temperature (degC)
  bio12   annual precipitation (mm)
  bio15   precipitation seasonality (coefficient of variation, %)
  bio17   precipitation of the driest quarter (mm)

bio15 and bio17 are the two that say "long dry season", which is what the seasonal
Pacific species share and the regions do not. Each variable's range is the q-th to
(1-q)-th quantile of the records, for a few values of q, so the trade-off between area
and capture is shown rather than chosen silently. A species with fewer than
MIN_PROFILE records gets no climate filter: its envelope would be a handful of points.

The profile is a summary of the records, not a statement of the Manual, and an answer
built on it has to say so.

Scoring is map_refine's: the map and the envelope come from one random half of the
records and capture is counted on the other half, so the envelope never sees the
records it is tested on. The base map is the adopted one (slope per region).

Run:  python -m benchmark.climate_profile                       (300-species sample)
      python -m benchmark.climate_profile --species "Hyptis brachiata,Eriosema crinitum"
      python -m benchmark.climate_profile --profile "Hyptis brachiata"
"""
from __future__ import annotations

import argparse
import io
import json
import random
import statistics
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import map_refine as R

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
CLIMATE = ROOT / "data_raw" / "climate_rasters" / "clipped"
VARS = {"bio1": "temperatura media anual (°C)",
        "bio12": "precipitación anual (mm)",
        "bio15": "estacionalidad de la lluvia (CV, %)",
        "bio17": "lluvia del trimestre más seco (mm)"}
QUANTILES = (0.0, 0.025, 0.05)          # envelope = [q, 1-q] of the training records
MIN_PROFILE = 10                        # training records needed to apply an envelope


# Two climatologies, both 30 arc-seconds. WorldClim interpolates station data and puts
# Holdridge's pluvial zones (4,000-8,000 mm) at a median 3,384 mm, 96% of them below the
# zone's floor; CHELSA models orographic rain and does not flatten the wet mountains as
# much. Results that rest on rainfall are reported against both.
SOURCES = {"worldclim": "cr_30s_{b}.tif", "chelsa": "chelsa_{v}.tif"}
REFERENCE = "worldclim"


def climate(source: str | None = None) -> dict[str, np.ndarray]:
    return _climate(source or REFERENCE)


@lru_cache(maxsize=None)
def _climate(source: str) -> dict[str, np.ndarray]:
    """Each variable on the DEM grid, as a vector over land pixels (NaN off-raster)."""
    import rasterio
    from rasterio.warp import Resampling, reproject
    from utils.distribution_map.geodata import PATHS
    from . import mapcompare as M
    with rasterio.open(PATHS["dem"]) as dem:
        shape, transform, crs = dem.shape, dem.transform, dem.crs
    M._grid()                     # fills M._GEO, the land mask included
    land = M._GEO["land"]
    out = {}
    for v in VARS:
        name = SOURCES[source].format(b=v.replace("bio", "bio_"), v=v)
        with rasterio.open(CLIMATE / name) as src:
            dst = np.full(shape, np.nan, dtype=np.float32)
            reproject(source=rasterio.band(src, 1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                      dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                      resampling=Resampling.nearest)
        out[v] = dst[land]
    return out


def envelope(pix: np.ndarray, q: float) -> dict[str, tuple[float, float]] | None:
    if len(pix) < MIN_PROFILE:
        return None
    c = climate()
    env = {}
    for v in VARS:
        x = c[v][pix]
        x = x[~np.isnan(x)]
        if len(x) < MIN_PROFILE:
            return None
        env[v] = (float(np.quantile(x, q)), float(np.quantile(x, 1 - q)))
    return env


def climate_mask(env: dict | None) -> np.ndarray:
    c = climate()
    n = len(c["bio1"])
    if env is None:
        return np.ones(n, dtype=bool)
    m = np.ones(n, dtype=bool)
    for v, (lo, hi) in env.items():
        x = c[v]
        m &= np.isnan(x) | ((x >= lo) & (x <= hi))   # off-raster coast pixels pass
    return m


# ------------------------------------------------------------------------ scoring

def score_species(cur, species: str) -> dict:
    from mpcr_rag.schema import Ficha
    cur.execute("SELECT ficha FROM mpcr.fichas WHERE species = %s", (species,))
    f = Ficha(**cur.fetchone()[0])
    lonlat, attrs = R._records(cur, species)
    pix = R.record_pixels(lonlat)
    keep = pix >= 0
    pix, attrs = pix[keep], [a for a, k in zip(attrs, keep) if k]
    ing = R.ingredients(f)
    rng = random.Random(f"{R.SEED}-{species}")       # the same halves as map_refine
    order = list(range(len(pix)))
    rng.shuffle(order)
    half = len(order) // 2
    folds = [(order[:half], order[half:]), (order[half:], order[:half])]
    variants = {"slope per region": None, **{f"climate q={q}": q for q in QUANTILES}}
    out = {"species": species, "n": len(pix)}
    for name, q in variants.items():
        cap, area, applied = 0, [], 0
        for train, test in folds:
            mk = R.mask(R.widen(ing, [attrs[i] for i in train]), True, False)
            if q is not None:
                env = envelope(pix[train], q)
                applied += env is not None
                mk = mk & climate_mask(env)
            cap += int(mk[pix[test]].sum())
            area.append(float(mk.mean()))
        out[name] = {"held_capture": cap / max(1, len(pix)), "area": float(np.mean(area)),
                     "applied": applied}
    return out


# ------------------------------------------------------------------------ the profile

def in_band(cur, species: str, pix: np.ndarray) -> np.ndarray:
    """Records inside the Manual's elevation range (its atypical values included) with
    the checker's 200 m tolerance. A record far below a páramo species' floor is a
    georeferencing error far more often than a find, and climate follows elevation
    closely enough here that a few such records move the warm end by 10 degrees
    (Hypericum irazuense: 25 of 327 records below 1,500 m, against 2,200-3,800 m)."""
    from mpcr_rag.schema import Ficha
    from .evidence_pack import _eff
    from .questions import CULTIVATION_MARGIN_M as TOL
    cur.execute("SELECT ficha FROM mpcr.fichas WHERE species = %s", (species,))
    row = cur.fetchone()
    eff = _eff(Ficha(**row[0])) if row else None
    if eff is None or not len(pix):
        return np.ones(len(pix), dtype=bool)
    e = R.grid()["elev"][pix]
    return (e >= eff[0] - TOL) & (e <= eff[1] + TOL)


def profile(cur, species: str, subset: list[int] | None = None) -> dict:
    """The profile from the species' records (or a subset of them, by position, which
    is how the comparison builds it from one half), and the life zones they fall in.
    Only records inside the Manual's elevation range count (in_band)."""
    import geopandas as gpd
    from shapely.geometry import Point
    lonlat, _ = R._records(cur, species)
    if subset is not None:
        lonlat = [lonlat[i] for i in subset]
    pix = R.record_pixels(lonlat)
    lonlat = [p for p, k in zip(lonlat, pix >= 0) if k]
    pix = pix[pix >= 0]
    n_all = len(pix)
    keep = in_band(cur, species, pix)
    lonlat = [p for p, k in zip(lonlat, keep) if k]
    pix = pix[keep]
    c = climate()
    out = {"species": species, "n": len(pix), "n_out_of_band": n_all - len(pix), "vars": {}}
    for v, label in VARS.items():
        x = c[v][pix]
        x = x[~np.isnan(x)]
        if len(x):
            out["vars"][v] = {"label": label, "p05": float(np.quantile(x, .05)),
                              "median": float(np.median(x)), "p95": float(np.quantile(x, .95))}
    zones = gpd.read_file(R.LIFE_ZONES).to_crs(4326)
    pts = gpd.GeoDataFrame(geometry=[Point(p) for p in lonlat], crs=4326)
    hit = gpd.sjoin(pts, zones[["nombre", "geometry"]], predicate="within")
    counts = hit["nombre"].value_counts()
    out["life_zones"] = {k.capitalize(): int(n) for k, n in counts.items()}
    return out


_ACCENTS = {"humedo": "húmedo", "paramo": "páramo", "transicion": "transición"}


def _accents(name: str) -> str:
    """The life-zone layer writes its names in capitals without accents."""
    return " ".join(_ACCENTS.get(w, w) for w in name.split())


def profile_text(p: dict) -> str:
    """The profile in Spanish, worded as a summary of records."""
    if p["n"] < MIN_PROFILE:
        return (f"La especie tiene {p['n']} registros con coordenadas en Costa Rica, "
                f"muy pocos para describir el clima donde crece.")
    v = p["vars"]
    def num(x: float) -> str:
        return f"{x:,.0f}".replace(",", ".")      # Spanish thousands separator
    parts = [f"{v[k]['label']}: {num(v[k]['p05'])}–{num(v[k]['p95'])}"
             for k in VARS if k in v]
    total = sum(p["life_zones"].values())
    zones = [f"{_accents(z.lower())} ({100 * n / total:.0f}%)"
             for z, n in list(p["life_zones"].items())[:3]] if total else []
    out_band = p.get("n_out_of_band", 0)
    excl = (f", sin contar {out_band} fuera de su rango altitudinal según el Manual"
            if out_band else "")
    txt = (f"Según sus {p['n']} registros en Costa Rica{excl} (WorldClim 2.1, 1 km; 5.º "
           f"a 95.º percentil): " + "; ".join(parts) + ".")
    if zones:
        txt += " Zonas de vida más frecuentes: " + ", ".join(zones) + "."
    return txt


def main() -> None:
    from mpcr_rag.store import pg_store
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="")
    ap.add_argument("--profile", default="", help="print the profile of these species")
    args = ap.parse_args()
    conn = pg_store.connect()
    with conn.cursor() as cur:
        if args.profile:
            for sp in [s.strip() for s in args.profile.split(",") if s.strip()]:
                print(f"\n{sp}\n  {profile_text(profile(cur, sp))}")
            conn.close()
            return
        if args.species:
            species = [s.strip() for s in args.species.split(",") if s.strip()]
        else:
            species = [r["species"] for r in json.loads(
                (DATA / "maps_sample.json").read_text(encoding="utf-8"))["sample"]]
        rows = []
        for i, sp in enumerate(species, 1):
            rows.append(score_species(cur, sp))
            if i % 50 == 0:
                print(f"  {i}/{len(species)}", flush=True)
    conn.close()
    out = DATA / ("climate_profile_scores.jsonl" if not args.species
                  else "climate_profile_check.jsonl")
    with io.open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    names = [k for k in rows[0] if k not in ("species", "n")]
    print(f"\n{len(rows)} species; held out (median per species)")
    print(f"{'variant':18s} {'capture':>8s} {'area':>6s} {'lift':>5s} {'>=half':>7s} {'filtered':>9s}")
    for name in names:
        cap = [r[name]["held_capture"] for r in rows]
        area = [r[name]["area"] for r in rows]
        lift = [c / a for c, a in zip(cap, area) if a > 0]
        print(f"{name:18s} {100 * statistics.median(cap):7.0f}% {100 * statistics.median(area):5.1f}%"
              f" {statistics.median(lift):5.1f} {sum(c >= .5 for c in cap):4d}/{len(rows)}"
              f" {sum(r[name]['applied'] == 2 for r in rows):6d}")
    if args.species:
        for r in rows:
            print(f"\n{r['species']} ({r['n']} records)")
            for name in names:
                v = r[name]
                print(f"   {name:18s} capture {100 * v['held_capture']:3.0f}%  area "
                      f"{100 * v['area']:4.1f}%")


if __name__ == "__main__":
    main()
