"""Protected areas as everyday places: elevation, botanical regions, and which species
have occurrence records inside each one.

"¿La puedo ver en el Parque Nacional Braulio Carrillo?" is how people ask "where". The
answer key needs, per park: its elevation band (DEM), the botanical regions it overlaps,
and the species recorded inside it (snapshot occurrences).

Tables (schema ``mpcr``):
  places          one row per terrestrial SINAC protected area
  species_places  (species_key, place_code) -> n_records, n_since_1970

Run:  python -m mpcr_rag.evidence.places
"""
from __future__ import annotations

import io
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

import config as cr_config

from ..store import pg_store

PA_SHP = cr_config.DATA_DIR + "/vectors/areas_protegidas_v2.shp"

# Categories a visitor can go to and where plants grow. Marine management areas and
# wetland-only designations are left out.
TERRESTRIAL = {"PN", "RB", "RNVS", "RVS", "ZP", "RF", "HH", "MN", "RNA"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mpcr.places (
    code        text PRIMARY KEY,
    name        text NOT NULL,
    category    text,             -- PN, RB, RNVS, ZP, ...
    label       text,             -- "Parque Nacional Braulio Carrillo"
    area_km2    double precision,
    elev_p05    double precision,
    elev_p50    double precision,
    elev_p95    double precision,
    regions     text[],
    vertientes  text[]
);
CREATE TABLE IF NOT EXISTS mpcr.species_places (
    species_key  bigint NOT NULL,
    place_code   text NOT NULL REFERENCES mpcr.places ON DELETE CASCADE,
    n_records    int NOT NULL,
    n_since_1970 int NOT NULL,
    PRIMARY KEY (species_key, place_code)
);
"""

_CATEGORY_LABEL = {
    "PN": "Parque Nacional", "RB": "Reserva Biológica",
    "RNVS": "Refugio Nacional de Vida Silvestre", "RVS": "Refugio Nacional de Vida Silvestre",
    "ZP": "Zona Protectora", "RF": "Reserva Forestal", "HH": "Humedal", "MN": "Monumento Nacional",
    "RNA": "Reserva Natural Absoluta",
}


def _places() -> gpd.GeoDataFrame:
    pa = gpd.read_file(PA_SHP).to_crs(4326)
    pa = pa[pa["siglas_cat"].isin(TERRESTRIAL)
            & ~pa["descripcio"].str.contains("marin", case=False, na=False)].copy()
    pa["geometry"] = pa.geometry.make_valid()      # SINAC polygons carry topology errors
    pa = pa.dissolve(by="codigo", aggfunc="first").reset_index()
    pa["label"] = pa["siglas_cat"].map(_CATEGORY_LABEL).fillna(pa["cat_manejo"]) + " " + pa["nombre_asp"]
    return pa


def _elevation(pa: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    with rasterio.open(cr_config.DEM_PATH) as dem:
        nod = dem.nodata
        for _, r in pa.iterrows():
            try:
                arr, _ = mask(dem, [r.geometry], crop=True, all_touched=True)
                v = arr[0].astype(float)
                v = v[(v != nod) & (v > -100)] if nod is not None else v[v > -100]
            except ValueError:
                v = np.array([])
            q = np.percentile(v, [5, 50, 95]) if v.size else [np.nan] * 3
            rows.append({"code": r["codigo"], "elev_p05": q[0], "elev_p50": q[1], "elev_p95": q[2]})
    return pd.DataFrame(rows)


def _regions(pa: gpd.GeoDataFrame) -> pd.DataFrame:
    reg = gpd.read_file(cr_config.REGIONES_BOTANICAS_SHP).to_crs(4326)[["Nombre", "Vertiente", "geometry"]]
    # project for a meaningful overlap area; ignore slivers under 1% of the park
    pa_m, reg_m = pa.to_crs(5367), reg.to_crs(5367)
    inter = gpd.overlay(pa_m[["codigo", "geometry"]], reg_m, how="intersection", keep_geom_type=True)
    inter["frac"] = inter.area / inter["codigo"].map(pa_m.set_index("codigo").area)
    inter = inter[inter["frac"] >= 0.01]
    g = inter.groupby("codigo")
    return pd.DataFrame({
        "regions": g["Nombre"].apply(lambda s: sorted(set(s.dropna()))),
        "vertientes": g["Vertiente"].apply(lambda s: sorted(set(s.dropna()))),
    }).rename_axis("code").reset_index()


def build(verbose: bool = True) -> dict:
    t0 = time.time()
    pa = _places()
    df = pa[["codigo", "nombre_asp", "siglas_cat", "label", "area_km2"]].rename(
        columns={"codigo": "code", "nombre_asp": "name", "siglas_cat": "category"})
    df = df.merge(_elevation(pa), on="code", how="left").merge(_regions(pa), on="code", how="left")
    if verbose:
        print(f"[places] {len(df)} terrestrial protected areas ({time.time() - t0:.0f}s)", flush=True)

    conn = pg_store.connect()
    cur = conn.cursor()
    cur.execute(_SCHEMA)
    cur.execute("TRUNCATE mpcr.places CASCADE")
    for r in df.itertuples():
        cur.execute("INSERT INTO mpcr.places VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (r.code, r.name, r.category, r.label, float(r.area_km2),
                     None if pd.isna(r.elev_p05) else float(r.elev_p05),
                     None if pd.isna(r.elev_p50) else float(r.elev_p50),
                     None if pd.isna(r.elev_p95) else float(r.elev_p95),
                     r.regions if isinstance(r.regions, list) else [],
                     r.vertientes if isinstance(r.vertientes, list) else []))
    conn.commit()

    occ = pd.read_sql("SELECT gbif_id, species_key, lon, lat, year FROM mpcr.occurrences "
                      "WHERE species_key IS NOT NULL", conn)
    pts = gpd.GeoDataFrame(occ, geometry=gpd.points_from_xy(occ["lon"], occ["lat"]), crs=4326)
    joined = gpd.sjoin(pts, pa[["codigo", "geometry"]], how="inner", predicate="within")
    agg = (joined.assign(recent=joined["year"].fillna(0) >= 1970)
                 .groupby(["species_key", "codigo"])
                 .agg(n_records=("gbif_id", "size"), n_since_1970=("recent", "sum"))
                 .reset_index())
    buf = io.StringIO()
    agg.to_csv(buf, sep="\t", header=False, index=False)
    buf.seek(0)
    cur.copy_expert("COPY mpcr.species_places (species_key, place_code, n_records, n_since_1970) "
                    "FROM STDIN WITH (FORMAT text)", buf)
    conn.commit()
    conn.close()
    out = {"places": len(df), "records_in_places": int(len(joined)),
           "species_place_pairs": int(len(agg)), "seconds": round(time.time() - t0)}
    if verbose:
        print(f"[places] {out}", flush=True)
    return out


def where_to_see(species: str, *, conn=None, limit: int = 8) -> dict:
    """Where a visitor could look for a species: protected areas with occurrence
    records, and the regions the Manual states versus those the records show.

    Two different kinds of evidence, never merged:
      confirmed_places  parks with cleaned GBIF records inside them (observed)
      manual_regions    regions the Manual text states (expert-stated range)
      record_regions    regions where records fall (may extend the Manual's)
    A park without records is not evidence of absence: collection effort is uneven.
    """
    own = conn is None
    conn = conn or pg_store.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT f.regions, (f.ficha->>'elev_min')::int,
                                  (f.ficha->>'elev_max')::int, e.taxon_key, e.regions,
                                  e.n_records, e.n_cells
                           FROM mpcr.fichas f
                           LEFT JOIN mpcr.species_evidence e ON e.species = f.species
                           WHERE f.species = %s""", (species,))
            row = cur.fetchone()
            if not row:
                return {"species": species, "found": False}
            manual_regions, elev_min, elev_max, key, rec_regions, n_rec, n_cells = row
            cur.execute("""SELECT p.label, p.category, s.n_records, s.n_since_1970,
                                  round(p.elev_p05) , round(p.elev_p95), p.regions
                           FROM mpcr.species_places s JOIN mpcr.places p ON p.code = s.place_code
                           WHERE s.species_key = %s
                           ORDER BY s.n_records DESC LIMIT %s""", (key, limit))
            places = [{"place": r[0], "category": r[1], "n_records": r[2],
                       "n_since_1970": r[3], "elev_p05": r[4], "elev_p95": r[5],
                       "regions": r[6]} for r in cur.fetchall()]
    finally:
        if own:
            conn.close()
    return {"species": species, "found": True, "elev_min": elev_min, "elev_max": elev_max,
            "n_records": n_rec or 0, "n_cells": n_cells or 0,
            "manual_regions": manual_regions or [], "record_regions": sorted(rec_regions or []),
            "confirmed_places": places}


if __name__ == "__main__":
    build()
