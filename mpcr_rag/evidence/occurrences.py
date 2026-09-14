"""Load the frozen GBIF snapshot into Postgres as cleaned, enriched occurrence records.

Tables (schema ``mpcr``):
  occurrences       one row per snapshot record that passes the map pipeline's
                    coordinate cleaning, with DEM elevation and botanical region
  taxa              Manual species -> GBIF accepted key (from evidence/taxa.py)
  species_evidence  per-catalog-species aggregates: record counts, unique cells,
                    elevation quantiles, months, record types, regions, years

Cleaning is identical to ``gbif_map._fetch_clean`` (the path the experts validated):
drop (0,0), drop uncertainty > 10 km, fix lat/lon swaps inside the Costa Rica box,
drop anything still outside it. The snapshot was downloaded with
hasGeospatialIssue=false, which the live path requests per query.

Run:  python -m mpcr_rag.evidence.occurrences              # snapshot + taxa + aggregates
      python -m mpcr_rag.evidence.occurrences --taxa-only  # after re-resolving taxa
"""
from __future__ import annotations

import io
import time
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

import config as cr_config

from .. import config
from ..query.gbif_map import _CR_BOX, _MAX_UNCERTAINTY_M, _SNAPSHOT_DIR, _SNAPSHOT_KEY
from ..store import pg_store
from . import taxa

COLS = ["gbifID", "speciesKey", "species", "decimalLatitude", "decimalLongitude",
        "coordinateUncertaintyInMeters", "elevation", "year", "month", "basisOfRecord",
        "establishmentMeans", "stateProvince", "locality"]

# ~30 arc-second cells (the WorldClim / DEM resolution) for counting distinct sites:
# many records of one species are duplicates of the same collecting locality.
CELL_DEG = 1 / 120

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mpcr.occurrences (
    gbif_id        bigint PRIMARY KEY,
    species_key    bigint,
    species        text,
    lon            double precision NOT NULL,
    lat            double precision NOT NULL,
    uncertainty_m  double precision,
    elev_reported  double precision,
    elev_dem       double precision,
    year           int,
    month          int,
    basis          text,
    established    text,
    province       text,
    locality       text,
    region         text,
    vertiente      text,
    cell           text              -- 30 arc-second cell id, for distinct-site counts
);
CREATE INDEX IF NOT EXISTS occurrences_species_key ON mpcr.occurrences (species_key);

CREATE TABLE IF NOT EXISTS mpcr.taxa (
    species        text PRIMARY KEY, -- Manual name
    taxon_key      bigint,           -- GBIF accepted key (NULL = no exact match)
    accepted_name  text,
    synonym        boolean,
    match          text
);
CREATE INDEX IF NOT EXISTS taxa_key ON mpcr.taxa (taxon_key);
"""


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    lat = pd.to_numeric(df["decimalLatitude"], errors="coerce")
    lon = pd.to_numeric(df["decimalLongitude"], errors="coerce")
    unc = pd.to_numeric(df["coordinateUncertaintyInMeters"], errors="coerce")
    lo_x, lo_y, hi_x, hi_y = _CR_BOX

    keep = lat.notna() & lon.notna() & ~((lat.abs() < 0.01) & (lon.abs() < 0.01))
    keep &= ~(unc > _MAX_UNCERTAINTY_M)
    in_box = lon.between(lo_x, hi_x) & lat.between(lo_y, hi_y)
    swapped = ~in_box & lat.between(lo_x, hi_x) & lon.between(lo_y, hi_y)
    lon2, lat2 = lon.where(~swapped, lat), lat.where(~swapped, lon)
    keep &= in_box | swapped

    out = pd.DataFrame({
        "gbif_id": pd.to_numeric(df["gbifID"], errors="coerce"),
        "species_key": pd.to_numeric(df["speciesKey"], errors="coerce"),
        "species": df["species"],
        "lon": lon2, "lat": lat2, "uncertainty_m": unc,
        "elev_reported": pd.to_numeric(df["elevation"], errors="coerce"),
        "year": pd.to_numeric(df["year"], errors="coerce"),
        "month": pd.to_numeric(df["month"], errors="coerce"),
        "basis": df["basisOfRecord"], "established": df["establishmentMeans"],
        "province": df["stateProvince"], "locality": df["locality"],
    })[keep & pd.to_numeric(df["gbifID"], errors="coerce").notna()]
    return out


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    with rasterio.open(cr_config.DEM_PATH) as dem:
        vals = np.array([v[0] for v in dem.sample(zip(df["lon"], df["lat"]))], dtype=float)
        nod = dem.nodata
    if nod is not None:
        vals[vals == nod] = np.nan
    vals[vals < -100] = np.nan
    df["elev_dem"] = vals

    regions = gpd.read_file(cr_config.REGIONES_BOTANICAS_SHP).to_crs(4326)[["Nombre", "Vertiente", "geometry"]]
    pts = gpd.GeoDataFrame(df[["gbif_id"]], geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
    joined = gpd.sjoin(pts, regions, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    df["region"] = joined["Nombre"].values
    df["vertiente"] = joined["Vertiente"].values
    df["cell"] = ((df["lon"] // CELL_DEG).astype(int).astype(str) + ":"
                  + (df["lat"] // CELL_DEG).astype(int).astype(str))
    return df


def _copy(cur, df: pd.DataFrame, table: str, cols: list[str]) -> None:
    buf = io.StringIO()
    df[cols].to_csv(buf, sep="\t", header=False, index=False, na_rep="\\N")
    buf.seek(0)
    cur.copy_expert(f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT text)", buf)


def load_snapshot(chunksize: int = 250_000, verbose: bool = True) -> dict:
    t0 = time.time()
    zip_path = _SNAPSHOT_DIR / f"{_SNAPSHOT_KEY}.zip"
    conn = pg_store.connect()
    cur = conn.cursor()
    cur.execute(_SCHEMA)
    cur.execute("TRUNCATE mpcr.occurrences")
    conn.commit()

    raw = kept = 0
    cols = ["gbif_id", "species_key", "species", "lon", "lat", "uncertainty_m", "elev_reported",
            "elev_dem", "year", "month", "basis", "established", "province", "locality",
            "region", "vertiente", "cell"]
    with zipfile.ZipFile(zip_path) as zf, zf.open(zf.namelist()[0]) as fh:
        for chunk in pd.read_csv(fh, sep="\t", usecols=COLS, dtype=str, chunksize=chunksize,
                                 on_bad_lines="skip", quoting=3):
            raw += len(chunk)
            df = _enrich(_clean(chunk).drop_duplicates("gbif_id"))
            for c in ("species_key", "year", "month", "gbif_id"):
                df[c] = df[c].astype("Int64")
            for c in ("locality", "province", "species", "basis", "established"):
                df[c] = df[c].str.replace(r"[\t\n\r\\]", " ", regex=True)
            _copy(cur, df, "mpcr.occurrences", cols)
            conn.commit()
            kept += len(df)
            if verbose:
                print(f"[occ] {raw:,} read, {kept:,} kept ({time.time() - t0:.0f}s)", flush=True)

    conn.close()
    return {"read": raw, "kept": kept, "seconds": round(time.time() - t0)}


def refresh_taxa_and_evidence() -> dict:
    """Load the taxon cache into Postgres and rebuild species_evidence. Separate from
    the snapshot load so a taxon re-resolution does not reload 1M records."""
    conn = pg_store.connect()
    cur = conn.cursor()
    cur.execute(_SCHEMA)
    tx = pd.DataFrame([{"species": sp, **v} for sp, v in taxa.load().items()])
    cur.execute("TRUNCATE mpcr.taxa")
    if not tx.empty:
        tx["taxon_key"] = tx["taxon_key"].astype("Int64")
        if "match" not in tx:
            tx["match"] = None
        _copy(cur, tx, "mpcr.taxa", ["species", "taxon_key", "accepted_name", "synonym", "match"])
    conn.commit()
    build_species_evidence(conn)
    conn.close()
    return {"taxa": len(tx)}


def build_species_evidence(conn=None) -> None:
    """Per-catalog-species occurrence aggregates (materialized for fast joins)."""
    own = conn is None
    conn = conn or pg_store.connect()
    with conn.cursor() as cur:
        cur.execute("""
        DROP TABLE IF EXISTS mpcr.species_evidence;
        CREATE TABLE mpcr.species_evidence AS
        SELECT t.species,
               t.taxon_key,
               count(o.gbif_id)                                         AS n_records,
               count(DISTINCT o.cell)                                   AS n_cells,
               count(*) FILTER (WHERE o.basis = 'PRESERVED_SPECIMEN')   AS n_specimens,
               count(*) FILTER (WHERE o.basis = 'HUMAN_OBSERVATION')    AS n_observations,
               count(*) FILTER (WHERE o.year >= 2000)                   AS n_since_2000,
               min(o.year) AS first_year, max(o.year) AS last_year,
               percentile_cont(0.05) WITHIN GROUP (ORDER BY o.elev_dem) AS elev_dem_p05,
               percentile_cont(0.50) WITHIN GROUP (ORDER BY o.elev_dem) AS elev_dem_p50,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY o.elev_dem) AS elev_dem_p95,
               array_remove(array_agg(DISTINCT o.region), NULL)         AS regions,
               array_remove(array_agg(DISTINCT o.vertiente), NULL)      AS vertientes,
               array_remove(array_agg(DISTINCT o.month), NULL)          AS record_months
        FROM mpcr.taxa t
        LEFT JOIN mpcr.occurrences o ON o.species_key = t.taxon_key
        GROUP BY t.species, t.taxon_key;
        ALTER TABLE mpcr.species_evidence ADD PRIMARY KEY (species);
        """)
    conn.commit()
    if own:
        conn.close()


if __name__ == "__main__":
    import sys
    if "--taxa-only" not in sys.argv:
        print(load_snapshot())
    print(refresh_taxa_and_evidence())
