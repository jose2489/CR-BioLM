"""
renderer.py — Distribution map renderer (Phase 3).

Takes a DistributionFicha and produces a PNG using a narrowing strategy:
  country → vertiente → region → park → canton → district

The narrowest non-empty scope drives:
  - the clip geometry used for elevation masking
  - the map bounding box (zoom level)

Layers (bottom-to-top):
  1. Unmatched botanical regions          — dark gray
  2. Matched botanical regions            — muted per-region color
  3. Named parks from ficha              — orange fill
  4. Canton / district highlight         — indigo fill (if applicable)
  5. Elevation mask within narrowest scope — cyan
  6. Protected areas clipped to regions  — amber outline
  7. GBIF presence points                — red dots
  + Dual legend boxes + Fuentes footer
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping
from shapely.ops import unary_union
from pathlib import Path

from .ficha import DistributionFicha, EntityRef
from .geodata import (
    load_regiones_botanicas, load_protected_areas,
    load_cantones, load_distritos, filter_pa_to_regions, PATHS,
)
from .style import REGION_COLORS, FALLBACK_REGION_COLOR, mute_color
from .gazetteer import _load_entities, normalize_text, get_mobot_buffers


# ---------------------------------------------------------------------------
# Entity → GDF resolution
# ---------------------------------------------------------------------------

def _entity_target(canonical_name: str) -> tuple[str, str, str] | None:
    """Return (target_shapefile, target_attribute, target_value) for an entity."""
    df = _load_entities()
    rows = df[df["canonical_name"] == canonical_name]
    if rows.empty:
        return None
    r = rows.iloc[0]
    ts = r["target_shapefile"].strip()
    ta = r["target_attribute"].strip()
    tv = r["target_value"].strip()
    return (ts, ta, tv) if (ts and ta and tv) else None


_LOADERS = {
    "regiones_botanicas": load_regiones_botanicas,
    "protected_areas":    load_protected_areas,
    "cantones":           load_cantones,
    "distritos":          load_distritos,
}


# ---------------------------------------------------------------------------
# Locality buffer radii (metres) by feature_type and MOBOT name patterns.
# Buffer is the *initial* search radius; ecological filters (region, vertiente,
# DEM elevation) do the precision work. Keep radii conservative.
# ---------------------------------------------------------------------------

_BUFFER_RADII_BY_FTYPE: dict[str, int] = {
    "estacion_biologica": 1_500,   # well-delimited field station
    "cerro":              2_500,   # specific summit / hill
    "volcan":             2_500,   # specific volcanic peak
    "cadena_menor":       3_000,   # punta geográfica, smaller ridge
    "cuenca":             3_000,   # river drainage point
    "isla":               3_000,   # island or coastal feature
    "localidad_buffer":   7_000,   # "vecindad de X" already implies surroundings
    "localidad":          4_000,   # default town / collecting site
}

# Override for names that indicate a specific point (not a town)
_SMALL_RADIUS_NAME_PATTERNS = re.compile(
    r"\b(estaci[oó]n|cerro|punta|finca|hacienda|rancho|parcela|laguna|boca|nacimiento)\b",
    re.I,
)
_LARGE_RADIUS_NAME_PATTERNS = re.compile(
    r"\b(vecindad|alrededores|cerca|region|comarca)\b",
    re.I,
)


def _locality_radius_m(feature_type: str, mobot_name: str) -> int:
    """Return buffer radius in metres for a given locality occurrence."""
    base = _BUFFER_RADII_BY_FTYPE.get(feature_type, 4_000)
    name_lc = (mobot_name or "").lower()
    if _SMALL_RADIUS_NAME_PATTERNS.search(name_lc):
        base = min(base, 2_000)
    if _LARGE_RADIUS_NAME_PATTERNS.search(name_lc):
        base = max(base, 6_000)
    return base


def _resolve_locality_buffers(
    ficha: DistributionFicha,
    matched_regions_gdf: gpd.GeoDataFrame | None,
) -> gpd.GeoDataFrame | None:
    """
    Build radius-calibrated circles around MOBOT gazetteer point localities,
    clipped to the intersection of the matched botanical region(s) and the
    occurrence's vertiente side.

    Radii by type (conservative — ecological filters do precision work):
      estacion_biologica : 1.5 km
      cerro / volcan     : 2.5 km
      isla / cuenca      : 3.0 km
      localidad_buffer   : 7.0 km  ("vecindad de X" implies surroundings)
      localidad (default): 4.0 km

    region_informal and shapefile-resolved types (cordillera, llanura, …)
    are NOT given circles — they render as region polygons already.
    """
    from .gazetteer import _load_mobot_points
    from shapely.ops import unary_union

    pts_gdf = _load_mobot_points()
    if pts_gdf is None:
        return None

    # Vertiente mask geometries in metric CRS
    regiones = load_regiones_botanicas()
    vert_geoms: dict[str, object] = {}
    for vn in ("carib", "pacifico"):
        vgdf = regiones[regiones["vert_norm"] == vn]
        if not vgdf.empty:
            vert_geoms[vn] = unary_union(vgdf.to_crs("EPSG:5367").geometry)

    # Matched regions union for clipping
    regions_union_5367 = None
    if matched_regions_gdf is not None and not matched_regions_gdf.empty:
        regions_union_5367 = unary_union(
            matched_regions_gdf.to_crs("EPSG:5367").geometry
        )

    clipped_gdfs = []
    seen_indices: set[int] = set()

    for occ in ficha.locality_occurrences:
        indices = occ.get("mobot_row_indices", [])
        if not indices:
            continue
        idx = indices[0]
        if idx in seen_indices:
            continue
        seen_indices.add(idx)

        point_row = pts_gdf.iloc[[idx]].to_crs("EPSG:5367")
        mobot_name = occ.get("mobot_name", "") or ""
        radius = _locality_radius_m(occ["feature_type"], mobot_name)
        circle = point_row.geometry.iloc[0].buffer(radius)

        # Clip 1: to matched botanical regions
        if regions_union_5367 is not None:
            circle = circle.intersection(regions_union_5367)

        # Clip 2: to occurrence's vertiente half
        occ_vert = occ.get("vertiente")
        if occ_vert and occ_vert != "ambas":
            vn = "carib" if occ_vert == "Caribe" else "pacifico"
            if vn in vert_geoms:
                circle = circle.intersection(vert_geoms[vn])

        if circle.is_empty:
            continue

        row_copy = point_row.copy()
        row_copy["geometry"] = [circle]
        clipped_gdfs.append(row_copy)

    if not clipped_gdfs:
        return None

    combined = gpd.GeoDataFrame(
        pd.concat(clipped_gdfs, ignore_index=True), crs="EPSG:5367"
    ).to_crs("EPSG:4326")
    return combined


def _resolve_to_gdf(entities: list[EntityRef]) -> gpd.GeoDataFrame | None:
    """
    Resolve a list of EntityRef objects to their combined GeoDataFrame.
    Uses entities.csv to find (shapefile, attribute, value) for each entity.
    """
    gdfs = []
    for entity in entities:
        target = _entity_target(entity.canonical_name)
        if target is None:
            continue
        ts, ta, tv = target
        loader = _LOADERS.get(ts)
        if loader is None:
            continue
        source = loader()
        if ta not in source.columns:
            continue
        norm_tv = normalize_text(tv)
        mask = source[ta].apply(normalize_text) == norm_tv
        hit = source[mask].copy()
        if not hit.empty:
            gdfs.append(hit)
    if not gdfs:
        return None
    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    return combined


# ---------------------------------------------------------------------------
# Map bounds
# ---------------------------------------------------------------------------

_CR_BOUNDS = (-86.1, 7.9, -82.4, 11.3)   # (xmin, ymin, xmax, ymax) in EPSG:4326

# Minimum map window: always show at least this fraction of CR's lon/lat span.
# Prevents maps from zooming into a tiny corner when only one small region matches.
_CR_LON_SPAN = _CR_BOUNDS[2] - _CR_BOUNDS[0]   # ~3.7°
_CR_LAT_SPAN = _CR_BOUNDS[3] - _CR_BOUNDS[1]   # ~3.4°
_MIN_LON_SPAN = _CR_LON_SPAN * 0.55            # always show ≥55% of CR width
_MIN_LAT_SPAN = _CR_LAT_SPAN * 0.55            # always show ≥55% of CR height

_SCOPE_MARGIN = {
    "country":   0.03,
    "vertiente": 0.04,
    "region":    0.05,
    "park":      0.08,
    "canton":    0.12,
    "district":  0.15,
}


def _compute_bounds(clip_geom, scope: str) -> tuple[float, float, float, float]:
    frac = _SCOPE_MARGIN.get(scope, 0.06)
    xmin, ymin, xmax, ymax = clip_geom.bounds
    dx = max((xmax - xmin) * frac, 0.05)
    dy = max((ymax - ymin) * frac, 0.05)

    out_xmin = max(xmin - dx, _CR_BOUNDS[0])
    out_ymin = max(ymin - dy, _CR_BOUNDS[1])
    out_xmax = min(xmax + dx, _CR_BOUNDS[2])
    out_ymax = min(ymax + dy, _CR_BOUNDS[3])

    # If the clip geometry already covers ≥70% of CR in either dimension,
    # just show the whole country — avoids clipping corners off wide-span species.
    lon_coverage = (xmax - xmin) / _CR_LON_SPAN
    lat_coverage = (ymax - ymin) / _CR_LAT_SPAN
    if lon_coverage >= 0.70 or lat_coverage >= 0.70:
        return _CR_BOUNDS

    # Enforce minimum window so the map never crops to a tiny corner.
    # Expand symmetrically around the centre of the clip geometry.
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    if (out_xmax - out_xmin) < _MIN_LON_SPAN:
        half = _MIN_LON_SPAN / 2
        out_xmin = max(cx - half, _CR_BOUNDS[0])
        out_xmax = min(cx + half, _CR_BOUNDS[2])

    if (out_ymax - out_ymin) < _MIN_LAT_SPAN:
        half = _MIN_LAT_SPAN / 2
        out_ymin = max(cy - half, _CR_BOUNDS[1])
        out_ymax = min(cy + half, _CR_BOUNDS[3])

    return out_xmin, out_ymin, out_xmax, out_ymax


# ---------------------------------------------------------------------------
# Elevation overlay
# ---------------------------------------------------------------------------

def _overlay_elevation(
    ax, clip_gdf: gpd.GeoDataFrame, dem_path: Path,
    elev_min: float, elev_max: float, alpha: float = 0.82,
) -> None:
    """Mask DEM to clip_gdf polygons; paint in-range pixels cyan."""
    with rasterio.open(dem_path) as src:
        dem_crs    = src.crs
        dem_nodata = src.nodata if src.nodata is not None else -9999

        proj = clip_gdf.to_crs(dem_crs) if clip_gdf.crs != dem_crs else clip_gdf
        geoms = [mapping(g) for g in proj.geometry if g is not None and g.is_valid]
        if not geoms:
            return
        try:
            out_arr, out_transform = rio_mask(src, geoms, crop=True, nodata=dem_nodata)
        except Exception as e:
            print(f"  [WARN] Elevation mask failed: {e}")
            return

        elev = out_arr[0].astype(float)
        elev[elev == dem_nodata] = np.nan

        h, w   = elev.shape
        rgba   = np.zeros((h, w, 4), dtype=np.float32)
        in_rng = (~np.isnan(elev)) & (elev >= elev_min) & (elev <= elev_max)
        if not in_rng.any():
            print(f"  [WARN] No DEM pixels in elevation range [{elev_min}–{elev_max}]")
            return
        rgba[in_rng] = [0.13, 0.85, 0.93, alpha]

        left   = out_transform.c
        top    = out_transform.f
        right  = left + out_transform.a * w
        bottom = top  + out_transform.e * h
        ax.imshow(rgba, extent=[left, right, bottom, top],
                  origin="upper", aspect="auto", zorder=5, interpolation="nearest")


def _overlay_elevation_outlier(
    ax, clip_gdf: gpd.GeoDataFrame, dem_path: Path,
    elev_min: float, elev_max: float,
    main_min: float, main_max: float,
) -> None:
    """Semi-transparent hatched overlay for outlier elevation pixels."""
    with rasterio.open(dem_path) as src:
        dem_crs    = src.crs
        dem_nodata = src.nodata if src.nodata is not None else -9999

        proj  = clip_gdf.to_crs(dem_crs) if clip_gdf.crs != dem_crs else clip_gdf
        geoms = [mapping(g) for g in proj.geometry if g is not None and g.is_valid]
        if not geoms:
            return
        try:
            out_arr, out_transform = rio_mask(src, geoms, crop=True, nodata=dem_nodata)
        except Exception as e:
            print(f"  [WARN] Outlier elevation mask failed: {e}")
            return

        elev = out_arr[0].astype(float)
        elev[elev == dem_nodata] = np.nan

        h, w = elev.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        in_outlier = (
            (~np.isnan(elev)) &
            (elev >= elev_min) & (elev <= elev_max) &
            ~((elev >= main_min) & (elev <= main_max))
        )
        if not in_outlier.any():
            return
        rgba[in_outlier] = [0.13, 0.85, 0.93, 0.38]

        left   = out_transform.c
        top    = out_transform.f
        right  = left + out_transform.a * w
        bottom = top  + out_transform.e * h
        ax.imshow(rgba, extent=[left, right, bottom, top],
                  origin="upper", aspect="auto", zorder=4, interpolation="nearest")

        try:
            plot_gdf = clip_gdf.to_crs("EPSG:4326") if clip_gdf.crs.to_epsg() != 4326 else clip_gdf
            union_geom = unary_union(plot_gdf.geometry)
            gpd.GeoDataFrame(geometry=[union_geom], crs="EPSG:4326").plot(
                ax=ax, facecolor="none", edgecolor="#22d3ee",
                linewidth=0.0, hatch="////", alpha=0.30, zorder=6,
            )
        except Exception as e:
            print(f"  [WARN] Outlier hatch failed: {e}")


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------

def _build_right_legend(
    ficha: DistributionFicha,
    matched_parks_gdf: gpd.GeoDataFrame | None,
    pa_filtered: gpd.GeoDataFrame | None,
    presencias_gdf,
    highlight_gdf: gpd.GeoDataFrame | None,
    scope: str,
    locality_buffers_gdf: gpd.GeoDataFrame | None = None,
) -> list:
    patches = []

    if scope not in ("country", "vertiente"):
        patches.append(mpatches.Patch(color="#2e3440", label="Fuera del rango geográfico"))
        patches.append(mpatches.Patch(
            facecolor="#4a5568", label="Zona geográfica — fuera del rango altitudinal",
        ))

    if ficha.elevation.has_data():
        elev_min = ficha.elevation.min_m
        elev_max = ficha.elevation.max_m
        patches.append(mpatches.Patch(
            color="#22d3ee", label=f"Hábitat óptimo ({int(elev_min)}–{int(elev_max)} m)",
        ))
        if ficha.elevation.outlier_min_m is not None:
            out_lo = ficha.elevation.outlier_min_m
            patches.append(mpatches.Patch(
                facecolor="#22d3ee", alpha=0.35, edgecolor="#22d3ee", linewidth=0.8,
                label=f"Reg. atípicos inf. ({int(out_lo)}–{int(elev_min)} m)", hatch="////",
            ))
        if ficha.elevation.outlier_max_m is not None:
            out_hi = ficha.elevation.outlier_max_m
            patches.append(mpatches.Patch(
                facecolor="#22d3ee", alpha=0.35, edgecolor="#22d3ee", linewidth=0.8,
                label=f"Reg. atípicos sup. ({int(elev_max)}–{int(out_hi)} m)", hatch="////",
            ))

    if presencias_gdf is not None and not presencias_gdf.empty:
        patches.append(mpatches.Patch(
            color="#ff4444", label=f"Presencias GBIF (n={len(presencias_gdf)})",
        ))

    if locality_buffers_gdf is not None and not locality_buffers_gdf.empty:
        n_loc = sum(
            1 for o in ficha.locality_occurrences
            if o.get("mobot_row_indices")
        )
        patches.append(mpatches.Patch(
            facecolor="#a855f7", alpha=0.35, edgecolor="#d8b4fe", linewidth=0.8,
            label=f"Localidades Gazetero MOBOT (n={n_loc})",
        ))

    if matched_parks_gdf is not None and not matched_parks_gdf.empty:
        n_parks = len(ficha.parks)
        patches.append(mpatches.Patch(
            facecolor="#f97316", alpha=0.75, edgecolor="#fed7aa", linewidth=0.8,
            label=f"Parques mencionados (n={n_parks})",
        ))

    if highlight_gdf is not None and not highlight_gdf.empty:
        if scope == "canton":
            label = f"Cantón(es) mencionado(s) (n={len(ficha.cantons)})"
        else:
            label = f"Distrito(s) mencionado(s) (n={len(ficha.districts)})"
        patches.append(mpatches.Patch(
            facecolor="#818cf8", alpha=0.70, edgecolor="#c7d2fe", linewidth=0.8,
            label=label,
        ))

    if pa_filtered is not None and not pa_filtered.empty:
        patches.append(mpatches.Patch(
            facecolor="#f59e0b", alpha=0.6, edgecolor="#fbbf24", linewidth=0.8,
            label=f"Áreas Protegidas (n={len(pa_filtered)})",
        ))
    return patches


def _build_left_legend(
    ficha: DistributionFicha, matched_regions_gdf: gpd.GeoDataFrame | None,
) -> list:
    patches = []

    vert_colors = {"Caribe": "#29b6f6", "Pacífico": "#4caf50"}
    if ficha.vertientes:
        for v in ficha.vertientes:
            patches.append(mpatches.Patch(
                color=vert_colors.get(v, "#78909c"),
                label=f"▶ Vertiente del {v}",
            ))
    else:
        patches.append(mpatches.Patch(color="#4b5563", label="▶ Sin vertiente especificada"))

    if ficha.regions:
        shown = sorted({r.canonical_name for r in ficha.regions})[:8]
        for name in shown:
            rc = REGION_COLORS.get(name, FALLBACK_REGION_COLOR)
            muted = mute_color(rc, saturation=0.75, lightness=0.55)
            patches.append(mpatches.Patch(color=muted, label=f"  {name}"))
        extra = len(set(r.canonical_name for r in ficha.regions)) - 8
        if extra > 0:
            patches.append(mpatches.Patch(color="#374151", label=f"  ... (+{extra} más)"))
    return patches


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_distribution_map(
    ficha: DistributionFicha,
    output_path: Path | str,
    presencias_gdf=None,
) -> Path:
    """
    Render a distribution map for the given Ficha and save as PNG.

    Returns the output Path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load base shapefiles ──────────────────────────────────────────────
    regiones = load_regiones_botanicas()
    pa_all   = load_protected_areas()

    # ── Resolve entity geometries ─────────────────────────────────────────
    matched_regions_gdf   = _resolve_to_gdf(ficha.regions)    if ficha.regions   else None
    matched_parks_gdf     = _resolve_to_gdf(ficha.parks)      if ficha.parks     else None
    matched_cantons_gdf   = _resolve_to_gdf(ficha.cantons)    if ficha.cantons   else None
    matched_districts_gdf = _resolve_to_gdf(ficha.districts)  if ficha.districts else None
    locality_buffers_gdf  = _resolve_locality_buffers(ficha, matched_regions_gdf)

    # Apply vertiente filter to botanical regions
    if matched_regions_gdf is not None and "vert_norm" in matched_regions_gdf.columns:
        if len(ficha.vertientes) == 1:
            vert = "carib" if ficha.vertientes[0] == "Caribe" else "pacifico"
            vf = matched_regions_gdf[matched_regions_gdf["vert_norm"] == vert]
            if not vf.empty:
                matched_regions_gdf = vf

    # ── Determine scope and clip geometry ─────────────────────────────────
    scope = ficha.effective_scope()

    if scope == "district" and matched_districts_gdf is not None:
        clip_gdf = matched_districts_gdf
    elif scope == "canton" and matched_cantons_gdf is not None:
        clip_gdf = matched_cantons_gdf
    elif scope == "park":
        # Clip to union of: matched regions + parks + locality buffers
        # so all mentioned areas are visible, not just the park polygon
        gdfs_for_clip = []
        if matched_regions_gdf is not None and not matched_regions_gdf.empty:
            gdfs_for_clip.append(matched_regions_gdf[["geometry"]])
        if matched_parks_gdf is not None and not matched_parks_gdf.empty:
            pk = matched_parks_gdf.to_crs(regiones.crs)[["geometry"]]
            gdfs_for_clip.append(pk)
        if locality_buffers_gdf is not None and not locality_buffers_gdf.empty:
            lb = locality_buffers_gdf.to_crs(regiones.crs)[["geometry"]]
            gdfs_for_clip.append(lb)
        if gdfs_for_clip:
            clip_gdf = gpd.GeoDataFrame(
                pd.concat(gdfs_for_clip, ignore_index=True), crs=regiones.crs,
            )
        elif matched_parks_gdf is not None:
            clip_gdf = matched_parks_gdf
        else:
            clip_gdf = regiones
    elif scope == "region" and matched_regions_gdf is not None:
        clip_gdf = matched_regions_gdf
    elif scope == "vertiente" and ficha.vertientes:
        vert_gdfs = []
        for v in ficha.vertientes:
            vn = "carib" if v == "Caribe" else "pacifico"
            vert_gdfs.append(regiones[regiones["vert_norm"] == vn])
        clip_gdf = gpd.GeoDataFrame(pd.concat(vert_gdfs), crs=regiones.crs) if vert_gdfs else regiones
    else:
        clip_gdf = regiones

    # Ensure clip_gdf is in the same CRS as regiones for extent calculation
    if clip_gdf.crs != regiones.crs:
        clip_gdf = clip_gdf.to_crs(regiones.crs)
    clip_geom = unary_union(clip_gdf.geometry)

    # ── Figure setup ──────────────────────────────────────────────────────
    xmin, ymin, xmax, ymax = _compute_bounds(clip_geom, scope)
    lon_span = xmax - xmin
    lat_span = ymax - ymin
    fig_w = 9
    fig_h = fig_w * (lat_span / lon_span) * 1.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.subplots_adjust(bottom=0.09)

    # ── Layer 1: Unmatched regions (context) ──────────────────────────────
    if matched_regions_gdf is not None and not matched_regions_gdf.empty:
        matched_idx = set(matched_regions_gdf.index.tolist())
        unmatched_gdf = regiones[~regiones.index.isin(matched_idx)].copy()
    else:
        unmatched_gdf = regiones.copy()

    if not unmatched_gdf.empty:
        unmatched_gdf.plot(
            ax=ax, color="#2e3440", edgecolor="#4c566a",
            linewidth=0.4, alpha=0.7, zorder=1,
        )

    # ── Layer 2: Matched botanical regions (muted color) ──────────────────
    if matched_regions_gdf is not None and not matched_regions_gdf.empty:
        for _, row in matched_regions_gdf.iterrows():
            base = REGION_COLORS.get(row.get("Nombre", ""), FALLBACK_REGION_COLOR)
            gpd.GeoDataFrame([row], crs=regiones.crs).plot(
                ax=ax,
                color=mute_color(base, saturation=0.75, lightness=0.55),
                edgecolor=mute_color(base, saturation=0.90, lightness=0.75),
                linewidth=0.6, alpha=0.90, zorder=2,
            )

    # ── Layer 3: Named parks (orange) ─────────────────────────────────────
    if matched_parks_gdf is not None and not matched_parks_gdf.empty:
        parks_plot = (
            matched_parks_gdf.to_crs(regiones.crs)
            if matched_parks_gdf.crs != regiones.crs else matched_parks_gdf
        )
        parks_plot.plot(
            ax=ax, facecolor="#f97316", edgecolor="#fed7aa",
            linewidth=1.0, alpha=0.75, zorder=6,
        )

    # ── Layer 3.5: Canton / district highlight (indigo) ───────────────────
    highlight_gdf = None
    if scope == "canton" and matched_cantons_gdf is not None:
        highlight_gdf = (
            matched_cantons_gdf.to_crs(regiones.crs)
            if matched_cantons_gdf.crs != regiones.crs else matched_cantons_gdf
        )
        highlight_gdf.plot(
            ax=ax, facecolor="#818cf8", edgecolor="#c7d2fe",
            linewidth=1.2, alpha=0.70, zorder=6,
        )
    elif scope == "district" and matched_districts_gdf is not None:
        highlight_gdf = (
            matched_districts_gdf.to_crs(regiones.crs)
            if matched_districts_gdf.crs != regiones.crs else matched_districts_gdf
        )
        highlight_gdf.plot(
            ax=ax, facecolor="#818cf8", edgecolor="#c7d2fe",
            linewidth=1.2, alpha=0.70, zorder=6,
        )

    # ── Layer 3.7: Locality buffers from MOBOT gazetteer (purple tint) ───────
    if locality_buffers_gdf is not None and not locality_buffers_gdf.empty:
        loc_plot = (
            locality_buffers_gdf.to_crs(regiones.crs)
            if locality_buffers_gdf.crs != regiones.crs else locality_buffers_gdf
        )
        loc_plot.plot(
            ax=ax, facecolor="#a855f7", edgecolor="#d8b4fe",
            linewidth=0.8, alpha=0.35, zorder=6,
        )
        loc_plot.plot(
            ax=ax, facecolor="none", edgecolor="#d8b4fe",
            linewidth=1.0, alpha=0.60, zorder=7,
        )

    # ── Layer 4: Elevation mask within narrowest scope ────────────────────
    elev_mask_gdf = (
        highlight_gdf if highlight_gdf is not None
        else matched_parks_gdf if (matched_parks_gdf is not None and not matched_parks_gdf.empty)
        else matched_regions_gdf
    )
    if elev_mask_gdf is None:
        elev_mask_gdf = regiones

    # Merge parks + locality buffers into regions for elevation mask
    if (scope == "park" and matched_parks_gdf is not None
            and matched_regions_gdf is not None and not matched_regions_gdf.empty):
        parks_reproj = matched_parks_gdf.to_crs(matched_regions_gdf.crs)
        stub = parks_reproj[["geometry"]].assign(
            Nombre="", Vertiente="", vert_norm="pacifico"
        )
        elev_mask_gdf = gpd.GeoDataFrame(
            pd.concat([matched_regions_gdf, stub], ignore_index=True),
            crs=matched_regions_gdf.crs,
        )

    # Also merge locality buffers into elevation mask
    if locality_buffers_gdf is not None and not locality_buffers_gdf.empty:
        loc_reproj = locality_buffers_gdf.to_crs(elev_mask_gdf.crs)
        loc_stub = loc_reproj[["geometry"]].assign(
            Nombre="", Vertiente="", vert_norm=""
        )
        elev_mask_gdf = gpd.GeoDataFrame(
            pd.concat([elev_mask_gdf, loc_stub], ignore_index=True),
            crs=elev_mask_gdf.crs,
        )

    dem_path = PATHS["dem"]
    if ficha.elevation.has_data() and dem_path.exists() and not elev_mask_gdf.empty:
        elev_min = ficha.elevation.min_m
        elev_max = ficha.elevation.max_m
        _overlay_elevation(ax=ax, clip_gdf=elev_mask_gdf, dem_path=dem_path,
                           elev_min=elev_min, elev_max=elev_max)

        if ficha.elevation.outlier_min_m is not None or ficha.elevation.outlier_max_m is not None:
            out_lo = ficha.elevation.outlier_min_m if ficha.elevation.outlier_min_m is not None else elev_min
            out_hi = ficha.elevation.outlier_max_m if ficha.elevation.outlier_max_m is not None else elev_max
            _overlay_elevation_outlier(
                ax=ax, clip_gdf=elev_mask_gdf, dem_path=dem_path,
                elev_min=out_lo, elev_max=out_hi,
                main_min=elev_min, main_max=elev_max,
            )

    # ── Layer 5: Protected areas clipped to matched regions ───────────────
    pa_filtered = None
    if matched_regions_gdf is not None and not matched_regions_gdf.empty:
        pa_filtered = filter_pa_to_regions(pa_all, matched_regions_gdf)
        if pa_filtered is not None and not pa_filtered.empty:
            pa_filtered.plot(
                ax=ax, facecolor="#f59e0b", edgecolor="none", alpha=0.15, zorder=7,
            )
            pa_filtered.plot(
                ax=ax, facecolor="none", edgecolor="#fbbf24",
                linewidth=0.7, alpha=0.60, zorder=8,
            )

    # ── Layer 6: GBIF presence points ────────────────────────────────────
    if presencias_gdf is not None and not presencias_gdf.empty:
        ax.scatter(
            presencias_gdf.geometry.x,
            presencias_gdf.geometry.y,
            s=12, c="#ff4444", edgecolors="#ffaaaa",
            linewidths=0.4, alpha=0.85, zorder=10,
        )

    # ── Title + x-label ───────────────────────────────────────────────────
    sp_display = ficha.species.replace("_", " ")
    ax.set_title(sp_display, color="white", fontsize=13, fontweight="bold", pad=4)
    ax.set_xlabel(
        f"Hábitat potencial — Manual de Plantas CR  |  Elevación: {ficha.elevation.label()}",
        color="#9ca3af", fontsize=8,
    )

    # ── Legends ───────────────────────────────────────────────────────────
    right_patches = _build_right_legend(
        ficha, matched_parks_gdf, pa_filtered, presencias_gdf, highlight_gdf, scope,
        locality_buffers_gdf=locality_buffers_gdf,
    )
    left_patches = _build_left_legend(ficha, matched_regions_gdf)

    if right_patches:
        leg_right = ax.legend(
            handles=right_patches, loc="lower right", fontsize=7.5,
            facecolor="#1f2937", labelcolor="white",
            framealpha=0.9, edgecolor="#374151",
        )
        ax.add_artist(leg_right)

    if left_patches:
        ax.legend(
            handles=left_patches, loc="lower left", fontsize=7.0,
            facecolor="#1f2937", labelcolor="white",
            framealpha=0.9, edgecolor="#374151",
        )

    # ── Fuentes footer ────────────────────────────────────────────────────
    fig.text(
        0.5, 0.015,
        "Fuentes: Regiones Botánicas (José Araya, 2026) · Áreas Silvestres Protegidas (SINAC) · "
        "Altitud (SRTM, INEC) · Presencias GBIF · Manual de Plantas de Costa Rica",
        ha="center", fontsize=5.8, color="#6b7280",
        transform=fig.transFigure,
    )

    # ── Axis chrome ───────────────────────────────────────────────────────
    ax.tick_params(colors="#6b7280", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    fig.savefig(output_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Distribution map saved → {output_path}")
    return output_path
