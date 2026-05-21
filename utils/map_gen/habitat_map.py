"""
habitat_map.py — V2
--------------------
Generates a habitat map for a species using:

  1. Regiones Botánicas de Costa Rica (José Araya, 2026)
     Shapefile: Jose_regiones_botanicas_con_vertiente.shp  (33 polygons, EPSG:5367)
     Columns: Nombre, Vertiente
  2. DEM raster (altitud_cr.tif) — elevation mask within matched regions
  3. Áreas Silvestres Protegidas (SINAC) — filtered to highlighted regions
  4. GBIF presence points (optional) — ground-truth overlay

Visual logic:
  - Unmatched regions       → dark gray (context)
  - Matched, out-of-range   → muted color per vertiente side
  - Matched, in elev range  → bright cyan (optimal habitat)
  - Protected areas         → green outline/fill overlay
  - GBIF points             → red dots on top
  - Two legend boxes + Fuentes footer

Region matching:
  TRANSLATION_TABLE_V2 maps Manual vocabulary → Nombre values in the shapefile.
  Vertiente filter uses the Vertiente shapefile attribute directly.
"""

import argparse
import re
import sys
import os
import unicodedata
from pathlib import Path

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

# Ensure project root is on sys.path before importing local packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as _cfg

from utils.distribution_map.style import (
    REGION_COLORS,
    FALLBACK_REGION_COLOR as _FALLBACK_COLOR,
    mute_color as _mute_color,
)
from utils.distribution_map.geodata import (
    load_regiones_botanicas,
    load_protected_areas,
    filter_pa_to_regions,
)

# ---------------------------------------------------------------------------
# Paths (for DEM and CLI usage)
# ---------------------------------------------------------------------------

REGIONES_BOTANICAS_SHP = Path(_cfg.REGIONES_BOTANICAS_SHP)
PROTECTED_AREAS_SHP    = Path(_cfg.PROTECTED_AREAS_V2_SHP)
DEM_PATH               = Path(_cfg.DEM_PATH)

# ---------------------------------------------------------------------------
# TRANSLATION TABLE V2
# Maps Manual de Plantas vocabulary → list of Nombre values in the shapefile.
# Patterns are matched against accent-normalized lowercase text.
# Order matters: more specific patterns must precede broader ones.
#
# The hardcoded table below is the baseline. An external CSV override file
# (data_raw/region_xref.csv) is merged on top at load time — rows with
# enabled=yes are appended, giving you an editable lookup without touching code.
# CSV columns: pattern, region_nombre, description, enabled
# ---------------------------------------------------------------------------
_TRANSLATION_TABLE_BASE: list[tuple[str, list[str]]] = [
    # ── Catch-alls (must precede individual cordillera entries) ──────────────
    (
        r"todas las cords?\.?\s*principales|todas las cordilleras\s*principales",
        ["Cordillera de Guanacaste", "Cordillera de Tilarán",
         "Cordillera Central", "Cordillera de Talamanca"],
    ),
    (
        r"toda\s*(?:la\s*)?vert\.?\s*carib|toda\s*(?:la\s*)?vertiente\s*carib",
        ["Llanuras de Tortuguero / Santa Clara", "Llanuras de San Carlos",
         "LLanura de los Guatusos", "Baja Talamanca"],
    ),
    # ── Cordilleras ──────────────────────────────────────────────────────────
    (r"cords?\.?\s*de\s*guanacaste|cords?\.?\s*guanacaste",
        ["Cordillera de Guanacaste"]),
    (r"cords?\.?\s*de\s*tilaran|cords?\.?\s*tilaran",
        ["Cordillera de Tilarán"]),
    (r"cords?\.?\s*central",
        ["Cordillera Central"]),
    (r"cords?\.?\s*de\s*talamanca|cords?\.?\s*talamanca|cord\.?\s*talamanca|de\s*talamanca",
        ["Cordillera de Talamanca"]),
    # ── Llanuras ─────────────────────────────────────────────────────────────
    (r"llanura(?:s)?\s*(?:de\s*)?tortuguero",
        ["Llanuras de Tortuguero / Santa Clara"]),
    (r"llanura(?:s)?\s*de\s*san\s*carlos",
        ["Llanuras de San Carlos"]),
    (r"llanura(?:s)?\s*de\s*(?:los\s*)?guatusos?|llanura(?:s)?\s*(?:de\s*)?guatuso",
        ["LLanura de los Guatusos"]),
    (r"llanura(?:s)?\s*de\s*guanacaste",
        ["Llanuras de Guanacaste"]),
    (r"llanura(?:s)?\s*del\s*diquis",
        ["Llanuras del Diquís"]),
    # ── Valles (specific before bare "valle central") ─────────────────────────
    (r"valle\s*central\s*oriental",
        ["Valle Central Oriental"]),
    (r"valle\s*central\s*occidental",
        ["Valle Central Occidental"]),
    (r"valle\s*central(?!\s*(?:oriental|occidental))",
        ["Valle Central Oriental", "Valle Central Occidental"]),
    (r"valle\s*(?:de\s*)?(?:el\s*)?general|valles?\s*de\s*general",
        ["Valle del General"]),
    (r"valle\s*(?:de\s*|del\s*)?(?:coto\s*)?brus|coto\s*brus",
        ["Valle del Coto Brus"]),
    # ── Fila Costeña (specific before bare) ──────────────────────────────────
    (r"s\.?\s*fila\s*costen[na]|sur\s*fila\s*costen[na]|fila\s*costen[na]\s*sur",
        ["Fila Costeña Sur"]),
    (r"n\.?\s*fila\s*costen[na]|norte\s*fila\s*costen[na]|fila\s*costen[na]\s*norte",
        ["Fila Costeña Norte"]),
    (r"fila\s*costen[na]",
        ["Fila Costeña Norte", "Fila Costeña Sur"]),
    # ── Penínsulas ───────────────────────────────────────────────────────────
    (r"pen[ii]nsula\s*de\s*nicoya|pen[ii]nsula\s*de\s*santa\s*elena|nicoya",
        ["Península de Nicoya"]),
    (r"pen[ii]nsula\s*de\s*osa|osa.*golfito|golfito",
        ["Península de Osa - Golfito"]),
    # ── Otros ────────────────────────────────────────────────────────────────
    (r"tarcoles|terraba",
        ["Tárcoles - Térraba"]),
    (r"baja\s*talamanca",
        ["Baja Talamanca"]),
    (r"puriscal|los\s*santos(?!\s*guanacaste)",
        ["Puriscal - Los Santos"]),
    (r"turrubares",
        ["Turrubares"]),
    (r"punta\s*burica",
        ["Punta Burica"]),
    (r"coto\s*colorado",
        ["Coto Colorado"]),
    (r"filas?\s*chonta|filas?\s*nara|chonta\s*(?:y\s*)?nara",
        ["Filas Chonta y Nara"]),
]

# ---------------------------------------------------------------------------
# Load external xref CSV and merge into the translation table
# ---------------------------------------------------------------------------
_XREF_PATH = Path(__file__).parent.parent.parent / "data_raw" / "region_xref.csv"

def _load_translation_table() -> list[tuple[str, list[str]]]:
    table = list(_TRANSLATION_TABLE_BASE)
    if _XREF_PATH.exists():
        try:
            import pandas as _pd
            xref = _pd.read_csv(_XREF_PATH)
            added = 0
            for _, row in xref.iterrows():
                if str(row.get("enabled", "yes")).strip().lower() != "yes":
                    continue
                pattern = str(row["pattern"]).strip()
                nombre  = str(row["region_nombre"]).strip()
                if pattern and nombre:
                    table.append((pattern, [nombre]))
                    added += 1
            if added:
                print(f"  [xref] Loaded {added} extra region mappings from region_xref.csv")
        except Exception as e:
            print(f"  [xref] Could not load region_xref.csv: {e}")
    return table

TRANSLATION_TABLE_V2 = _load_translation_table()

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text))
    return nfkd.encode("ascii", "ignore").decode("ascii").lower()


def match_parks(geographic_notes: str, pa_gdf: gpd.GeoDataFrame) -> set[str]:
    """
    Return set of nombre_asp values from the PA layer whose name appears
    in geographic_notes. Matching is accent-normalized and case-insensitive.
    Short names (<5 chars) require word boundaries to avoid false positives.
    """
    text = _normalize(geographic_notes)
    matched: set[str] = set()
    for nombre in pa_gdf["nombre_asp"].dropna().unique():
        norm = _normalize(nombre)
        if len(norm) < 5:
            pat = r'\b' + re.escape(norm) + r'\b'
        else:
            pat = re.escape(norm)
        if re.search(pat, text):
            matched.add(nombre)
    return matched


def match_regions_v2(geographic_notes: str) -> tuple[set[str], str]:
    """
    Parse geographic_notes and return (matched_nombres, vert_flag).

    matched_nombres : set of Nombre values found via TRANSLATION_TABLE_V2
    vert_flag       : "carib" | "pacifico" | "ambas" | "none"
    """
    text = _normalize(geographic_notes)

    has_caribe   = bool(re.search(r"vert\.?\s*carib|vertiente\s*carib", text))
    has_pacifico = bool(re.search(r"vert\.?\s*pac|vertiente\s*pacif", text))
    ambas        = bool(re.search(r"ambas\s*vert", text))

    if ambas or (has_caribe and has_pacifico):
        vert_flag = "ambas"
    elif has_caribe:
        vert_flag = "carib"
    elif has_pacifico:
        vert_flag = "pacifico"
    else:
        vert_flag = "none"

    matched: set[str] = set()
    for pattern, nombres in TRANSLATION_TABLE_V2:
        if re.search(pattern, text):
            matched.update(nombres)

    return matched, vert_flag


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_habitat_map(
    species_name: str,
    geographic_notes: str,
    elevation_min: float,
    elevation_max: float,
    presencias_gdf=None,
    output_path: str | Path = "outputs/habitat_map.png",
    dem_path: Path = DEM_PATH,
    elev_outlier_min: float | None = None,
    elev_outlier_max: float | None = None,
) -> Path:
    """
    Generate the species habitat map and save as PNG.

    Layers (bottom to top):
      1. Unmatched regions — dark gray
      2. Matched botanical regions — muted color per region
      3. Named parks from Manual — bright orange fill + outline
      3b. Elevation highlight (cyan) within regions + named parks
      3c. Outlier elevation bands (hatched cyan)
      4. All protected areas clipped to matched regions — amber outline
      5. GBIF points — red dots
    Plus: dual legend boxes + Fuentes footer.

    Returns the output Path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load shapefiles ───────────────────────────────────────────────────
    regiones = load_regiones_botanicas()
    pa_all   = load_protected_areas()

    notes_text = geographic_notes or ""

    # ── Region matching ───────────────────────────────────────────────────
    matched_nombres, vert_flag = match_regions_v2(notes_text)

    if matched_nombres:
        nombre_mask = regiones["Nombre"].isin(matched_nombres)
        if vert_flag in ("carib", "pacifico"):
            vert_mask = regiones["vert_norm"] == vert_flag
            matched_mask = nombre_mask & vert_mask
            if matched_mask.sum() == 0:
                matched_mask = nombre_mask
        else:
            matched_mask = nombre_mask
        fallback = False
    else:
        print(f"  [WARN] No region match for: '{notes_text[:60]}' — showing all")
        matched_mask = np.ones(len(regiones), dtype=bool)
        fallback = True

    matched_gdf   = regiones[matched_mask].copy()
    unmatched_gdf = regiones[~matched_mask].copy()

    # ── Named park matching ───────────────────────────────────────────────
    matched_park_names = match_parks(notes_text, pa_all)
    named_parks_gdf = (
        pa_all[pa_all["nombre_asp"].isin(matched_park_names)].copy()
        if matched_park_names else pa_all.iloc[0:0]
    )
    if matched_park_names:
        print(f"  [parks] Named parks matched: {sorted(matched_park_names)}")

    # ── Protected areas clipped to matched regions (background layer) ─────
    pa_filtered = (
        filter_pa_to_regions(pa_all, matched_gdf)
        if not matched_gdf.empty
        else pa_all.iloc[0:0]
    )

    # ── Combined geometry for elevation masking (regions + named parks) ───
    elev_mask_gdf = matched_gdf.copy()
    if not named_parks_gdf.empty:
        parks_reproj = named_parks_gdf.to_crs(matched_gdf.crs) if named_parks_gdf.crs != matched_gdf.crs else named_parks_gdf
        elev_mask_gdf = gpd.GeoDataFrame(
            pd.concat([matched_gdf, parks_reproj[["geometry"]].assign(
                Nombre="", Vertiente="", vert_norm="pacifico"
            )], ignore_index=True),
            crs=matched_gdf.crs,
        )

    # ── Elevation flags ────────────────────────────────────────────────────
    has_elevation = (
        Path(dem_path).exists()
        and elevation_min is not None
        and elevation_max is not None
        and not (np.isnan(float(elevation_min)) or np.isnan(float(elevation_max)))
    )
    elev_min = float(elevation_min) if has_elevation else 0
    elev_max = float(elevation_max) if has_elevation else 9999

    has_outlier = (
        has_elevation
        and not matched_gdf.empty
        and (elev_outlier_min is not None or elev_outlier_max is not None)
        and not (
            (elev_outlier_min is not None and np.isnan(float(elev_outlier_min))) or
            (elev_outlier_max is not None and np.isnan(float(elev_outlier_max)))
        )
    )

    # ── Figure layout ──────────────────────────────────────────────────────
    XLIM = (-86.1, -82.4)
    YLIM = (7.9,   11.3)
    lon_span = XLIM[1] - XLIM[0]
    lat_span = YLIM[1] - YLIM[0]
    fig_w = 9
    fig_h = fig_w * (lat_span / lon_span) * 1.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    plt.subplots_adjust(bottom=0.09)   # reserve space for Fuentes footer

    # ── Layer 1: Unmatched regions ────────────────────────────────────────
    if not unmatched_gdf.empty:
        unmatched_gdf.plot(
            ax=ax, color="#2e3440", edgecolor="#4c566a",
            linewidth=0.4, alpha=0.7, zorder=1,
        )

    # ── Layer 2: Matched regions (per-region color, muted) ────────────────
    for _, row in matched_gdf.iterrows():
        base_color = REGION_COLORS.get(row.get("Nombre", ""), _FALLBACK_COLOR)
        muted_fill = _mute_color(base_color, saturation=0.75, lightness=0.55)
        muted_edge = _mute_color(base_color, saturation=0.90, lightness=0.75)
        gpd.GeoDataFrame([row], crs=regiones.crs).plot(
            ax=ax, color=muted_fill, edgecolor=muted_edge,
            linewidth=0.6, alpha=0.90, zorder=2,
        )

    # ── Layer 3: Named parks (bright orange fill) ─────────────────────────
    if not named_parks_gdf.empty:
        named_parks_gdf.plot(
            ax=ax, facecolor="#f97316", edgecolor="#fed7aa",
            linewidth=1.0, alpha=0.75, zorder=6,
        )

    # ── Layer 3b: Elevation highlight (regions + named parks) ────────────
    if has_elevation and not elev_mask_gdf.empty:
        _overlay_elevation(
            ax=ax, matched_gdf=elev_mask_gdf, dem_path=dem_path,
            elev_min=elev_min, elev_max=elev_max,
        )

    # ── Layer 3c: Outlier elevation bands ────────────────────────────────
    if has_outlier and not elev_mask_gdf.empty:
        out_lo = float(elev_outlier_min) if elev_outlier_min is not None else elev_min
        out_hi = float(elev_outlier_max) if elev_outlier_max is not None else elev_max
        _overlay_elevation_outlier(
            ax=ax, matched_gdf=elev_mask_gdf, dem_path=dem_path,
            elev_min=out_lo, elev_max=out_hi,
            main_min=elev_min, main_max=elev_max,
        )

    # ── Layer 4: All protected areas clipped to matched regions ──────────
    if not pa_filtered.empty:
        pa_filtered.plot(
            ax=ax, facecolor="#f59e0b", edgecolor="none",
            alpha=0.15, zorder=7,
        )
        pa_filtered.plot(
            ax=ax, facecolor="none", edgecolor="#fbbf24",
            linewidth=0.7, alpha=0.60, zorder=8,
        )

    # ── Layer 5: GBIF points ──────────────────────────────────────────────
    if presencias_gdf is not None and not presencias_gdf.empty:
        ax.scatter(
            presencias_gdf.geometry.x,
            presencias_gdf.geometry.y,
            s=12, c="#ff4444", edgecolors="#ffaaaa",
            linewidths=0.4, alpha=0.85, zorder=10,
        )

    # ── Title & x-label ───────────────────────────────────────────────────
    sp_display = species_name.replace("_", " ")
    elev_txt = f"{int(elev_min)}–{int(elev_max)} m" if has_elevation else "elevación no disponible"
    if has_outlier:
        out_lo_v = float(elev_outlier_min) if elev_outlier_min is not None else elev_min
        out_hi_v = float(elev_outlier_max) if elev_outlier_max is not None else elev_max
        parts = []
        if elev_outlier_min is not None:
            parts.append(f"inf: {int(out_lo_v)}–{int(elev_min)} m")
        if elev_outlier_max is not None:
            parts.append(f"sup: {int(elev_max)}–{int(out_hi_v)} m")
        elev_txt += f"  (atípicos — {', '.join(parts)})"

    ax.set_title(f"{sp_display}", color="white", fontsize=13, fontweight="bold", pad=4)
    ax.set_xlabel(
        f"Hábitat potencial — Manual de Plantas CR  |  Elevación: {elev_txt}",
        color="#9ca3af", fontsize=8,
    )

    # ── Right legend (layers) ─────────────────────────────────────────────
    right_patches = []
    if not fallback:
        right_patches.append(mpatches.Patch(color="#2e3440", label="Fuera del rango geográfico"))
        right_patches.append(mpatches.Patch(
            facecolor="#4a5568",
            label="Zona geográfica — fuera del rango altitudinal",
        ))
    if has_elevation:
        right_patches.append(mpatches.Patch(
            color="#22d3ee", label=f"Hábitat óptimo ({int(elev_min)}–{int(elev_max)} m)",
        ))
    if has_outlier:
        out_lo_v = float(elev_outlier_min) if elev_outlier_min is not None else elev_min
        out_hi_v = float(elev_outlier_max) if elev_outlier_max is not None else elev_max
        if elev_outlier_min is not None:
            right_patches.append(mpatches.Patch(
                facecolor="#22d3ee", alpha=0.35, edgecolor="#22d3ee", linewidth=0.8,
                label=f"Reg. atípicos inf. ({int(out_lo_v)}–{int(elev_min)} m)", hatch="////",
            ))
        if elev_outlier_max is not None:
            right_patches.append(mpatches.Patch(
                facecolor="#22d3ee", alpha=0.35, edgecolor="#22d3ee", linewidth=0.8,
                label=f"Reg. atípicos sup. ({int(elev_max)}–{int(out_hi_v)} m)", hatch="////",
            ))
    if presencias_gdf is not None and not presencias_gdf.empty:
        right_patches.append(mpatches.Patch(
            color="#ff4444", label=f"Presencias GBIF (n={len(presencias_gdf)})",
        ))
    if not named_parks_gdf.empty:
        right_patches.append(mpatches.Patch(
            facecolor="#f97316", alpha=0.75, edgecolor="#fed7aa", linewidth=0.8,
            label=f"Parques mencionados (n={len(matched_park_names)})",
        ))
    if not pa_filtered.empty:
        right_patches.append(mpatches.Patch(
            facecolor="#f59e0b", alpha=0.6, edgecolor="#fbbf24", linewidth=0.8,
            label=f"Áreas Protegidas (n={len(pa_filtered)})",
        ))

    # ── Left legend (vertientes + region names with their specific colors) ─
    left_patches = []
    vert_header_color = {
        "carib":    "#29b6f6",   # blue tint for Caribe header
        "pacifico": "#4caf50",   # green tint for Pacifico header
        "ambas":    "#78909c",
        "none":     "#4b5563",
    }
    vert_header_entries = {
        "carib":    [("▶ Vertiente del Caribe",   vert_header_color["carib"])],
        "pacifico": [("▶ Vertiente del Pacífico",  vert_header_color["pacifico"])],
        "ambas":    [("▶ Vertiente del Caribe",    vert_header_color["carib"]),
                     ("▶ Vertiente del Pacífico",  vert_header_color["pacifico"])],
        "none":     [("▶ Sin vertiente especificada", vert_header_color["none"])],
    }
    for label, color in vert_header_entries.get(vert_flag, vert_header_entries["none"]):
        left_patches.append(mpatches.Patch(color=color, label=label))

    shown = sorted(matched_nombres)[:8]
    for nombre in shown:
        region_color = REGION_COLORS.get(nombre, _FALLBACK_COLOR)
        muted = _mute_color(region_color, saturation=0.75, lightness=0.55)
        left_patches.append(mpatches.Patch(color=muted, label=f"  {nombre}"))
    if len(matched_nombres) > 8:
        left_patches.append(mpatches.Patch(
            color="#374151", label=f"  ... (+{len(matched_nombres) - 8} más)",
        ))

    # Render right legend first, then re-add it so left legend doesn't replace it
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

    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Habitat map saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Elevation overlay helpers (unchanged from V1)
# ---------------------------------------------------------------------------

def _overlay_elevation(ax, matched_gdf, dem_path, elev_min, elev_max):
    """Mask DEM to matched polygons, paint in-range pixels cyan."""
    with rasterio.open(dem_path) as src:
        dem_crs    = src.crs
        dem_nodata = src.nodata if src.nodata is not None else -9999

        if matched_gdf.crs.to_epsg() != int(str(dem_crs).split(":")[-1]):
            matched_proj = matched_gdf.to_crs(dem_crs)
        else:
            matched_proj = matched_gdf

        geoms = [mapping(g) for g in matched_proj.geometry if g is not None and g.is_valid]
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
        rgba[in_rng] = [0.13, 0.85, 0.93, 0.82]

        if not in_rng.any():
            print(f"  [WARN] No DEM pixels in elevation range [{elev_min}–{elev_max}]")
            return

        left   = out_transform.c
        top    = out_transform.f
        right  = left + out_transform.a * w
        bottom = top  + out_transform.e * h

        ax.imshow(rgba, extent=[left, right, bottom, top],
                  origin="upper", aspect="auto", zorder=5, interpolation="nearest")


def _overlay_elevation_outlier(ax, matched_gdf, dem_path, elev_min, elev_max,
                                main_min, main_max):
    """Hatched semi-transparent overlay for outlier elevation pixels."""
    with rasterio.open(dem_path) as src:
        dem_crs    = src.crs
        dem_nodata = src.nodata if src.nodata is not None else -9999

        if matched_gdf.crs.to_epsg() != int(str(dem_crs).split(":")[-1]):
            matched_proj = matched_gdf.to_crs(dem_crs)
        else:
            matched_proj = matched_gdf

        geoms = [mapping(g) for g in matched_proj.geometry if g is not None and g.is_valid]
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
        rgba[in_outlier] = [0.13, 0.85, 0.93, 0.38]

        if not in_outlier.any():
            return

        left   = out_transform.c
        top    = out_transform.f
        right  = left + out_transform.a * w
        bottom = top  + out_transform.e * h

        ax.imshow(rgba, extent=[left, right, bottom, top],
                  origin="upper", aspect="auto", zorder=4, interpolation="nearest")

        try:
            if matched_gdf.crs.to_epsg() != 4326:
                plot_gdf = matched_gdf.to_crs(epsg=4326)
            else:
                plot_gdf = matched_gdf
            union_geom = unary_union(plot_gdf.geometry)
            gpd.GeoDataFrame(geometry=[union_geom], crs="EPSG:4326").plot(
                ax=ax, facecolor="none", edgecolor="#22d3ee",
                linewidth=0.0, hatch="////", alpha=0.30, zorder=6,
            )
        except Exception as e:
            print(f"  [WARN] Outlier hatch polygon failed: {e}")


# ---------------------------------------------------------------------------
# CLI (standalone usage)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate species habitat map from Manual de Plantas + DEM + Regiones Botanicas."
    )
    parser.add_argument("--species",  required=True, help="Species name")
    parser.add_argument("--notes",    required=True, help="geographic_notes text from Manual")
    parser.add_argument("--elev-min", type=float, default=None)
    parser.add_argument("--elev-max", type=float, default=None)
    parser.add_argument("--output",   default="outputs/habitat_map.png")
    args = parser.parse_args()

    generate_habitat_map(
        species_name=args.species,
        geographic_notes=args.notes,
        elevation_min=args.elev_min,
        elevation_max=args.elev_max,
        output_path=args.output,
    )
