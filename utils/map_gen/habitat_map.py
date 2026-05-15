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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Paths (loaded from project config)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as _cfg

REGIONES_BOTANICAS_SHP = Path(_cfg.REGIONES_BOTANICAS_SHP)
PROTECTED_AREAS_SHP    = Path(_cfg.PROTECTED_AREAS_V2_SHP)
DEM_PATH               = Path(_cfg.DEM_PATH)

# ---------------------------------------------------------------------------
# TRANSLATION TABLE V2
# Maps Manual de Plantas vocabulary → list of Nombre values in the shapefile.
# Patterns are matched against accent-normalized lowercase text.
# Order matters: more specific patterns must precede broader ones.
# ---------------------------------------------------------------------------
TRANSLATION_TABLE_V2: list[tuple[str, list[str]]] = [
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
# Per-region color palette — one stable color per Nombre value.
# Medium-brightness colors chosen for readability on the dark (#111827) background.
# Grouped semantically: cordilleras=blues, llanuras=greens, valles=oranges,
# peninsulas=purples, filas=pinks, others=mixed.
# ---------------------------------------------------------------------------
REGION_COLORS: dict[str, str] = {
    # Cordilleras
    "Cordillera de Guanacaste":             "#4fc3f7",  # sky blue
    "Cordillera de Tilarán":               "#29b6f6",  # bright blue
    "Cordillera Central":                  "#03a9f4",  # medium blue
    "Cordillera de Talamanca":             "#0288d1",  # deep blue
    # Llanuras
    "Llanuras de Guanacaste":             "#81c784",  # light green
    "Llanuras de Tortuguero / Santa Clara": "#4caf50", # green
    "Llanuras de San Carlos":             "#26a69a",  # teal-green
    "LLanura de los Guatusos":            "#00897b",  # teal
    "Llanuras del Diquís":               "#80cbc4",  # pale teal
    # Valles
    "Valle Central Oriental":             "#ffb74d",  # amber-orange
    "Valle Central Occidental":           "#ffa726",  # orange
    "Valle del General":                  "#ff8a65",  # salmon-orange
    "Valle del Coto Brus":               "#ff7043",  # deep orange
    # Penínsulas
    "Península de Nicoya":               "#ce93d8",  # lilac
    "Península de Osa - Golfito":        "#ab47bc",  # purple
    # Filas Costeñas
    "Fila Costeña Norte":               "#f06292",  # pink
    "Fila Costeña Sur":                 "#ec407a",  # deep pink
    # Others
    "Tárcoles - Térraba":              "#dce775",  # yellow-green
    "Baja Talamanca":                    "#ffee58",  # yellow
    "Puriscal - Los Santos":            "#a5d6a7",  # pale green
    "Turrubares":                        "#80deea",  # cyan
    "Coto Colorado":                     "#b0bec5",  # blue-gray
    "Punta Burica":                      "#f48fb1",  # light pink
    "Filas Chonta y Nara":              "#c5e1a5",  # pale lime
}
_FALLBACK_COLOR = "#78909c"   # used for any Nombre not in the palette

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text))
    return nfkd.encode("ascii", "ignore").decode("ascii").lower()


def _mute_color(hex_color: str, saturation: float = 0.3, lightness: float = 0.4) -> str:
    """Return a desaturated/darkened version of a hex color."""
    import colorsys
    r, g, b = mcolors.to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s * saturation, v * lightness)
    return mcolors.to_hex((r2, g2, b2))


def load_regiones_botanicas() -> gpd.GeoDataFrame:
    """Load, reproject, and clean-up the botanical regions shapefile."""
    gdf = gpd.read_file(REGIONES_BOTANICAS_SHP)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    # Fix malformed "NULLVertiente..." values from shapefile data quality issue
    null_mask = gdf["Vertiente"].str.startswith("NULL", na=False)
    if null_mask.any():
        print(f"  [WARN] Fixing {null_mask.sum()} malformed Vertiente value(s) in shapefile")
        gdf.loc[null_mask, "Vertiente"] = (
            gdf.loc[null_mask, "Vertiente"].str.replace(r"^NULL", "", regex=True)
        )
    gdf["vert_norm"] = gdf["Vertiente"].apply(
        lambda v: "carib" if "carib" in str(v).lower() else "pacifico"
    )
    return gdf


def load_protected_areas() -> gpd.GeoDataFrame:
    """Load and reproject the protected areas shapefile."""
    gdf = gpd.read_file(PROTECTED_AREAS_SHP)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def filter_pa_to_regions(pa_gdf: gpd.GeoDataFrame,
                          region_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Return protected areas clipped to the union of highlighted regions.
    Geometries are truncated at region boundaries so no PA outline
    extends into non-highlighted areas.
    """
    if region_gdf.empty:
        return pa_gdf.iloc[0:0]
    try:
        # Repair invalid geometries before clip (common in SINAC shapefiles)
        pa_valid     = pa_gdf.copy()
        pa_valid["geometry"] = pa_valid.geometry.buffer(0)
        region_valid = region_gdf.copy()
        region_valid["geometry"] = region_valid.geometry.buffer(0)

        region_union = unary_union(region_valid.geometry)
        clipped = gpd.clip(pa_valid, region_union)
        # Drop empty slivers from clipping artefacts
        clipped = clipped[~clipped.is_empty].reset_index(drop=True)
        return clipped
    except Exception as e:
        print(f"  [WARN] Protected-area clip failed: {e}")
        return pa_gdf.iloc[0:0]


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
      2. Matched geographic regions — muted by vertiente side
      3. Elevation highlight (cyan) within matched regions
      3b. Outlier elevation bands (hatched cyan)
      4. Protected areas — green outline/fill
      5. GBIF points — red dots
    Plus: dual legend boxes + Fuentes footer.

    Returns the output Path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load shapefiles ───────────────────────────────────────────────────
    regiones = load_regiones_botanicas()
    pa_all   = load_protected_areas()

    # ── Region matching ───────────────────────────────────────────────────
    matched_nombres, vert_flag = match_regions_v2(geographic_notes or "")

    if matched_nombres:
        nombre_mask = regiones["Nombre"].isin(matched_nombres)
        if vert_flag in ("carib", "pacifico"):
            vert_mask = regiones["vert_norm"] == vert_flag
            matched_mask = nombre_mask & vert_mask
            if matched_mask.sum() == 0:
                # Vertiente filter removed everything — fall back to name-only
                matched_mask = nombre_mask
        else:
            matched_mask = nombre_mask
        fallback = False
    else:
        print(f"  [WARN] No region match for: '{(geographic_notes or '')[:60]}' — showing all")
        matched_mask = np.ones(len(regiones), dtype=bool)
        fallback = True

    matched_gdf   = regiones[matched_mask].copy()
    unmatched_gdf = regiones[~matched_mask].copy()

    # ── Protected areas filtered to highlighted regions ───────────────────
    pa_filtered = (
        filter_pa_to_regions(pa_all, matched_gdf)
        if not matched_gdf.empty
        else pa_all.iloc[0:0]
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

    # ── Layer 3: Elevation highlight ──────────────────────────────────────
    if has_elevation and not matched_gdf.empty:
        _overlay_elevation(
            ax=ax, matched_gdf=matched_gdf, dem_path=dem_path,
            elev_min=elev_min, elev_max=elev_max,
        )

    # ── Layer 3b: Outlier elevation bands ────────────────────────────────
    if has_outlier:
        out_lo = float(elev_outlier_min) if elev_outlier_min is not None else elev_min
        out_hi = float(elev_outlier_max) if elev_outlier_max is not None else elev_max
        _overlay_elevation_outlier(
            ax=ax, matched_gdf=matched_gdf, dem_path=dem_path,
            elev_min=out_lo, elev_max=out_hi,
            main_min=elev_min, main_max=elev_max,
        )

    # ── Layer 4: Protected areas (clipped to matched regions) ────────────
    if not pa_filtered.empty:
        pa_filtered.plot(
            ax=ax, facecolor="#f59e0b", edgecolor="none",
            alpha=0.20, zorder=7,
        )
        pa_filtered.plot(
            ax=ax, facecolor="none", edgecolor="#fbbf24",
            linewidth=0.9, alpha=0.80, zorder=8,
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
