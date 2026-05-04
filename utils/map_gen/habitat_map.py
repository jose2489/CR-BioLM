"""
habitat_map.py
--------------
Generates a single high-quality habitat map for a species using:

  1. Unidades Fitogeográficas shapefile (B. Hammel, 2014) — 43 unique polygons
  2. DEM raster (altitud_cr.tif) — elevation mask within matched units
  3. GBIF presence points (optional) — ground-truth overlay

Visual logic:
  - All unmatched units → gray (context)
  - Matched units but outside elevation range → muted color
  - Matched units AND inside elevation range → deep saturated color  ← the habitat
  - GBIF points → red dots on top

Translation table maps Manual geographic_notes vocabulary → SUBUNIDAD codes.

Usage (standalone):
    python utils/map_gen/habitat_map.py \
        --species "Werauhia kupperiana" \
        --notes "vert. Carib. Cords. de Guanacaste" \
        --elev-min 0 --elev-max 1100 \
        --output outputs/test_habitat_map.png
"""

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.transform import rowcol
from shapely.geometry import mapping

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SHP_PATH = Path(r"C:/Users/Jose/Documents/Tesis/raw_data/shapesunidadesfitogeogrficas/unidades fitogeograficas_marzo2014.shp")
DEM_PATH = Path("data_raw/topography/altitud_cr.tif")

# ---------------------------------------------------------------------------
# TRANSLATION TABLE
# Manual vocabulary token → list of SUBUNIDAD codes that match
#
# Structure: list of (regex_pattern, [subunidad, ...]) tuples.
# Patterns are matched against the normalized (no-accents, lowercase)
# geographic_notes string. Multiple patterns can add to the same subunidad.
# Order matters: more specific patterns should come first.
# ---------------------------------------------------------------------------
TRANSLATION_TABLE: list[tuple[str, list[str]]] = [

    # ── Cordillera de Guanacaste ──────────────────────────────────────────
    # "Cord. de Guanacaste" / "Cords. de Guanacaste" / "Cord. Guanacaste"
    (r"cord[s]?\. de guanacaste|cord[s]?\. guanacaste", [
        "8.1",   # Vertiente Pacífica Cord. Guanacaste
        "8.2",   # Vertiente Caribe Cord. Guanacaste
        "8.3",   # Cimas Cord. Guanacaste
    ]),

    # ── Cordillera de Tilarán ─────────────────────────────────────────────
    (r"cord[s]?\. de tilaran|cord[s]?\. tilaran", [
        "9.1",   # Vertiente Pacífica Cord. Tilarán
        "9.2",   # Vertiente Caribe Cord. Tilarán
        "9.3",   # Cimas Cord. Tilarán
    ]),

    # ── Cordillera Central ────────────────────────────────────────────────
    (r"cord[s]?\. central|cord\. cen-?\s*tral", [
        "10.1",  # Vertiente Pacífica Cord. Central
        "10.2",  # Vertiente Caribe Cord. Central
        "10.3",  # Cimas Cord. Central
        "10.4",  # Páramos Cord. Central
    ]),

    # ── Cordillera de Talamanca ───────────────────────────────────────────
    (r"cord[s]?\. de tala-?\s*manca|cord[s]?\. tala-?\s*manca|talamanca", [
        "11.1",  # Vertiente Pacífica Talamanca
        "11.2",  # Robledales Pacífica Talamanca
        "11.3",  # Vertiente Caribe Talamanca
        "11.4",  # Robledales Caribe Talamanca
        "11.5",  # Páramos Talamanca
    ]),

    # ── Vertiente Caribe — narrows down cordillera selections ─────────────
    # Applied as a FILTER on top of the above (see logic in matcher)
    (r"vert\. carib|vertiente carib", [
        "1.1", "1.2",   # Llanuras de Guatuso
        "2.1", "2.2",   # Llanuras de San Carlos
        "3.1", "3.2",   # Llanuras de Tortuguero
        "4.1", "4.2",   # Caribe Sur
        "8.2",          # Cord. Guanacaste Caribe
        "9.2",          # Cord. Tilarán Caribe
        "10.2",         # Cord. Central Caribe
        "11.3", "11.4", # Talamanca Caribe
    ]),

    # ── Vertiente Pacífica — narrows down cordillera selections ──────────
    (r"vert\. pac|vertiente pacif", [
        "8.1",          # Cord. Guanacaste Pacífico
        "9.1",          # Cord. Tilarán Pacífico
        "10.1",         # Cord. Central Pacífico
        "11.1", "11.2", # Talamanca Pacífico
        "12.1", "12.2", # Pacífico Central serranías
        "13.2",         # Litoral Pacífico Tárcoles-Térraba
        "14.1", "14.2", # Fila Costeña Norte y Sur
        "15.1",         # Valle del General
    ]),

    # ── Llanuras específicas ──────────────────────────────────────────────
    (r"llanura[s]? de tortu-?\s*guero|llanura[s]? de tortuguero", [
        "3.1", "3.2",
    ]),
    (r"llanura[s]? de san carlos|llanura[s]? de santa clara", [
        "2.1", "2.2",
    ]),
    (r"llanura[s]? de guatuso|llanura[s]? de los guatusos", [
        "1.1", "1.2",
    ]),
    (r"llanura[s]? de guanacaste|llanuras de guanacaste", [
        "6.1",
    ]),

    # ── Baja Talamanca / Caribe Sur ───────────────────────────────────────
    (r"baja talamanca|caribe sur", [
        "4.1", "4.2",
    ]),

    # ── Pacífico Central / serranías ──────────────────────────────────────
    (r"paci.fico central|p\.n\. carara|tarcoles|rio grande de tarcoles", [
        "12.1", "12.2",
        "13.2",
    ]),

    # ── Pacífico sur: "desde P.N. Carara al S." / "desde el Río Grande al S."
    # This phrase means everything south of Carara on the Pacific slope
    (r"desde p\.n\. carara al s|desde el rio grande.*al s|desde tarcoles al s", [
        "12.1", "12.2",  # Pacífico Central serranías
        "13.2",          # Litoral Tárcoles-Térraba
        "14.1", "14.2",  # Fila Costeña Norte y Sur
        "15.1",          # Valle del General
        "16.1",          # Coto Brus
        "17.1", "17.2", "17.3", "17.4",  # Osa / Golfo Dulce
    ]),

    # ── S vert. Pac. — southern Pacific slope (additive, not just a filter)
    # "S vert. Pac." without a specific place implies the whole southern Pacific
    (r"s vert\. pac(?!\. n)|s vert\. pacifico|sur.*vert\. pac", [
        "12.1", "12.2",  # Pacífico Central
        "13.2",          # Litoral Tárcoles-Térraba
        "14.1", "14.2",  # Fila Costeña
        "15.1",          # Valle del General
        "17.1", "17.2",  # Golfo Dulce / Térraba
    ]),

    # ── Fila Costeña ──────────────────────────────────────────────────────
    (r"fila coste.a|fila coste-?\s*na", [
        "14.1", "14.2",
    ]),
    (r"n fila coste|fila coste.a norte", ["14.1"]),
    (r"s fila coste|fila coste.a sur",  ["14.2"]),

    # ── Valle de El General ───────────────────────────────────────────────
    (r"valle de (el )?general|valle del general|v\. de general|valle.*general|general.*valle", [
        "15.1",
    ]),

    # ── Valle de Coto Brus ────────────────────────────────────────────────
    (r"coto brus", ["16.1"]),

    # ── Región Golfo Dulce / Osa / Burica / Golfito ───────────────────────
    (r"golfo dulce|region de golfito|regio.n de golfo|golfito", [
        "17.2", "17.3",  # Golfo Dulce + Osa elevadas
    ]),
    (r"pen[.i]+ de osa|peninsula de osa|pen\. osa", [
        "17.3", "17.4",
    ]),
    (r"terraba.sierpe|humedal terraba|terraba", [
        "17.1",
    ]),
    (r"punta burica|burica", [
        "17.3",
    ]),

    # ── Guanacaste (llanuras / Pacífico norte) ────────────────────────────
    (r"llanuras de guanacaste|n vert\. pac|vert\. pac\. n |vert\. pac\., n|pacifico norte", [
        "5.1",   # Santa Elena
        "6.1",   # Llanuras Guanacaste-Valle Central
        "6.2",   # Karst Tempisque-Nicoya
        "6.4",   # Herbáceo Tempisque
    ]),

    # ── Península de Nicoya ───────────────────────────────────────────────
    (r"pen[.i]+ de nicoya|peninsula de nicoya|pen\. nicoya", [
        "7.1", "7.2",
        "6.2", "6.4",
    ]),

    # ── Península de Santa Elena ──────────────────────────────────────────
    (r"santa elena|islas murciélago|islas murcielago", [
        "5.1",
    ]),

    # ── Valle Central ─────────────────────────────────────────────────────
    (r"valle central|cerros de escazu|cerros de la carpintera|tablazo", [
        "6.1",
        "10.1", "10.2",  # Cord. Central adjacent
    ]),

    # ── Cimas / Páramos ───────────────────────────────────────────────────
    (r"paramo|paramera|cimas", [
        "8.3", "9.3", "10.3", "10.4", "11.5",
    ]),

    # ── División Continental ──────────────────────────────────────────────
    # "cerca de la División Continental" implies both flanks of cordilleras
    (r"divisio.n continental|division continental", [
        "8.2", "8.3",   # Guanacaste ambas
        "9.2", "9.3",   # Tilarán ambas
        "10.2", "10.3", # Central ambas
        "11.3", "11.4", # Talamanca Caribe + cimas
    ]),

    # ── Isla del Coco ─────────────────────────────────────────────────────
    (r"isla del coco", ["18.1", "18.2"]),

    # ── Montes del Aguacate / Cerro Caraigres / Cerro Turrubares ─────────
    # These are within Pacífico Central region
    (r"aguacate|caraigres|turrubares", [
        "12.1", "12.2",
    ]),

    # ── Catch-all: toda la vert. Caribe ───────────────────────────────────
    (r"toda la vert\. carib|toda la vertiente carib", [
        "1.1", "1.2",
        "2.1", "2.2",
        "3.1", "3.2",
        "4.1", "4.2",
        "8.2", "9.2", "10.2",
        "11.3", "11.4",
    ]),

    # ── Catch-all: todas las cords. principales ───────────────────────────
    (r"todas las cords\. principales|todas las cord", [
        "8.1", "8.2",
        "9.1", "9.2",
        "10.1", "10.2",
        "11.1", "11.3",
    ]),
]

# Assign a consistent color per major geographic group (by SUBUNIDAD prefix)
GROUP_COLORS = {
    "1":  "#4fc3f7",  # Llanuras Guatuso — light blue
    "2":  "#29b6f6",  # Llanuras San Carlos — blue
    "3":  "#0288d1",  # Llanuras Tortuguero — dark blue
    "4":  "#0277bd",  # Caribe Sur — darker blue
    "5":  "#e65100",  # Santa Elena — burnt orange
    "6":  "#f57c00",  # Guanacaste llanuras — orange
    "7":  "#fb8c00",  # Nicoya — amber
    "8":  "#66bb6a",  # Cord. Guanacaste — green
    "9":  "#43a047",  # Cord. Tilarán — medium green
    "10": "#2e7d32",  # Cord. Central — dark green
    "11": "#1b5e20",  # Talamanca — deep green
    "12": "#7b1fa2",  # Pacífico Central — purple
    "13": "#9c27b0",  # Litoral Pacífico — violet
    "14": "#ab47bc",  # Fila Costeña — light purple
    "15": "#f48fb1",  # Valle General — pink
    "16": "#f06292",  # Coto Brus — rose
    "17": "#c2185b",  # Osa / Golfo Dulce — dark rose
    "18": "#90a4ae",  # Isla del Coco — gray-blue
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text))
    return nfkd.encode("ascii", "ignore").decode("ascii").lower()


def load_shapefile() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(SHP_PATH)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def match_subunidades(geographic_notes: str) -> set[str]:
    """
    Parse geographic_notes text and return a set of SUBUNIDAD codes.

    Two-phase approach:
      Phase 1 — collect all codes from PLACE patterns (specific locations,
                 cordilleras, llanuras, valleys, peninsulas, etc.)
      Phase 2 — apply vertiente filter: if only one vertiente is mentioned
                 (and not "ambas verts"), remove the opposite side's
                 cordillera sub-units (8.x-11.x) from the result.

    "vert. carib" and "vert. pac" are FILTERS only — they do not add
    new units on their own; they prune the set built by place patterns.
    The only exception: "toda la vert. carib" and catch-all patterns,
    which explicitly enumerate llanuras and are additive.
    """
    text = _normalize(geographic_notes)

    # ── Detect vertiente flags ────────────────────────────────────────────
    has_caribe   = bool(re.search(r"vert\. carib|vertiente carib", text))
    has_pacifico = bool(re.search(r"vert\. pac|vertiente pacif", text))
    ambas        = bool(re.search(r"ambas vert", text))
    toda_caribe  = bool(re.search(r"toda la vert\. carib|toda la vertiente carib", text))
    todas_cords  = bool(re.search(r"todas las cords\.|todas las cord", text))

    # ── Phase 1: collect matched codes from all PLACE patterns ───────────
    # Skip pure-vertiente entries (those are used only as filters in phase 2)
    VERTIENTE_ONLY_PATTERNS = {
        r"vert\. carib|vertiente carib",
        r"vert\. pac|vertiente pacif",
    }

    matched: set[str] = set()
    for pattern, subunidades in TRANSLATION_TABLE:
        if pattern in VERTIENTE_ONLY_PATTERNS:
            continue  # handled in phase 2 as filter
        if re.search(pattern, text):
            matched.update(subunidades)

    # ── Special catch-alls ────────────────────────────────────────────────
    if toda_caribe:
        matched.update(["1.1","1.2","2.1","2.2","3.1","3.2","4.1","4.2",
                         "8.2","9.2","10.2","11.3","11.4"])
    if todas_cords:
        matched.update(["8.1","8.2","9.1","9.2","10.1","10.2","11.1","11.3"])

    # If nothing matched at all, return empty (caller shows fallback)
    if not matched:
        return set()

    # ── Phase 2: vertiente filter on cordillera units (8.x – 11.x) ───────
    # Caribe-side suffixes: .2, .3, .4  (x.2 = Caribe, x.3 = cimas, x.4 = robledales Caribe)
    # Pacífico-side suffixes: .1, .2(Pac)  — for 11.x: .1=Pac, .2=Pac robledales
    # Note: for 11.x,  .1 and .2 are Pacífico; .3 and .4 are Caribe
    CARIBE_SIDE  = {"8": {".2",".3"}, "9": {".2",".3"}, "10": {".2",".3",".4"}, "11": {".3",".4",".5"}}
    PACIFICO_SIDE= {"8": {".1",".3"}, "9": {".1",".3"}, "10": {".1",".3",".4"}, "11": {".1",".2",".5"}}
    # Note: .3 (cimas) and .5 (paramo) are kept for both since peaks span both sides

    if not ambas:
        to_remove: set[str] = set()
        for sub in list(matched):
            parts = sub.split(".")
            prefix, suffix_num = parts[0], parts[1]
            if prefix not in CARIBE_SIDE:
                continue
            suffix = "." + suffix_num

            if has_caribe and not has_pacifico:
                # Keep only Caribe-side; remove pure-Pacífico suffixes
                if suffix not in CARIBE_SIDE[prefix]:
                    to_remove.add(sub)
            elif has_pacifico and not has_caribe:
                # Keep only Pacífico-side; remove pure-Caribe suffixes
                if suffix not in PACIFICO_SIDE[prefix]:
                    to_remove.add(sub)

        matched -= to_remove

    return matched


def get_group_color(subunidad: str) -> str:
    prefix = subunidad.split(".")[0]
    return GROUP_COLORS.get(prefix, "#78909c")


def generate_habitat_map(
    species_name: str,
    geographic_notes: str,
    elevation_min: float,
    elevation_max: float,
    presencias_gdf=None,
    output_path: str | Path = "outputs/habitat_map.png",
    dem_path: Path = DEM_PATH,
    shp_path: Path = SHP_PATH,
    elev_outlier_min: float | None = None,
    elev_outlier_max: float | None = None,
) -> Path:
    """
    Main function. Generates the three-layer habitat map and saves to output_path.

    Returns the output Path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    ufito = load_shapefile()
    matched_codes = match_subunidades(geographic_notes or "")
    fallback = len(matched_codes) == 0

    if fallback:
        print(f"  [WARN] No SUBUNIDAD match for notes: '{geographic_notes[:60]}' — showing all units")
        matched_mask = np.ones(len(ufito), dtype=bool)
    else:
        matched_mask = ufito["SUBUNIDAD"].apply(lambda s: str(s) in matched_codes)

    matched_gdf   = ufito[matched_mask].copy()
    unmatched_gdf = ufito[~matched_mask].copy()

    # ── Elevation raster ─────────────────────────────────────────────────
    has_elevation = (
        dem_path.exists()
        and elevation_min is not None
        and elevation_max is not None
        and not (np.isnan(float(elevation_min)) or np.isnan(float(elevation_max)))
    )

    elev_min = float(elevation_min) if has_elevation else 0
    elev_max = float(elevation_max) if has_elevation else 9999

    # For each matched unit, compute which pixels are in elevation range
    # We'll rasterize the DEM clipped per-unit and build two alpha layers:
    #   in_range_pixels  → deep color
    #   out_range_pixels → muted color (still within the polygon)

    # ── Fixed CR mainland extent (excludes Isla del Coco at ~5.5°N) ─────────
    # Hard-coded from known CR mainland boundaries + small padding
    XLIM = (-86.1, -82.4)
    YLIM = (7.9,   11.3)

    # Aspect ratio: ~1.2 lon per lat degree at CR latitude
    lon_span = XLIM[1] - XLIM[0]
    lat_span = YLIM[1] - YLIM[0]
    fig_w = 9
    fig_h = fig_w * (lat_span / lon_span) * 1.2

    # ── Figure setup ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    # Lock to full CR extent — never auto-zoom
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)

    # 1. Unmatched units — gray
    if not unmatched_gdf.empty:
        unmatched_gdf.plot(
            ax=ax,
            color="#2e3440",
            edgecolor="#4c566a",
            linewidth=0.4,
            alpha=0.7,
            zorder=1,
        )

    # 2. Matched units — muted color (full polygon, will be overlaid by elevation)
    for _, row in matched_gdf.iterrows():
        sub = str(row["SUBUNIDAD"])
        color = get_group_color(sub)
        muted = _mute_color(color, saturation=0.55, lightness=0.55)
        gpd.GeoDataFrame([row], crs=ufito.crs).plot(
            ax=ax,
            color=muted,
            edgecolor=_mute_color(color, saturation=0.7, lightness=0.7),
            linewidth=0.6,
            alpha=0.90,
            zorder=2,
        )

    # 3. Elevation highlight within matched units
    if has_elevation and not matched_gdf.empty:
        _overlay_elevation(
            ax=ax,
            matched_gdf=matched_gdf,
            dem_path=dem_path,
            elev_min=elev_min,
            elev_max=elev_max,
        )

    # 3b. Outlier elevation band (hatched, semi-transparent)
    has_outlier = (
        has_elevation
        and not matched_gdf.empty
        and (elev_outlier_min is not None or elev_outlier_max is not None)
        and not (
            (elev_outlier_min is not None and np.isnan(float(elev_outlier_min))) or
            (elev_outlier_max is not None and np.isnan(float(elev_outlier_max)))
        )
    )
    if has_outlier:
        out_lo = float(elev_outlier_min) if elev_outlier_min is not None else elev_min
        out_hi = float(elev_outlier_max) if elev_outlier_max is not None else elev_max
        _overlay_elevation_outlier(
            ax=ax,
            matched_gdf=matched_gdf,
            dem_path=dem_path,
            elev_min=out_lo,
            elev_max=out_hi,
            main_min=elev_min,
            main_max=elev_max,
        )

    # 4. GBIF presence points
    if presencias_gdf is not None and not presencias_gdf.empty:
        ax.scatter(
            presencias_gdf.geometry.x,
            presencias_gdf.geometry.y,
            s=12,
            c="#ff4444",
            edgecolors="#ffaaaa",
            linewidths=0.4,
            alpha=0.85,
            zorder=10,
            label=f"Presencias GBIF (n={len(presencias_gdf)})",
        )

    # ── Decorations ───────────────────────────────────────────────────────
    sp_display = species_name.replace("_", " ")
    elev_txt = f"{int(elev_min)}–{int(elev_max)} m" if has_elevation else "elevación no disponible"
    if has_outlier:
        out_lo = float(elev_outlier_min) if elev_outlier_min is not None else elev_min
        out_hi = float(elev_outlier_max) if elev_outlier_max is not None else elev_max
        outlier_parts = []
        if elev_outlier_min is not None:
            outlier_parts.append(f"inf: {int(out_lo)}–{int(elev_min)} m")
        if elev_outlier_max is not None:
            outlier_parts.append(f"sup: {int(elev_max)}–{int(out_hi)} m")
        elev_txt += f"  (atípicos — {', '.join(outlier_parts)})"

    ax.set_title(
        f"{sp_display}",
        color="white", fontsize=13, fontweight="bold", pad=4,
    )
    ax.set_xlabel(
        f"Hábitat potencial — Manual de Plantas CR  |  Elevación: {elev_txt}",
        color="#9ca3af", fontsize=8,
    )

    # Notes box
    note = (geographic_notes[:100] + "…") if geographic_notes and len(geographic_notes) > 100 else (geographic_notes or "")
    if note:
        ax.text(
            0.01, 0.01, f"Notas: {note}",
            transform=ax.transAxes, fontsize=6,
            color="#9ca3af", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1f2937", alpha=0.8, linewidth=0),
        )

    # Legend
    legend_patches = []
    if not fallback:
        legend_patches.append(mpatches.Patch(color="#4b5563", label="Fuera del rango geográfico"))
        legend_patches.append(mpatches.Patch(color="#6b7280", label="Zona geográfica — fuera del rango altitudinal"))
        if has_elevation:
            main_elev_txt = f"{int(elev_min)}–{int(elev_max)} m"
            legend_patches.append(mpatches.Patch(color="#22d3ee", label=f"Hábitat óptimo ({main_elev_txt})"))
        if has_outlier:
            out_lo = float(elev_outlier_min) if elev_outlier_min is not None else elev_min
            out_hi = float(elev_outlier_max) if elev_outlier_max is not None else elev_max
            if elev_outlier_min is not None:
                legend_patches.append(mpatches.Patch(
                    facecolor="#22d3ee", alpha=0.35,
                    edgecolor="#22d3ee", linewidth=0.8,
                    label=f"Reg. atípicos inf. ({int(out_lo)}–{int(elev_min)} m)",
                    hatch="////",
                ))
            if elev_outlier_max is not None:
                legend_patches.append(mpatches.Patch(
                    facecolor="#22d3ee", alpha=0.35,
                    edgecolor="#22d3ee", linewidth=0.8,
                    label=f"Reg. atípicos sup. ({int(elev_max)}–{int(out_hi)} m)",
                    hatch="////",
                ))
    if presencias_gdf is not None and not presencias_gdf.empty:
        legend_patches.append(mpatches.Patch(color="#ff4444", label=f"Presencias GBIF (n={len(presencias_gdf)})"))

    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc="lower right", fontsize=7.5,
            facecolor="#1f2937", labelcolor="white",
            framealpha=0.9, edgecolor="#374151",
        )

    ax.tick_params(colors="#6b7280", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    plt.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Habitat map saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Elevation overlay helper
# ---------------------------------------------------------------------------

def _overlay_elevation(ax, matched_gdf, dem_path, elev_min, elev_max):
    """
    For each matched polygon, reads the DEM, creates two RGBA arrays:
      - in-range pixels  → bright cyan with alpha
      - (out-of-range pixels already covered by muted polygon below)
    Renders them via imshow with exact extent.
    """
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_transform = src.transform
        dem_nodata = src.nodata if src.nodata is not None else -9999

        # Reproject shapefile to DEM CRS if needed
        if matched_gdf.crs.to_epsg() != int(str(dem_crs).split(":")[-1]):
            matched_proj = matched_gdf.to_crs(dem_crs)
        else:
            matched_proj = matched_gdf

        # Process all matched units together for efficiency
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

        # Build RGBA image: in-range → cyan, else transparent
        h, w = elev.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)

        in_range = (~np.isnan(elev)) & (elev >= elev_min) & (elev <= elev_max)
        rgba[in_range]  = [0.13, 0.85, 0.93, 0.82]   # cyan #22d9ec, alpha=0.82
        # out_range within polygon: already covered by muted polygon → leave transparent

        if not in_range.any():
            print(f"  [WARN] No DEM pixels found in elevation range [{elev_min}-{elev_max}]")
            return

        # Compute geographic extent for imshow
        left   = out_transform.c
        top    = out_transform.f
        right  = left + out_transform.a * w
        bottom = top  + out_transform.e * h

        ax.imshow(
            rgba,
            extent=[left, right, bottom, top],
            origin="upper",
            aspect="auto",
            zorder=5,
            interpolation="nearest",
        )


def _overlay_elevation_outlier(ax, matched_gdf, dem_path, elev_min, elev_max,
                                main_min, main_max):
    """
    Renders a hatched semi-transparent cyan layer for outlier elevation pixels
    that fall outside the main range [main_min, main_max] but within [elev_min, elev_max].
    Skips pixels already covered by the main highlight.
    """
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
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

        # Pixels in outlier range but NOT in the main range
        in_outlier = (
            (~np.isnan(elev)) &
            (elev >= elev_min) & (elev <= elev_max) &
            ~((elev >= main_min) & (elev <= main_max))
        )
        rgba[in_outlier] = [0.13, 0.85, 0.93, 0.38]  # cyan at 38% alpha

        if not in_outlier.any():
            return

        left   = out_transform.c
        top    = out_transform.f
        right  = left + out_transform.a * w
        bottom = top  + out_transform.e * h

        # Base translucent fill
        ax.imshow(
            rgba,
            extent=[left, right, bottom, top],
            origin="upper",
            aspect="auto",
            zorder=4,
            interpolation="nearest",
        )

        # Hatching: draw the outlier union polygon with hatch pattern on top
        try:
            from shapely.ops import unary_union
            import geopandas as gpd_local

            # Reproject back to WGS84 for plotting
            if matched_gdf.crs.to_epsg() != 4326:
                plot_gdf = matched_gdf.to_crs(epsg=4326)
            else:
                plot_gdf = matched_gdf

            union_geom = unary_union(plot_gdf.geometry)
            union_gdf = gpd.GeoDataFrame(geometry=[union_geom], crs="EPSG:4326")
            union_gdf.plot(
                ax=ax,
                facecolor="none",
                edgecolor="#22d3ee",
                linewidth=0.0,
                hatch="////",
                alpha=0.30,
                zorder=6,
            )
        except Exception as e:
            print(f"  [WARN] Outlier hatch polygon failed: {e}")


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _mute_color(hex_color: str, saturation: float = 0.3, lightness: float = 0.4) -> str:
    """Return a desaturated/darkened version of a hex color."""
    import colorsys
    r, g, b = mcolors.to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s * saturation, v * lightness)
    return mcolors.to_hex((r2, g2, b2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate species habitat map from Manual + DEM + Ufito.")
    parser.add_argument("--species",   required=True,  help="Species name")
    parser.add_argument("--notes",     required=True,  help="geographic_notes text from Manual")
    parser.add_argument("--elev-min",  type=float, default=None)
    parser.add_argument("--elev-max",  type=float, default=None)
    parser.add_argument("--output",    default="outputs/habitat_map.png")
    args = parser.parse_args()

    generate_habitat_map(
        species_name=args.species,
        geographic_notes=args.notes,
        elevation_min=args.elev_min,
        elevation_max=args.elev_max,
        output_path=args.output,
    )
