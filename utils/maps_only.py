"""
maps_only.py
------------
Minimal local-only pipeline: GBIF → altitude enrichment → habitat map.
No climate rasters, no RF, no SHAP/LIME, no LLM calls.

Entry point: run_maps_only(especie_nombre)
CLI:         python main.py maps-only -s "Sobralia amabilis"
             python main.py maps-only -f species.txt
"""

import os
import pandas as pd
from datetime import datetime

import config
from data.gbif_extractor import GBIFExtractor
from data.expert_maps import ExpertMapLoader
from utils.geoprocesamiento import extraer_altitud
from utils.maps_checklist import Checklist
from utils.distribution_map import build_ficha, generate_distribution_map


_CATALOG_PATH = os.path.join("outputs", "picked_species_enhanced_clean.csv")


def _crear_directorio_maps_only(especie_nombre: str) -> str:
    """Create outputs/{especie}/maps_only_YYYYMMDD_HHMMSS/ and return the path."""
    especie_fmt = especie_nombre.replace(" ", "_")
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(config.OUTPUT_BASE_DIR, especie_fmt, f"maps_only_{timestamp}")
    os.makedirs(ruta, exist_ok=True)
    return ruta


def run_maps_only(especie_nombre: str) -> bool:
    """
    Minimal pipeline for one species.

    Steps:
      1. Create output directory
      2. Load CR boundary
      3. Fetch GBIF occurrences (Mesoamerica) + clip to CR
      4. Enrich with altitude from DEM
      5. Load Manual de Plantas catalog row
      6. Generate habitat map (V2)

    Returns True on success, False on fatal error.
    """
    cl = Checklist(f"{especie_nombre} — maps-only")

    # 1. Output directory
    out_dir = _crear_directorio_maps_only(especie_nombre)
    cl.check("Directorio de salida creado", ok=True, detail=out_dir)

    # 2. CR boundary
    try:
        map_loader  = ExpertMapLoader()
        cr_bounds   = map_loader.load_country_boundary(config.DEFAULT_COUNTRY)
        meso_bounds = map_loader.load_mesoamerica_boundary()
        if meso_bounds is None or meso_bounds.empty:
            meso_bounds = cr_bounds
        cl.check("Límite CR cargado", ok=True)
    except Exception as e:
        cl.check("Límite CR cargado", ok=False, detail=str(e))
        cl.done()
        return False

    # 3. GBIF fetch + CR clip
    try:
        extractor      = GBIFExtractor()
        presencias_meso = extractor.fetch_occurrences_mesoamerica(especie_nombre)
        presencias_meso = extractor.clean_spatial_outliers(presencias_meso, meso_bounds)

        if presencias_meso is None or presencias_meso.empty:
            cl.check("GBIF fetched", ok=False, detail="sin registros en Mesoamérica")
            cl.done()
            return False

        presencias_cr = extractor.clean_spatial_outliers(presencias_meso, cr_bounds)
        if presencias_cr is None or presencias_cr.empty:
            presencias_cr = presencias_meso  # fallback
        cl.check("GBIF fetched",
                 ok=True,
                 detail=f"Meso n={len(presencias_meso)}, CR n={len(presencias_cr)}")
    except Exception as e:
        cl.check("GBIF fetched", ok=False, detail=str(e))
        cl.done()
        return False

    # 4. Altitude enrichment
    try:
        ruta_altitud  = config.DEM_PATH
        presencias_cr = extraer_altitud(presencias_cr, ruta_altitud)
        cl.check("Altitud enriquecida", ok=True, detail="DEM EPSG:4326")
    except Exception as e:
        cl.check("Altitud enriquecida", ok=False, detail=str(e))
        # Non-fatal — continue without altitude column

    # 5. Manual catalog row + build Ficha
    geographic_notes = None
    elevation_min    = None
    elevation_max    = None
    elev_outlier_min = None
    elev_outlier_max = None

    try:
        catalog = pd.read_csv(_CATALOG_PATH)
        row_df  = catalog[catalog["species"] == especie_nombre]
        if not row_df.empty:
            r = row_df.iloc[0]
            _geo             = r["geographic_notes"]
            geographic_notes = str(_geo) if (not pd.isna(_geo) if not isinstance(_geo, str) else True) else None
            _raw             = r.get("habitat_raw", "")
            habitat_raw      = str(_raw) if pd.notna(_raw) else ""
            elevation_min    = r["elevation_min_m"]
            elevation_max    = r["elevation_max_m"]
            _out_min         = r["elev_outlier_min_m"]
            _out_max         = r["elev_outlier_max_m"]
            elev_outlier_min = None if (not _out_min or str(_out_min) == "nan") else float(_out_min)
            elev_outlier_max = None if (not _out_max or str(_out_max) == "nan") else float(_out_max)
            cl.check("Catálogo Manual cargado", ok=True,
                     detail=f"elev {elevation_min}–{elevation_max} m")

            # Build and save Ficha JSON sidecar
            try:
                ficha = build_ficha(
                    habitat_raw      = habitat_raw,
                    geographic_notes = geographic_notes or "",
                    species          = especie_nombre,
                )
                # Supplement elevation from catalog if parser didn't extract it
                if not ficha.elevation.has_data() and elevation_min is not None and elevation_max is not None:
                    import math
                    if not (math.isnan(float(elevation_min)) or math.isnan(float(elevation_max))):
                        from utils.distribution_map.ficha import ElevationRange
                        ficha = ficha.__class__(
                            **{**ficha.to_dict(),
                               "elevation": ElevationRange(
                                   min_m=float(elevation_min),
                                   max_m=float(elevation_max),
                                   outlier_min_m=elev_outlier_min,
                                   outlier_max_m=elev_outlier_max,
                               )}
                        )
                ficha_path = os.path.join(out_dir, "ficha.json")
                ficha.save(ficha_path)
                cl.check("Ficha estructurada", ok=True, detail=ficha.summary())
            except Exception as fe:
                cl.check("Ficha estructurada", ok=False, detail=str(fe))
                ficha = None

        else:
            cl.check("Catálogo Manual cargado", ok=False,
                     detail=f"'{especie_nombre}' no en catálogo")
            ficha = None
    except Exception as e:
        cl.check("Catálogo Manual cargado", ok=False, detail=str(e))
        ficha = None

    # 6. Distribution map (Ficha-driven renderer)
    try:
        if ficha is None:
            from utils.distribution_map import build_ficha as _bf
            ficha = _bf(habitat_raw="", geographic_notes="", species=especie_nombre)
        map_path = generate_distribution_map(
            ficha=ficha,
            output_path=os.path.join(out_dir, "mapa_habitat_manual.png"),
            presencias_gdf=presencias_cr,
        )
        cl.check("Mapa de hábitat renderizado", ok=True, detail=str(map_path))
    except Exception as e:
        cl.check("Mapa de hábitat renderizado", ok=False, detail=str(e))
        cl.done()
        return False

    cl.done()
    return True
