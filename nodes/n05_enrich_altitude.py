"""
N05: EnrichAltitude
Agrega columna de altitud a las presencias de Costa Rica
extrayendo valores desde un DEM raster.

INPUT  (state):
    - presencias_cr : GeoDataFrame con ocurrencias en CR

OUTPUT (state):
    - presencias_cr  : GeoDataFrame enriquecido con columna de altitud
    - info_altitud   : string descriptivo del rango altitudinal detectado
    - error_messages : append si las estadísticas de altitud fallan

ARCHIVO: n05_enrich_altitude.json
"""
import os
from graph_state import GraphState
from utils.geoprocesamiento import extraer_altitud
from node_utils import save_node_json

DEM_PATH = os.path.join("data_raw", "topography", "altitud_cr.tif")


def enrich_altitude_node(state: GraphState) -> GraphState:
    """
    Nodo N05: enriquece presencias_cr con datos de altitud desde un DEM.

    Parámetros:
        state (GraphState): estado del pipeline.

    Retorna:
        GraphState: estado actualizado con presencias_cr (con altitud)
                    e info_altitud con el rango detectado.
    """
    print("\n[N05:EnrichAltitude] Enriqueciendo presencias con altitud...")
    presencias_cr = state['presencias_cr']
    presencias_cr = extraer_altitud(presencias_cr, DEM_PATH)

    info_altitud = "No disponible"
    alt_min = alt_max = alt_med = None

    try:
        col_altitud = [c for c in presencias_cr.columns if 'alt' in c.lower()]
        if col_altitud:
            alt_series = presencias_cr[col_altitud[0]].dropna()
            if not alt_series.empty:
                alt_min = float(alt_series.min())
                alt_max = float(alt_series.max())
                alt_med = float(alt_series.mean())
                info_altitud = f"{alt_min:.0f} - {alt_max:.0f} msnm (Promedio: {alt_med:.0f} m)"
        print(f"[N05:✓] Altitud detectada: {info_altitud}")
    except Exception as e:
        print(f"[N05:WARN] Error al calcular estadísticas de altitud: {e}")
        state['error_messages'].append(f"Estadísticas de altitud no disponibles: {e}")

    state['presencias_cr'] = presencias_cr
    state['info_altitud'] = info_altitud

    save_node_json({
        "node": "N05_EnrichAltitude",
        "altitud_min_msnm": alt_min,
        "altitud_max_msnm": alt_max,
        "altitud_promedio_msnm": alt_med,
        "info_altitud": info_altitud,
        "dem_usado": DEM_PATH,
        "status": "ok" if info_altitud != "No disponible" else "warn"
    }, state['output_dir'], "n05_enrich_altitude.json")

    return state