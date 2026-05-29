"""
N03: LoadClimate
Carga capas climáticas WorldClim recortadas a Mesoamérica,
filtrando solo las variables expertas definidas.

INPUT  (state):
    - meso_boundary : GeoDataFrame con el límite de Mesoamérica

OUTPUT (state):
    - climate_rasters : dict {nombre_variable: ruta_raster} filtrado
                        por VARIABLES_EXPERTAS
    - should_continue : False si no se encontraron rasters
    - error_messages  : append si falla la carga

ARCHIVO: n03_load_climate.json
"""
import os
from graph_state import GraphState
from data.climate_loader import ClimateLoader
from node_utils import save_node_json

VARIABLES_EXPERTAS = ["bio_14", "bio_15", "bio_16", "bio_17", "bio_18", "bio_19"]


def load_climate_node(state: GraphState) -> GraphState:
    """
    Nodo N03: carga y filtra capas climáticas WorldClim para Mesoamérica.

    Parámetros:
        state (GraphState): estado del pipeline.

    Retorna:
        GraphState: estado actualizado con climate_rasters.
                    Detiene el pipeline si no se encuentran rasters.
    """
    print("\n[N03:LoadClimate] Cargando capas climáticas...")
    meso_bounds = state['meso_boundary']
    climate_loader = ClimateLoader()
    raster_paths = climate_loader.get_climate_layers(meso_bounds, region_name='meso')

    if not raster_paths:
        print("[N03:ERROR] Fallo al cargar rasters climáticos.")
        state['error_messages'].append("Fallo en carga de capas climáticas")
        state['should_continue'] = False
        save_node_json({
            "node": "N03_LoadClimate",
            "variables_cargadas": 0,
            "status": "error",
            "error": "No se encontraron rasters climáticos"
        }, state['output_dir'], "n03_load_climate.json")
        return state

    raster_paths_filtrado = {
        k: v for k, v in raster_paths.items()
        if any(var in k or var in v for var in VARIABLES_EXPERTAS)
    }

    state['climate_rasters'] = raster_paths_filtrado

    save_node_json({
        "node": "N03_LoadClimate",
        "variables_expertas": VARIABLES_EXPERTAS,
        "variables_cargadas": len(raster_paths_filtrado),
        "variables": list(raster_paths_filtrado.keys()),
        "status": "ok"
    }, state['output_dir'], "n03_load_climate.json")

    print(f"[N03:✓] {len(raster_paths_filtrado)} variables climáticas cargadas")
    return state