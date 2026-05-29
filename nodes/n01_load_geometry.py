"""
N01: LoadGeometry
Carga los límites geográficos de Costa Rica y Mesoamérica.

INPUT  (state):
    - (ninguno requerido del estado previo)

OUTPUT (state):
    - cr_boundary   : GeoDataFrame con el límite de Costa Rica
    - meso_boundary : GeoDataFrame con el límite de Mesoamérica
                      (fallback a cr_boundary si no está disponible)
    - error_messages: append si Mesoamérica no se pudo cargar

ARCHIVO: n01_load_geometry.json
"""
import os
from graph_state import GraphState
from data.expert_maps import ExpertMapLoader
from node_utils import save_node_json
import config


def load_geometry_node(state: GraphState) -> GraphState:
    """
    Nodo N01: carga los límites geográficos de CR y Mesoamérica.

    Parámetros:
        state (GraphState): estado del pipeline.

    Retorna:
        GraphState: estado actualizado con cr_boundary y meso_boundary.
    """
    print("\n[N01:LoadGeometry] Cargando límites geográficos...")
    map_loader = ExpertMapLoader()
    cr_bounds = map_loader.load_country_boundary(config.DEFAULT_COUNTRY)
    meso_bounds = map_loader.load_mesoamerica_boundary()

    if meso_bounds is None:
        print("[N01:WARN] No se pudo cargar Mesoamérica. Usando solo CR.")
        meso_bounds = cr_bounds
        state['error_messages'].append("Límite Mesoamericano no disponible")

    state['cr_boundary'] = cr_bounds
    state['meso_boundary'] = meso_bounds

    save_node_json({
        "node": "N01_LoadGeometry",
        "cr_boundary_crs": str(cr_bounds.crs) if cr_bounds is not None else None,
        "cr_boundary_rows": len(cr_bounds) if cr_bounds is not None else 0,
        "meso_boundary_crs": str(meso_bounds.crs) if meso_bounds is not None else None,
        "meso_boundary_rows": len(meso_bounds) if meso_bounds is not None else 0,
        "status": "ok"
    }, state['output_dir'], "n01_load_geometry.json")

    print("[N01:✓] Límites cargados correctamente")
    return state