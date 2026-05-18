"""
N01: LoadGeometry
Carga los límites geográficos de Costa Rica y Mesoamérica.
"""
import os
import json
from graph_state import GraphState
from data.expert_maps import ExpertMapLoader
import config


def load_geometry_node(state: GraphState) -> GraphState:
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

    # Salida JSON
    output = {
        "node": "N01_LoadGeometry",
        "cr_boundary_crs": str(cr_bounds.crs) if cr_bounds is not None else None,
        "cr_boundary_rows": len(cr_bounds) if cr_bounds is not None else 0,
        "meso_boundary_crs": str(meso_bounds.crs) if meso_bounds is not None else None,
        "meso_boundary_rows": len(meso_bounds) if meso_bounds is not None else 0,
        "status": "ok"
    }
    _save_json(output, state['output_dir'], "n01_load_geometry.json")

    print("[N01:✓] Límites cargados correctamente")
    return state


def _save_json(data: dict, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
