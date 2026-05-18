"""
N02: FetchGBIF
Extrae ocurrencias de GBIF para la especie objetivo.
"""
import os
import json
from graph_state import GraphState
from data.gbif_extractor import GBIFExtractor


def fetch_gbif_node(state: GraphState) -> GraphState:
    print(f"\n[N02:FetchGBIF] Descargando ocurrencias para {state['species_name']}...")

    extractor = GBIFExtractor()
    species_name = state['species_name']
    meso_bounds = state['meso_boundary']
    cr_bounds = state['cr_boundary']

    presencias_meso = extractor.fetch_occurrences_mesoamerica(species_name)
    presencias_meso = extractor.clean_spatial_outliers(presencias_meso, meso_bounds)

    presencias_cr = None
    if presencias_meso is not None and not presencias_meso.empty:
        presencias_cr = extractor.clean_spatial_outliers(presencias_meso, cr_bounds)

    # Validación crítica
    if presencias_meso is None or presencias_meso.empty:
        print(f"[N02:ERROR] No hay registros para {species_name} en Mesoamérica.")
        state['error_messages'].append("Sin registros GBIF en Mesoamérica")
        state['should_continue'] = False
        _save_json({
            "node": "N02_FetchGBIF",
            "species_name": species_name,
            "presencias_meso": 0,
            "presencias_cr": 0,
            "status": "error",
            "error": "Sin registros GBIF en Mesoamérica"
        }, state['output_dir'], "n02_fetch_gbif.json")
        return state

    if presencias_cr is None or presencias_cr.empty:
        print(f"[N02:WARN] No hay registros en CR. Usando fallback Mesoamérica.")
        presencias_cr = presencias_meso
        state['error_messages'].append("Sin registros GBIF específicos de CR")

    state['presencias_meso'] = presencias_meso
    state['presencias_cr'] = presencias_cr

    # Salida JSON
    _save_json({
        "node": "N02_FetchGBIF",
        "species_name": species_name,
        "presencias_meso": len(presencias_meso),
        "presencias_cr": len(presencias_cr),
        "columnas": list(presencias_meso.columns),
        "status": "ok"
    }, state['output_dir'], "n02_fetch_gbif.json")

    print(f"[N02:✓] Mesoamérica: {len(presencias_meso)} | CR: {len(presencias_cr)}")
    return state


def _save_json(data: dict, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
