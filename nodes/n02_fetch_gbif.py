"""
N02: FetchGBIF
Extrae ocurrencias de GBIF para la especie objetivo en Mesoamérica y CR.

INPUT  (state):
    - species_name  : nombre científico de la especie
    - meso_boundary : GeoDataFrame con el límite de Mesoamérica
    - cr_boundary   : GeoDataFrame con el límite de Costa Rica

OUTPUT (state):
    - presencias_meso  : GeoDataFrame con ocurrencias limpias en Mesoamérica
    - presencias_cr    : GeoDataFrame con ocurrencias limpias en CR
                         (fallback a presencias_meso si CR queda vacío)
    - should_continue  : False si no hay registros en Mesoamérica
    - error_messages   : append en caso de error o fallback

ARCHIVO: n02_fetch_gbif.json
"""
import os
from graph_state import GraphState
from data.gbif_extractor import GBIFExtractor
from node_utils import save_node_json


def fetch_gbif_node(state: GraphState) -> GraphState:
    """
    Nodo N02: descarga y limpia ocurrencias GBIF para la especie.

    Parámetros:
        state (GraphState): estado del pipeline.

    Retorna:
        GraphState: estado actualizado con presencias_meso y presencias_cr.
                    Detiene el pipeline si no hay registros en Mesoamérica.
    """
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

    if presencias_meso is None or presencias_meso.empty:
        print(f"[N02:ERROR] No hay registros para {species_name} en Mesoamérica.")
        state['error_messages'].append("Sin registros GBIF en Mesoamérica")
        state['should_continue'] = False
        save_node_json({
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

    save_node_json({
        "node": "N02_FetchGBIF",
        "species_name": species_name,
        "presencias_meso": len(presencias_meso),
        "presencias_cr": len(presencias_cr),
        "columnas": list(presencias_meso.columns),
        "status": "ok"
    }, state['output_dir'], "n02_fetch_gbif.json")

    print(f"[N02:✓] Mesoamérica: {len(presencias_meso)} | CR: {len(presencias_cr)}")
    return state