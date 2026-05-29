"""
N12: GenerateConservationContext
Genera contexto de conservación cruzando las presencias de CR
con capas vectoriales de áreas protegidas, humedales y corredores biológicos.

INPUT  (state):
    - presencias_cr : GeoDataFrame con ocurrencias en Costa Rica
    - output_dir    : directorio de salida de la ejecución

OUTPUT (state):
    - contexto_conservacion : dict con resultados del cruce espacial
                              por cada capa vectorial analizada

ARCHIVO: n12_conservation_context.json
"""
import os
from graph_state import GraphState
from utils.geoprocesamiento import generar_contexto_conservacion
from node_utils import save_node_json

RUTAS_VECTORES = {
    "Áreas Silvestres Protegidas":        os.path.join("data_raw", "vectors", "areas_protegidas_cr.shp"),
    "Humedales":                           os.path.join("data_raw", "vectors", "humedales_cr.shp"),
    "Corredores Biológicos":               os.path.join("data_raw", "vectors", "corredores_biologicos_cr.shp"),
    "Áreas de Conservación (Macro-regiones)": os.path.join("data_raw", "vectors", "areas_conservacion_cr.shp"),
}


def generate_conservation_context_node(state: GraphState) -> GraphState:
    """
    Nodo N12: cruza presencias de CR con capas vectoriales de conservación.

    Parámetros:
        state (GraphState): estado del pipeline.

    Retorna:
        GraphState: estado actualizado con contexto_conservacion.
    """
    print("\n[N12:GenerateConservationContext] Analizando contexto de conservación...")
    presencias_cr = state['presencias_cr']
    output_dir    = state['output_dir']

    contexto_conservacion = generar_contexto_conservacion(presencias_cr, RUTAS_VECTORES)
    state['contexto_conservacion'] = contexto_conservacion

    try:
        contexto_serializable = _make_serializable(contexto_conservacion)
    except Exception:
        contexto_serializable = str(contexto_conservacion)

    save_node_json({
        "node": "N12_GenerateConservationContext",
        "capas_analizadas": list(RUTAS_VECTORES.keys()),
        "contexto": contexto_serializable,
        "status": "ok"
    }, output_dir, "n12_conservation_context.json")

    print("[N12:✓] Contexto de conservación generado")
    return state


def _make_serializable(obj):
    """
    Convierte recursivamente objetos a tipos JSON-serializables.

    Parámetros:
        obj: cualquier objeto Python (dict, list, DataFrame, Series, etc.)

    Retorna:
        Versión serializable del objeto; fallback a str() si no se reconoce.
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    try:
        import pandas as pd
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
    except ImportError:
        pass
    return str(obj)