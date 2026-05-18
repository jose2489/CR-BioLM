"""
N12: GenerateConservationContext
Genera contexto de conservación cruzando presencias con capas vectoriales de CR.
"""
import os
import json
from graph_state import GraphState
from utils.geoprocesamiento import generar_contexto_conservacion

RUTAS_VECTORES = {
    "Áreas Silvestres Protegidas": os.path.join("data_raw", "vectors", "areas_protegidas_cr.shp"),
    "Humedales": os.path.join("data_raw", "vectors", "humedales_cr.shp"),
    "Corredores Biológicos": os.path.join("data_raw", "vectors", "corredores_biologicos_cr.shp"),
    "Áreas de Conservación (Macro-regiones)": os.path.join("data_raw", "vectors", "areas_conservacion_cr.shp")
}


def generate_conservation_context_node(state: GraphState) -> GraphState:
    print("\n[N12:GenerateConservationContext] Analizando contexto de conservación...")

    presencias_cr = state['presencias_cr']
    output_dir = state['output_dir']

    contexto_conservacion = generar_contexto_conservacion(presencias_cr, RUTAS_VECTORES)
    state['contexto_conservacion'] = contexto_conservacion

    # Serializar contexto (puede contener tipos no serializables)
    try:
        contexto_serializable = _make_serializable(contexto_conservacion)
    except Exception:
        contexto_serializable = str(contexto_conservacion)

    _save_json({
        "node": "N12_GenerateConservationContext",
        "capas_analizadas": list(RUTAS_VECTORES.keys()),
        "contexto": contexto_serializable,
        "status": "ok"
    }, output_dir, "n12_conservation_context.json")

    print(f"[N12:✓] Contexto de conservación generado")
    return state


def _make_serializable(obj):
    """Convierte recursivamente objetos a tipos JSON-serializables."""
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


def _save_json(data: dict, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
