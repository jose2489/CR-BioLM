"""
N10: ExplainLIME
Genera explicaciones locales por instancia usando LIME
y exporta gráfico de explicación.

INPUT  (state):
    - rf_model   : instancia entrenada de RandomForestSDM
    - X_train    : DataFrame con features de entrenamiento (contexto LIME)
    - X_test     : DataFrame con features de prueba
    - output_dir : directorio de salida de la ejecución

OUTPUT (state):
    - lime_data      : dict con resultados de la explicación LIME.
                       Dict vacío si LIME falla.
    - error_messages : append si LIME no está disponible

ARCHIVO: n10_explain_lime.json
"""
import os
from graph_state import GraphState
from xai.lime_explainer import LIMEExplainer
from node_utils import save_node_json


def explain_lime_node(state: GraphState) -> GraphState:
    """
    Nodo N10: genera explicaciones locales LIME para instancias del set de prueba.

    Parámetros:
        state (GraphState): estado del pipeline.

    Retorna:
        GraphState: estado actualizado con lime_data.
                    Si LIME falla, lime_data queda como dict vacío
                    y el pipeline continúa.
    """
    print("\n[N10:ExplainLIME] Generando explicaciones locales (LIME)...")
    rf_model   = state['rf_model']
    X_train    = state['X_train']
    X_test     = state['X_test']
    output_dir = state['output_dir']

    try:
        lime_explainer = LIMEExplainer()
        lime_data = lime_explainer.explain_and_plot(rf_model, X_train, X_test, output_dir)
        state['lime_data'] = lime_data

        save_node_json({
            "node": "N10_ExplainLIME",
            "plot_path": os.path.join(output_dir, "lime_local_explanation.png"),
            "status": "ok"
        }, output_dir, "n10_explain_lime.json")

        print(f"[N10:✓] LIME generado y guardado en {output_dir}")

    except Exception as e:
        print(f"[N10:WARN] LIME falló: {e}")
        state['error_messages'].append(f"LIME no disponible: {e}")
        state['lime_data'] = {}
        save_node_json({
            "node": "N10_ExplainLIME",
            "status": "error",
            "error": str(e)
        }, output_dir, "n10_explain_lime.json")

    return state