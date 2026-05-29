"""
N09: ExplainSHAP
Calcula importancia global de features usando valores SHAP
y genera gráfico de resumen.

INPUT  (state):
    - rf_model   : instancia entrenada de RandomForestSDM
    - X_test     : DataFrame con features de prueba
    - output_dir : directorio de salida de la ejecución

OUTPUT (state):
    - shap_data      : dict con top_variable, top_features, direccion_impacto
                       y zona_ideal. Dict vacío si SHAP falla.
    - error_messages : append si SHAP no está disponible

ARCHIVO: n09_explain_shap.json
"""
import os
from graph_state import GraphState
from xai.shap_explainer import SHAPExplainer
from node_utils import save_node_json


def explain_shap_node(state: GraphState) -> GraphState:
    """
    Nodo N09: calcula valores SHAP y exporta gráfico de importancia global.

    Parámetros:
        state (GraphState): estado del pipeline.

    Retorna:
        GraphState: estado actualizado con shap_data.
                    Si SHAP falla, shap_data queda como dict vacío
                    y el pipeline continúa.
    """
    print("\n[N09:ExplainSHAP] Calculando importancia global (SHAP)...")
    rf_model   = state['rf_model']
    X_test     = state['X_test']
    output_dir = state['output_dir']

    try:
        shap_explainer = SHAPExplainer()
        shap_data = shap_explainer.explain_and_plot(rf_model, X_test, output_dir)
        state['shap_data'] = shap_data

        top_features = shap_data.get('top_features') or shap_data.get('top_3_variables', [])
        save_node_json({
            "node": "N09_ExplainSHAP",
            "top_variable": shap_data.get('top_variable') or shap_data.get('variable_principal_calculada'),
            "top_features": top_features,
            "direccion_impacto": shap_data.get('direccion_impacto'),
            "zona_ideal": shap_data.get('zona_ideal_tecnica'),
            "plot_path": os.path.join(output_dir, "shap_summary.png"),
            "status": "ok"
        }, output_dir, "n09_explain_shap.json")

        print(f"[N09:✓] SHAP calculado | Factor principal: "
              f"{shap_data.get('top_variable') or shap_data.get('variable_principal_calculada')}")

    except Exception as e:
        print(f"[N09:WARN] SHAP falló: {e}")
        state['error_messages'].append(f"SHAP no disponible: {e}")
        state['shap_data'] = {}
        save_node_json({
            "node": "N09_ExplainSHAP",
            "status": "error",
            "error": str(e)
        }, output_dir, "n09_explain_shap.json")

    return state