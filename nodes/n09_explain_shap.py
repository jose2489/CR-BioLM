"""
N09: ExplainSHAP
Calcula importancia global de features usando SHAP.
"""
import os
import json
from graph_state import GraphState
from xai.shap_explainer import SHAPExplainer


def explain_shap_node(state: GraphState) -> GraphState:
    print("\n[N09:ExplainSHAP] Calculando importancia global (SHAP)...")

    rf_model = state['rf_model']
    X_test = state['X_test']
    output_dir = state['output_dir']

    try:
        shap_explainer = SHAPExplainer()
        shap_data = shap_explainer.explain_and_plot(rf_model, X_test, output_dir)
        state['shap_data'] = shap_data

        # Extraer top features para JSON (solo tipos serializables)
        top_features = shap_data.get('top_features') or shap_data.get('top_3_variables', [])
        _save_json({
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
        _save_json({
            "node": "N09_ExplainSHAP",
            "status": "error",
            "error": str(e)
        }, output_dir, "n09_explain_shap.json")

    return state


def _save_json(data: dict, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
