"""
N10: ExplainLIME
Genera explicaciones locales por instancia usando LIME.
"""
import os
import json
from graph_state import GraphState
from xai.lime_explainer import LIMEExplainer


def explain_lime_node(state: GraphState) -> GraphState:
    print("\n[N10:ExplainLIME] Generando explicaciones locales (LIME)...")

    rf_model = state['rf_model']
    X_train = state['X_train']
    X_test = state['X_test']
    output_dir = state['output_dir']

    try:
        lime_explainer = LIMEExplainer()
        lime_data = lime_explainer.explain_and_plot(rf_model, X_train, X_test, output_dir)
        state['lime_data'] = lime_data

        _save_json({
            "node": "N10_ExplainLIME",
            "plot_path": os.path.join(output_dir, "lime_local_explanation.png"),
            "status": "ok"
        }, output_dir, "n10_explain_lime.json")

        print(f"[N10:✓] LIME generado y guardado en {output_dir}")

    except Exception as e:
        print(f"[N10:WARN] LIME falló: {e}")
        state['error_messages'].append(f"LIME no disponible: {e}")
        state['lime_data'] = {}
        _save_json({
            "node": "N10_ExplainLIME",
            "status": "error",
            "error": str(e)
        }, output_dir, "n10_explain_lime.json")

    return state


def _save_json(data: dict, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
