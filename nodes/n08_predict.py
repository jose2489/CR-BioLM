"""
N08: Predict
Genera predicciones usando el modelo Random Forest entrenado.
"""
import os
import json
from graph_state import GraphState


def predict_node(state: GraphState) -> GraphState:
    print("\n[N08:Predict] Generando predicciones...")

    rf_model = state['rf_model']
    X_test = state['X_test']

    predictions = rf_model.model.predict(X_test)
    probabilities = rf_model.model.predict_proba(X_test)[:, 1]

    state['predictions'] = predictions
    state['probabilities'] = probabilities

    _save_json({
        "node": "N08_Predict",
        "num_muestras": len(predictions),
        "presencias_predichas": int((predictions == 1).sum()),
        "ausencias_predichas": int((predictions == 0).sum()),
        "probabilidad_media": float(probabilities.mean()),
        "probabilidad_min": float(probabilities.min()),
        "probabilidad_max": float(probabilities.max()),
        "status": "ok"
    }, state['output_dir'], "n08b_predict.json")

    print(f"[N08:✓] Predicciones generadas: {len(predictions)} muestras")
    return state


def _save_json(data: dict, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
