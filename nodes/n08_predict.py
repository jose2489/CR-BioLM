"""
N08: Predict
Genera predicciones de presencia/ausencia y probabilidades
usando el modelo Random Forest entrenado.

INPUT  (state):
    - rf_model : instancia entrenada de RandomForestSDM
    - X_test   : DataFrame con features de prueba

OUTPUT (state):
    - predictions   : array con clases predichas (0/1)
    - probabilities : array con probabilidad de presencia por muestra

ARCHIVO: n08_predict.json
"""
from graph_state import GraphState
from node_utils import save_node_json


def predict_node(state: GraphState) -> GraphState:
    """
    Nodo N08: genera predicciones y probabilidades sobre el set de prueba.

    Parámetros:
        state (GraphState): estado del pipeline.

    Retorna:
        GraphState: estado actualizado con predictions y probabilities.
    """
    print("\n[N08:Predict] Generando predicciones...")
    rf_model = state['rf_model']
    X_test   = state['X_test']

    predictions   = rf_model.model.predict(X_test)
    probabilities = rf_model.model.predict_proba(X_test)[:, 1]

    state['predictions']   = predictions
    state['probabilities'] = probabilities

    save_node_json({
        "node": "N08_Predict",
        "num_muestras": len(predictions),
        "presencias_predichas": int((predictions == 1).sum()),
        "ausencias_predichas": int((predictions == 0).sum()),
        "probabilidad_media": float(probabilities.mean()),
        "probabilidad_min": float(probabilities.min()),
        "probabilidad_max": float(probabilities.max()),
        "status": "ok"
    }, state['output_dir'], "n08_predict.json")

    print(f"[N08:✓] Predicciones generadas: {len(predictions)} muestras")
    return state