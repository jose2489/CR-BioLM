"""
N07: TrainRandomForest
Entrena y evalúa el modelo Random Forest para SDM.

INPUT  (state):
    - X_train : DataFrame con features de entrenamiento
    - y_train : Series con etiquetas de entrenamiento
    - X_test  : DataFrame con features de prueba
    - y_test  : Series con etiquetas de prueba

OUTPUT (state):
    - rf_model   : instancia entrenada de RandomForestSDM
    - rf_metrics : dict con accuracy, roc_auc y confusion_matrix

ARCHIVO: n07_train_random_forest.json
"""
from graph_state import GraphState
from models.random_forest import RandomForestSDM
from node_utils import save_node_json


def train_random_forest_node(state: GraphState) -> GraphState:
    """
    Nodo N07: entrena un Random Forest y evalúa su desempeño.

    Parámetros:
        state (GraphState): estado del pipeline.

    Retorna:
        GraphState: estado actualizado con rf_model y rf_metrics.
    """
    print("\n[N07:TrainRandomForest] Entrenando Random Forest...")
    X_train = state['X_train']
    y_train = state['y_train']
    X_test  = state['X_test']
    y_test  = state['y_test']

    rf_model = RandomForestSDM()
    rf_model.train(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)

    state['rf_model']   = rf_model
    state['rf_metrics'] = rf_metrics

    # Serializar confusion matrix (ndarray → lista)
    cm = rf_metrics.get('confusion_matrix')
    try:
        cm_serializable = cm.tolist()
    except AttributeError:
        cm_serializable = cm

    save_node_json({
        "node": "N07_TrainRandomForest",
        "accuracy": rf_metrics.get('accuracy'),
        "roc_auc": rf_metrics.get('roc_auc'),
        "confusion_matrix": cm_serializable,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": list(X_train.columns),
        "status": "ok"
    }, state['output_dir'], "n07_train_random_forest.json")

    print(f"[N07:✓] RF entrenado | Accuracy: {rf_metrics.get('accuracy', 0):.3f} | "
          f"ROC-AUC: {rf_metrics.get('roc_auc', 0):.3f}")
    return state