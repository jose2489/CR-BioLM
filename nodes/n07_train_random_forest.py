"""
N07: TrainRandomForest
Entrena y evalúa el modelo Random Forest para SDM.
"""
import os
import json
from graph_state import GraphState
from models.random_forest import RandomForestSDM


def train_random_forest_node(state: GraphState) -> GraphState:
    print("\n[N07:TrainRandomForest] Entrenando Random Forest...")

    X_train = state['X_train']
    y_train = state['y_train']
    X_test = state['X_test']
    y_test = state['y_test']

    rf_model = RandomForestSDM()
    rf_model.train(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)

    state['rf_model'] = rf_model
    state['rf_metrics'] = rf_metrics

    # Serializar confusion matrix (ndarray → lista)
    confusion_matrix_serializable = None
    if 'confusion_matrix' in rf_metrics:
        cm = rf_metrics['confusion_matrix']
        try:
            confusion_matrix_serializable = cm.tolist()
        except AttributeError:
            confusion_matrix_serializable = cm

    _save_json({
        "node": "N07_TrainRandomForest",
        "accuracy": rf_metrics.get('accuracy'),
        "roc_auc": rf_metrics.get('roc_auc'),
        "confusion_matrix": confusion_matrix_serializable,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": list(X_train.columns),
        "status": "ok"
    }, state['output_dir'], "n08a_train_random_forest.json")

    print(f"[N07:✓] RF entrenado | Accuracy: {rf_metrics.get('accuracy', 0):.3f} | "
          f"ROC-AUC: {rf_metrics.get('roc_auc', 0):.3f}")
    return state


def _save_json(data: dict, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
