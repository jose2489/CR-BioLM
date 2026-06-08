import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from .base_model import BaseModel
import config


class RandomForestSDM(BaseModel):
    """
    Implementación concreta del algoritmo Random Forest para el
    Modelado de Distribución de Especies (SDM).

    Parámetros de construcción configurables en config.py:
        SEED           : semilla de reproducibilidad
        RF_N_ESTIMATORS: número de árboles del ensemble
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            random_state=config.SEED,
            n_jobs=-1  # utiliza todos los núcleos del procesador
        )
        self.feature_names = None

    def train(self, X_train, y_train):
        """
        Entrena el modelo con los datos de entrenamiento.

        Parámetros:
            X_train: DataFrame con features de entrenamiento
            y_train: Series con etiquetas de entrenamiento
        """
        print("[INFO] Entrenando algoritmo Random Forest...")
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        print("[INFO] Entrenamiento completado exitosamente.")

    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo sobre el conjunto de prueba.

        Parámetros:
            X_test: DataFrame con features de prueba
            y_test: Series con etiquetas reales

        Retorna:
            Dict con accuracy, roc_auc y confusion_matrix.
        """
        print("[INFO] Evaluando rendimiento del modelo con datos de prueba...")
        y_pred  = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        cm  = confusion_matrix(y_test, y_pred)

        print(f"[METRICAS] Exactitud (Accuracy)        : {acc:.4f}")
        print(f"[METRICAS] Área bajo la curva (ROC-AUC): {roc:.4f}")
        print(f"[METRICAS] Matriz de Confusión:\n{cm}")

        return {
            "accuracy":         acc,
            "roc_auc":          roc,
            "confusion_matrix": cm.tolist()
        }

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_model(self):
        """
        Retorna el clasificador interno, necesario para inyectarlo en SHAP o LIME.
        """
        return self.model