from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from .base_model import BaseModel

class RandomForestSDM(BaseModel):
    """
    Implementacion concreta del algoritmo Random Forest para el 
    Modelado de Distribucion de Especies (SDM).
    """
    def __init__(self, random_state=42, n_estimators=100):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=self.random_state,
            n_jobs=-1 # Utiliza todos los nucleos del procesador
        )
        self.feature_names = None

    def train(self, X_train, y_train):
        print("[INFO] Entrenando algoritmo Random Forest...")
        
        # Guardamos los nombres de las columnas para usarlos luego en xAI
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
            
        self.model.fit(X_train, y_train)
        print("[INFO] Entrenamiento completado exitosamente.")

    def evaluate(self, X_test, y_test):
        print("[INFO] Evaluando rendimiento del modelo con datos de prueba...")
        
        # Predicciones crudas (0 o 1) para Exactitud y Matriz de Confusion
        y_pred = self.model.predict(X_test)
        
        # Probabilidades continuas (0.0 a 1.0) para el ROC-AUC
        y_proba = self.model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        print(f"[METRICAS] Exactitud (Accuracy) : {acc:.4f}")
        print(f"[METRICAS] Area bajo la curva (ROC-AUC) : {roc:.4f}")
        print(f"[METRICAS] Matriz de Confusion :\n{cm}")

        # Retornamos un diccionario empaquetado para que el LLM lo consuma facilmente
        return {
            "accuracy": acc,
            "roc_auc": roc,
            "confusion_matrix": cm.tolist()
        }

    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def get_model(self):
        """
        Metodo auxiliar para retornar el objeto clasificador puro, 
        necesario para inyectarlo en SHAP o LIME.
        """
        return self.model