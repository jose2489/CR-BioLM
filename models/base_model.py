from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Clase abstracta que define la estructura obligatoria para cualquier 
    algoritmo predictivo dentro del pipeline CR-BioLM.
    """
    
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Todo modelo debe implementar una logica de entrenamiento.
        """
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """
        Todo modelo debe retornar un diccionario con metricas estandar 
        (ej. accuracy, roc_auc, matriz_confusion).
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Requerido para la explicabilidad (SHAP/LIME). Debe retornar 
        las probabilidades continuas, no solo clasificaciones binarias.
        """
        pass