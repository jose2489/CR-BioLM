from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    Clase abstracta que define la estructura obligatoria para cualquier 
    cliente de Inteligencia Artificial Generativa.
    """
    
    @abstractmethod
    def generate_profile(self, species_name, metrics_dict, shap_dict, output_dir):
        """
        Toma los resultados matematicos y de explicabilidad, genera un prompt, 
        consulta al modelo de lenguaje y guarda el resultado en disco.
        """
        pass