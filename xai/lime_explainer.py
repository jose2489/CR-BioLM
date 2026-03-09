import os
import numpy as np
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
from utils.translator import traducir_lista_variables

class LIMEExplainer:
    """
    Clase dedicada a la explicabilidad LOCAL mediante LIME.
    Explica el razonamiento del modelo para una coordenada (instancia) específica.
    """
    
    def explain_and_plot(self, trained_model_wrapper, X_train, X_test, output_dir):
        print("[INFO] Calculando explicabilidad local (LIME) para un punto de alta idoneidad...")
        
        # 1. Extraer el modelo y las columnas
        rf_model = trained_model_wrapper.get_model()
        feature_names = X_train.columns.tolist()

        # Traducimos la lista de nombres para LIME
        nombres_traducidos = traducir_lista_variables(feature_names)

        # 2. Inicializar el Explainer de LIME usando los datos de entrenamiento
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=nombres_traducidos, # <-- Inyectamos los nombres bonitos aquí
            class_names=['Ausencia', 'Presencia'],
            mode='classification',
            random_state=42
        )

        # 3. Buscar el píxel/punto en X_test donde el modelo está MÁS seguro de que hay presencia
        probabilidades = rf_model.predict_proba(X_test)[:, 1]
        idx_presencia_maxima = np.argmax(probabilidades)
        instancia_a_explicar = X_test.iloc[idx_presencia_maxima]

        # 4. Generar la explicación matemática para ese único punto
        exp = explainer.explain_instance(
            data_row=instancia_a_explicar.values,
            predict_fn=rf_model.predict_proba,
            num_features=6 # Mostramos el top 6 de variables para este punto
        )

        # 5. Generar y guardar el gráfico de LIME
        fig = exp.as_pyplot_figure()
        plt.title(f"LIME: Por qué hay presencia en el punto local elegido", fontsize=11)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "lime_local_explanation.png")
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        print(f"[INFO] Gráfico LIME guardado exitosamente en: {plot_path}")