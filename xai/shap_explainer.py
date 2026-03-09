import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.translator import traducir_lista_variables

class SHAPExplainer:
    """
    Clase dedicada a la explicabilidad global y local del modelo utilizando 
    la teoria de juegos cooperativos (Valores de Shapley).
    """
    
    def explain_and_plot(self, trained_model_wrapper, X_test, output_dir):
        """
        Calcula los valores SHAP, genera el grafico de resumen y extrae 
        las variables mas limitantes para alimentar al LLM.
        """
        print("[INFO] Calculando valores SHAP (Explicabilidad Global)...")
        
        # 1. Extraer el modelo base (Scikit-Learn) del wrapper de nuestra arquitectura
        rf_model = trained_model_wrapper.get_model()
        
        # 2. Limpieza de indices de Pandas (La solucion al bug de colapso)
        # Forzamos a que el DataFrame tenga un indice secuencial limpio (0, 1, 2...)
        if isinstance(X_test, pd.DataFrame):
            X_test_clean = X_test.reset_index(drop=True)
            feature_names = X_test_clean.columns.tolist()
        else:
            X_test_clean = X_test
            feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

        # 3. Calculo matematico de SHAP
        # TreeExplainer esta altamente optimizado para Random Forest
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test_clean)
        
        # 4. Extraccion de la clase positiva (Presencia = 1)
        # Scikit-Learn RF retorna explicaciones para ambas clases. Solo nos interesa la clase 1.
        if isinstance(shap_values, list):
            shap_values_pos = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_pos = shap_values[:, :, 1]
        else:
            shap_values_pos = shap_values

        # 5. Generacion y guardado del grafico
        print("[INFO] Generando grafico de impacto SHAP...")
        plt.figure(figsize=(10, 6))
        
        # --- EL TRUCO DEFINITIVO PARA SHAP ---
        # Traducimos la lista de nombres originales
        nombres_traducidos = traducir_lista_variables(feature_names)
        
        # Forzamos a SHAP a usar nuestros nombres inyectando el parámetro 'feature_names'
        shap.summary_plot(
            shap_values_pos, 
            X_test_clean, 
            feature_names=nombres_traducidos, 
            show=False
        )
        
        # Guardar en la carpeta dinamica de la ejecucion
        plot_path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"[INFO] Grafico SHAP guardado exitosamente en: {plot_path}")

        # 6. Extraccion de estadisticas para el LLM
        # Calculamos el impacto absoluto promedio de cada variable
        mean_shap = np.abs(shap_values_pos).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Variable': feature_names, 
            'Importancia_SHAP': mean_shap
        })
        importance_df = importance_df.sort_values(by='Importancia_SHAP', ascending=False)
        
        top_1_variable = importance_df.iloc[0]['Variable']
        top_3_variables = importance_df.head(3)['Variable'].tolist()
        
        print(f"[INFO] Factor limitante principal identificado: {top_1_variable}")

        # ==========================================
        # --- LÓGICA DE INTERPRETACIÓN ECOLÓGICA ---
        # ==========================================
        
        # A. Candado Lógico (Dirección del Impacto)
        # Obtenemos las probabilidades de presencia predichas por el modelo
        probabilidades = rf_model.predict_proba(X_test_clean)[:, 1]
        valores_climaticos = X_test_clean[top_1_variable].values
        
        # Medimos si cuando la variable sube, la probabilidad sube (positivo) o baja (negativo)
        correlacion = np.corrcoef(valores_climaticos, probabilidades)[0, 1]
        direccion = "POSITIVO" if correlacion > 0 else "NEGATIVO"

        # B. Radar de Zonas (Ecorregión Ganadora)
        columnas_eco = [c for c in feature_names if c.startswith('Eco_')]
        zona_ideal = None
        
        if columnas_eco:
            # Filtramos el DataFrame de importancia solo para las ecorregiones
            importancias_eco = {col: importance_df[importance_df['Variable'] == col]['Importancia_SHAP'].values[0] for col in columnas_eco}
            # Elegimos la ecorregión que más impacta al modelo
            zona_top_tecnica = max(importancias_eco, key=importancias_eco.get)
            
            # Verificamos que al menos tenga un impacto real en las predicciones
            if importancias_eco[zona_top_tecnica] > 0:
                zona_ideal = zona_top_tecnica

        # Retornamos el diccionario empaquetado exactamente con las llaves que 
        # nuestro archivo `prompt_templates.py` está esperando leer.
        return {
            "variable_principal_calculada": top_1_variable,
            "direccion_impacto": direccion,
            "zona_ideal_tecnica": zona_ideal,
            "top_features": top_3_variables,
            "importance_dataframe": importance_df #
        }