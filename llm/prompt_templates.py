from utils.translator import traducir_variable

def build_ecological_prompt(species_name, rf_metrics, cnn_metrics, shap_dict, area_km2=None):
    """
    Construye el prompt narrativo final, inyectando la comparativa de modelos (RF vs CNN), 
    el área calculada, la zona ideal, el candado lógico y los nombres traducidos.
    """
    # 1. Extracción de métricas de ambos modelos
    rf_auc = rf_metrics.get('roc_auc', 0.0)
    cnn_auc = cnn_metrics.get('roc_auc', 0.0)
    
    # 2. Extracción BLINDADA (SHAP se extrae del modelo tabular para explicabilidad)
    var_tecnica = shap_dict.get('variable_principal_calculada') or shap_dict.get('top_variable', '')
    direccion = shap_dict.get('direccion_impacto', 'NEGATIVO')
    zona_tecnica = shap_dict.get('zona_ideal_tecnica')
    top_features = shap_dict.get('top_features') or shap_dict.get('top_3_variables', [])
    
    # 3. Traducción a lenguaje humano
    var_humana = traducir_variable(var_tecnica)
    zona_humana = traducir_variable(zona_tecnica) if zona_tecnica else "Bosque Pluvial Montano"
    
    secundaria_1 = traducir_variable(top_features[1]) if len(top_features) > 1 else "otros factores térmicos"
    secundaria_2 = traducir_variable(top_features[2]) if len(top_features) > 2 else "variables geográficas"

    # Formateo del área matemática
    area_texto = f"{area_km2:,.2f} km²" if area_km2 else "No calculada"

    # 4. Ensamblaje del Prompt Dinámico Comparativo
    prompt = f"""Actúa como un ecólogo experto en biodiversidad tropical y modelado espacial.

A partir de los registros de presencia y mapas bioclimáticos, hemos modelado el nicho ecológico de la especie {species_name} en Costa Rica comparando dos enfoques de Inteligencia Artificial.

RESULTADOS DE LOS MODELOS PREDICTIVOS:
1. Modelo Tradicional (Tabular 1D): Alcanzó un ROC-AUC de {rf_auc:.4f}. Este modelo evalúa el clima exactamente en el punto geográfico (1x1 píxel) de la planta.
2. Modelo de Deep Learning (Espacial 2D): Alcanzó un ROC-AUC de {cnn_auc:.4f}. Este modelo evalúa un parche espacial (topografía y microclima en una ventana de 15x15 píxeles) fusionado con el tipo de ecosistema.

HALLAZGOS ECOLÓGICOS (Vía SHAP y LIME):
- Factor ambiental más limitante: {var_humana}.
- [DATO MATEMÁTICO CLAVE]: Descubrimos que los valores ALTOS de "{var_humana}" tienen un impacto {direccion} en la idoneidad del hábitat.
- [ZONA ECOLÓGICA IDEAL]: El ecosistema de mayor idoneidad y refugio geográfico es "{zona_humana}".
- Factores secundarios relevantes: {secundaria_1} y {secundaria_2}.
- Área total geográficamente idónea en el país: {area_texto}.

Tu tarea es redactar la Discusión Científica estructurada en Markdown.

INSTRUCCIONES Y REGLAS ESTRICTAS:
1. SECCIÓN: "## 1. Requisitos Ecológicos y Área de Distribución". 
Explica la biología de la especie basándote ESTRICTAMENTE en el [DATO MATEMÁTICO CLAVE] (si es POSITIVO, la especie prospera con valores altos; si es NEGATIVO, se estresa con valores altos). Integra la zona ideal ("{zona_humana}") y la extensión de su área idónea estimada en {area_texto}.

2. SECCIÓN: "## 2. Análisis Comparativo de Modelos (1D vs 2D)". 
Analiza la diferencia de rendimiento (ROC-AUC) entre el modelo tabular ({rf_auc:.4f}) y el modelo de Deep Learning ({cnn_auc:.4f}). Explica, desde la ecología espacial, por qué analizar el contexto topográfico/microclimático circundante (2D) captura mejor los gradientes ecológicos que evaluar un punto aislado (1D).

3. LENGUAJE CIENTÍFICO: Tono formal, académico y directo. NO uses códigos técnicos (ej. no digas bio_1, usa los nombres naturales). No generes introducciones vacías, ve directo al texto de las secciones.
"""
    return prompt