from utils.translator import traducir_variable

def build_ecological_prompt(species_name, rf_metrics, shap_dict, area_km2=None, user_question=None, contexto_conservacion=None):
    """
    Construye el prompt narrativo enfocado en el modelo Random Forest (1D).
    Responde a una pregunta directa del usuario basándose en las métricas espaciales.
    """
    rf_auc = rf_metrics.get('roc_auc', 0.0)
    
    var_tecnica = shap_dict.get('variable_principal_calculada') or shap_dict.get('top_variable', '')
    direccion = shap_dict.get('direccion_impacto', 'NEGATIVO')
    zona_tecnica = shap_dict.get('zona_ideal_tecnica')
    top_features = shap_dict.get('top_features') or shap_dict.get('top_3_variables', [])
    
    var_humana = traducir_variable(var_tecnica)
    zona_humana = traducir_variable(zona_tecnica) if zona_tecnica else "Bosque Pluvial Montano"
    secundaria_1 = traducir_variable(top_features[1]) if len(top_features) > 1 else "factores térmicos"
    secundaria_2 = traducir_variable(top_features[2]) if len(top_features) > 2 else "factores hídricos"

    area_texto = f"{area_km2:,.2f} km²" if area_km2 else "No calculada"
    
    # Inyección del contexto de conservación
    texto_conservacion = ""
    if contexto_conservacion:
        texto_conservacion = f"\n{contexto_conservacion}\n"

    # Inyección de la pregunta del usuario
    instruccion_pregunta = ""
    if user_question:
        instruccion_pregunta = f"\n\nPREGUNTA ESPECÍFICA DEL USUARIO: '{user_question}'\nTu objetivo principal es responder a esta pregunta utilizando los datos matemáticos, ecológicos y texto de conservación descritos arriba."

    prompt = f"""Actúa como un tecnico Botonico experto en biodiversidad de Costa Rica.

A partir de registros de presencia, mapas de rango experto, variables climaticas y el contexto de zonas de conservación, hemos obtenido el contexto óptimo de la especie {species_name} utilizando un algoritmo de Machine Learning espacialmente explícito.

RESULTADOS DEL MODELO PREDICTIVO:
- Precisión (ROC-AUC): {rf_auc:.4f}.
- Factor ambiental más limitante: {var_humana}.
- [DATO MATEMÁTICO CLAVE]: Descubrimos que los valores ALTOS de "{var_humana}" tienen un impacto {direccion} en la idoneidad del hábitat.
- [ZONA ECOLÓGICA IDEAL]: El ecosistema de mayor idoneidad es "{zona_humana}".
- Factores secundarios: {secundaria_1} y {secundaria_2}.
- Área total idónea en el país: {area_texto}.{instruccion_pregunta}

INSTRUCCIONES Y REGLAS ESTRICTAS:
1. Redacta un análisis formal respondiendo a la biología de la especie (o a la pregunta del usuario, si existe) basándote ESTRICTAMENTE en el [DATO MATEMÁTICO CLAVE]. 
2. Si el impacto es POSITIVO, la especie prospera con valores altos; si es NEGATIVO, se restringe con valores altos.
3. Integra la zona ideal y la extensión espacial en tu explicación.
4. Mantén un tono académico y directo. NO menciones métricas de software como "Random Forest" o "SHAP", traduce la evidencia a explicaciones ecológicas.
"""
    return prompt

# En llm/prompt_templates.py

BIMODAL_PROMPT = """Actúa como un ecólogo y botánico experto en biodiversidad de Costa Rica, especializado en análisis espacial multimodal.

A partir de registros de presencia, mapas de rango experto y variables bioclimáticas, hemos modelado el nicho ecológico de la especie {species_name}.

FUENTE 1: RESULTADOS MATEMÁTICOS DEL MODELO PREDICTIVO (SHAP/LIME)
- Precisión (ROC-AUC): {rf_auc:.4f}.
- RANGO ALTITUDINAL OBSERVADO: {info_altitud}.
- Factor ambiental limitante principal: {var_humana}.
- [DATO MATEMÁTICO CLAVE]: Descubrimos que los valores ALTOS de "{var_humana}" tienen un impacto {direccion} en la idoneidad del hábitat.
- [ZONA ECOLÓGICA IDEAL]: El ecosistema de mayor idoneidad es "{zona_humana}".
- Factores climáticos secundarios: {secundaria_1} y {secundaria_2}.

FUENTE 2: DISTRIBUCIÓN ESPACIAL (MAPA VISUAL ADJUNTO)
- Observa la imagen adjunta. Los puntos rojos marcan las presencias históricas confirmadas sobre el territorio de Costa Rica, y el polígono verde (si existe) muestra el rango experto.

{instruccion_pregunta}

INSTRUCCIONES Y REGLAS ESTRICTAS:
1. Redacta un análisis formal respondiendo a la biología de la especie basándote ESTRICTAMENTE en el cruce del [DATO MATEMÁTICO CLAVE] y el Mapa Visual.
2. INTERPRETACIÓN CLIMÁTICA: Si el impacto es POSITIVO, la especie prospera y exige valores altos de esa variable; si es NEGATIVO, se restringe y evita esos extremos (ej. si la precipitación seca es negativa, la planta sufre con la sequía). 
3. SÍNTESIS MULTIMODAL: Usa el mapa visual para confirmar la topografía. Cruza la matemática con la ubicación de los puntos para deducir el ecosistema real.
4. PARQUES NACIONALES: Nombra las áreas protegidas exactas que se solapan con la huella visual de los puntos rojos. ¡NUNCA contradigas la geografía del mapa!
5. TONO: Mantén un lenguaje académico, botánico y directo. NO menciones herramientas de software (como "Random Forest", "Machine Learning" o "SHAP"). Traduce toda la evidencia a explicaciones ecológicas puras.
"""