import os
import base64
import requests
# IMPORTANTE: Asegúrate de importar la función traducir_variable desde tu archivo de templates
from .prompt_templates import BIMODAL_PROMPT, traducir_variable
from config_llm import is_multimodal

class OpenRouterClient:
    def __init__(self, api_key, model="openrouter/free"):
        """
        Cliente para comunicarse con OpenRouter soportando modelos multimodales.
        """
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def _codificar_imagen(self, ruta_imagen):
        """Convierte la imagen a Base64."""
        with open(ruta_imagen, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_profile(self, species_name, rf_metrics, shap_dict, output_dir, image_path,
                         user_question=None, model_override=None, info_altitud="No disponible",
                         manual_image_path=None, texto_manual=""):
        """Orquesta la extracción de datos, inyección de prompt, inferencia y guardado."""
        
        modelo_a_usar = model_override if model_override else self.model
        print(f"[INFO] Preparando síntesis bimodal para OpenRouter ({modelo_a_usar})...")

        # ==========================================================
        # 1. EXTRACCIÓN Y TRADUCCIÓN DE VARIABLES DESDE SHAP
        # ==========================================================
        try:
            if isinstance(shap_dict, dict) and len(shap_dict) > 0:
                # Verificamos si es el diccionario crudo o el procesado
                if 'variable_principal_calculada' in shap_dict or 'top_variable' in shap_dict:
                    var_tecnica = shap_dict.get('variable_principal_calculada') or shap_dict.get('top_variable', 'Desconocida')
                    direccion = shap_dict.get('direccion_impacto', 'DESCONOCIDO')
                    zona_tecnica = shap_dict.get('zona_ideal_tecnica', 'Desconocida')
                    top_features = shap_dict.get('top_features') or shap_dict.get('top_3_variables', [])
                    
                    secundaria_1_tec = top_features[1] if len(top_features) > 1 else "factores térmicos"
                    secundaria_2_tec = top_features[2] if len(top_features) > 2 else "factores hídricos"
                else:
                    # Si es el diccionario crudo de SHAP { 'bio_17': 0.45, ... }
                    top_variables = list(shap_dict.keys())
                    var_tecnica = top_variables[0] if len(top_variables) > 0 else "Desconocida"
                    secundaria_1_tec = top_variables[1] if len(top_variables) > 1 else "factores térmicos"
                    secundaria_2_tec = top_variables[2] if len(top_variables) > 2 else "factores hídricos"
                    
                    valor_impacto = shap_dict.get(var_tecnica, 0)
                    direccion = "POSITIVO" if valor_impacto > 0 else "NEGATIVO"
                    zona_tecnica = "Bosque Maduro"

                # Traducimos a lenguaje humano para que el LLM entienda
                var_humana = traducir_variable(var_tecnica)
                secundaria_1 = traducir_variable(secundaria_1_tec)
                secundaria_2 = traducir_variable(secundaria_2_tec)
                zona_humana = traducir_variable(zona_tecnica) if zona_tecnica != 'Desconocida' else "Ecosistema predominante"

            else:
                var_humana, secundaria_1, secundaria_2, direccion, zona_humana = "Desconocida", "N/A", "N/A", "Desconocido", "Desconocida"
                
        except Exception as e:
            print(f"[WARN] Error procesando variables SHAP para el prompt: {e}")
            var_humana, secundaria_1, secundaria_2, direccion, zona_humana = "Desconocida", "N/A", "N/A", "Desconocido", "Desconocida"

        # ==========================================================
        # 2. CONFIGURAR VARIABLES RESTANTES Y FORMATEAR PROMPT
        # ==========================================================
        rf_auc = rf_metrics.get('roc_auc', 0.0) if rf_metrics else 0.0
        rf_accuracy = rf_metrics.get('accuracy', 0.0) if rf_metrics else 0.0
        pregunta_texto = user_question if user_question else "Analiza el hábitat ideal de esta especie."

        # Detectar si es modelo de solo texto
        is_text = not is_multimodal(modelo_a_usar)

        if is_text:
            # PROMPT OPTIMIZADO PARA SOLO TEXTO
            print("[INFO] Generando prompt optimizado para solo texto")
            
            # Construir resumen SHAP legible
            shap_summary = f"""
        1. {var_humana} (factor principal, impacto {direccion})
        2. {secundaria_1} (factor secundario)
        3. {secundaria_2} (factor terciario)
        Ecosistema predominante: {zona_humana}
        """
            
            prompt_listo = f"""## ANÁLISIS ECOLÓGICO: {species_name}

        ### Métricas del Modelo Predictivo
        - Precisión (Accuracy): {rf_accuracy:.1%}
        - Capacidad predictiva (ROC-AUC): {rf_auc:.3f}

        ### Factores Climáticos Clave (SHAP)
        {shap_summary}

        ### Rango Altitudinal Observado
        {info_altitud}

        ### Contexto Botánico
        {texto_manual if texto_manual else 'Información del Manual de Plantas no disponible'}

        ### Pregunta del Usuario
        {pregunta_texto}

        ---

        **TAREA**: Genera un perfil ecológico completo en español para esta especie. Incluye:
        1. Descripción de su distribución geográfica y altitudinal
        2. Requerimientos climáticos específicos basados en los factores SHAP
        3. Tipo de ecosistema donde prospera
        4. Recomendaciones de conservación y amenazas potenciales
        5. Respuesta directa a la pregunta del usuario

        Sé técnico pero accesible. Usa los datos proporcionados para fundamentar el análisis."""

        else:
            # PROMPT ORIGINAL BIMODAL (CON IMÁGENES)
            print("[INFO] Generando prompt bimodal (texto + imágenes)")
            
            fuente_manual = ""
            if texto_manual:
                fuente_manual = (
                    "\nFUENTE 3: REFERENCIA BOTÁNICA (Manual de Plantas de Costa Rica)\n"
                    + texto_manual
                    + "\nLa imagen 2 adjunta muestra el hábitat potencial según el Manual, "
                    "cruzado con las Unidades Fitogeográficas de CR. Úsala para validar o "
                    "contrastar los hallazgos matemáticos del modelo predictivo.\n"
                )
            
            try:
                prompt_listo = BIMODAL_PROMPT.format(
                    species_name=species_name,
                    rf_auc=rf_auc,
                    info_altitud=info_altitud,
                    var_humana=var_humana,
                    direccion=direccion,
                    zona_humana=zona_humana,
                    secundaria_1=secundaria_1,
                    secundaria_2=secundaria_2,
                    area_texto="No calculada",
                    fuente_manual=fuente_manual,
                    instruccion_pregunta=f"PREGUNTA DEL USUARIO: {pregunta_texto}"
                )
            except KeyError as e:
                print(f"[ERROR] Falta una llave en el BIMODAL_PROMPT: {e}. Revisa llm/prompt_templates.py")
                return False

        # ==========================================================
        # 3. CODIFICAR IMAGEN(ES) Y ARMAR PAYLOAD PARA LA API
        # ==========================================================

        # Detectar si el modelo no soporta imágenes
        is_text = not is_multimodal(modelo_a_usar)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        if is_text:
            # Solo texto, sin imágenes
            print("[INFO] Modelo Textual detectado - usando prompt solo texto")
            payload = {
                "model": modelo_a_usar,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_listo  # Solo texto
                    }
                ]
            }
        else:
            # Modelos visuales: Con imágenes
            print(f"[DEBUG] API Key present: {bool(self.api_key)}")
            print(f"[DEBUG] Model: {modelo_a_usar}")
            print(f"[DEBUG] Is multimodal: {not is_text}")
            print("[INFO] Modelo visual detectado - incluyendo imágenes")
            imagen_base64 = self._codificar_imagen(image_path)
            content_parts = [
                {"type": "text", "text": prompt_listo},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{imagen_base64}"}}
            ]
            
            if manual_image_path and os.path.isfile(str(manual_image_path)):
                manual_b64 = self._codificar_imagen(str(manual_image_path))
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{manual_b64}"}
                })
                print(f"[INFO] Segunda imagen (Manual) adjunta: {manual_image_path}")
            
            # Build payload (no respuesta reference here)
            payload = {
                "model": modelo_a_usar,
                "messages": [
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ]
            }
        # ==========================================================
        # 4. EJECUTAR LLAMADA A OPENROUTER Y GUARDAR RESULTADOS
        # ==========================================================
        try:
            print(f"[LLM] Solicitando análisis a la API de OpenRouter...")
            # Add timeout to avoid hanging and catch connection errors
            respuesta = requests.post(self.url, headers=headers, json=payload, timeout=60)
            
            print(f"[DEBUG] Status code: {respuesta.status_code}")
            print(f"[DEBUG] Respuesta raw: {respuesta.text[:500]}")
            
            if respuesta.status_code == 200:
                perfil_texto = respuesta.json()['choices'][0]['message']['content']
                
                # Creado de metadatos
                encabezado_metadatos = f"""================================================================================
METADATOS DEL EXPERIMENTO (Arquitectura CR-BioLM Multimodal)
================================================================================
Modelo LLM       : {modelo_a_usar} (Visión + Texto)
Especie          : {species_name}
Pregunta Usuario : {pregunta_texto}

FUENTES DE DATOS Y VARIABLES ESPACIALES INTEGRADAS:
- Imagen 1       : {os.path.basename(str(image_path))} — Hábitat botánico (Manual + Ufito + GBIF)
- Imagen 2       : {os.path.basename(str(manual_image_path)) if manual_image_path else 'N/A'} — Modelo predictivo RF
- Modelo RF      : Random Forest (AUC: {rf_auc:.4f})
- Presencias     : Registros GBIF limpios (Mesoamérica para RF, CR para mapa)
================================================================================

[ANÁLISIS HÍBRIDO GENERADO POR IA]
"""
                modelo_limpio = modelo_a_usar.replace('/', '_').replace('-', '_').replace(':', '_')
                ruta_salida_txt = os.path.join(output_dir, f"llm_profile_BIMODAL_{modelo_limpio}.txt")
                
                if perfil_texto is None:
                    perfil_texto = "[El modelo no generó respuesta para esta consulta]"

                with open(ruta_salida_txt, "w", encoding="utf-8") as file:
                    file.write(encabezado_metadatos)
                    file.write(perfil_texto)
                print(f"[EXITO] Perfil Multimodal guardado en: {ruta_salida_txt}")
                return True
            else:
                print(f"[ERROR] Falló la API de OpenRouter: {respuesta.text}")
                # Save error to file
                error_path = os.path.join(output_dir, "llm_api_error.txt")
                with open(error_path, "w", encoding="utf-8") as f:
                    f.write(f"Status: {respuesta.status_code}\n\n{respuesta.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Falló la conexión a OpenRouter: {e}")
            # Save connection error
            error_path = os.path.join(output_dir, "llm_api_error.txt")
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"Connection error: {str(e)}")
            return False
        except Exception as e:
            print(f"[ERROR] Falló la inferencia multimodal: {e}")
            import traceback
            traceback.print_exc()
            return False