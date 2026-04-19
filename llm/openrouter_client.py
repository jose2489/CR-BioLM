import io
import os
import base64
import requests
from PIL import Image
# IMPORTANTE: Asegúrate de importar la función traducir_variable desde tu archivo de templates
from .prompt_templates import PROMPT_T0, PROMPT_T1, PROMPT_T2, PROMPT_T3, _REGLA_STRICTA, traducir_variable

TIER_PROMPTS = {"T0": PROMPT_T0, "T1": PROMPT_T1, "T2": PROMPT_T2, "T3": PROMPT_T3}

MAX_IMG_PX = 1024  # max pixels on the longest edge before base64 encoding

class OpenRouterClient:
    def __init__(self, api_key, model="openai/gpt-4o"):
        """
        Cliente para comunicarse con OpenRouter soportando modelos multimodales.
        """
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def _codificar_imagen(self, ruta_imagen, max_px=MAX_IMG_PX):
        """Redimensiona la imagen a max_px en el lado más largo, luego convierte a Base64."""
        img = Image.open(ruta_imagen)
        img.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        fmt = img.format or "PNG"
        if fmt not in ("PNG", "JPEG", "WEBP"):
            fmt = "PNG"
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate_profile(self, species_name, rf_metrics, shap_dict, output_dir, image_path,
                         user_question=None, model_override=None, info_altitud="No disponible",
                         manual_image_path=None, texto_manual="", tier="T3"):
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
        pregunta_texto = user_question if user_question else "Analiza el hábitat ideal de esta especie."

        # Asegúrate de que las llaves de .format() coincidan exactamente con tu BIMODAL_PROMPT
        fuente_manual = ""
        if texto_manual:
            fuente_manual = (
                "\nFUENTE 3: REFERENCIA BOTÁNICA (Manual de Plantas de Costa Rica)\n"
                + texto_manual
                + "\nLa imagen 2 adjunta muestra el hábitat potencial según el Manual, "
                "cruzado con las Unidades Fitogeográficas de CR. Úsala para validar o "
                "contrastar los hallazgos matemáticos del modelo predictivo.\n"
            )
        # Seleccionar prompt según tier — cada uno describe honestamente lo que el LLM recibe
        template = TIER_PROMPTS.get(tier, PROMPT_T3)
        print(f"[INFO] Usando prompt {tier}")

        # T1 solo necesita instruccion_pregunta; T2 agrega fuente_manual; T3 agrega métricas RF
        format_kwargs = dict(
            species_name=species_name,
            instruccion_pregunta=f"PREGUNTA DEL USUARIO: {pregunta_texto}",
            fuente_manual=fuente_manual,
            _regla=_REGLA_STRICTA,
            rf_auc=rf_auc,
            info_altitud=info_altitud,
            var_humana=var_humana,
            direccion=direccion,
            zona_humana=zona_humana,
            secundaria_1=secundaria_1,
            secundaria_2=secundaria_2,
        )
        try:
            prompt_listo = template.format(**format_kwargs)
        except KeyError as e:
            print(f"[ERROR] Falta una llave en el prompt {tier}: {e}. Revisa llm/prompt_templates.py")
            return False

        # ==========================================================
        # 3. CODIFICAR IMAGEN(ES) Y ARMAR PAYLOAD PARA LA API
        # T0 es texto únicamente — no se envían imágenes
        # ==========================================================
        if tier == "T0":
            content_parts = [{"type": "text", "text": prompt_listo}]
            print("[INFO] T0: llamada texto-únicamente (sin imágenes)")
        else:
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

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

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
            respuesta = requests.post(self.url, headers=headers, json=payload)
            
            if respuesta.status_code == 200:
                perfil_texto = respuesta.json()['choices'][0]['message']['content']
                
                # Metadata header: tier-aware labels
                if tier == "T0":
                    _fuentes_line = "- Fuentes        : NINGUNA — conocimiento previo del LLM únicamente"
                else:
                    _img1_label = {
                        "T1": "Distribución GBIF Mesoamérica (puntos de presencia)",
                        "T2": "Hábitat botánico (Manual + Unidades Fitogeográficas + GBIF)",
                        "T3": "Hábitat botánico (Manual + Unidades Fitogeográficas + GBIF)",
                    }.get(tier, "Imagen 1")
                    _img2_line = ""
                    if tier == "T3" and manual_image_path:
                        _img2_line = f"\n- Imagen 2       : {os.path.basename(str(manual_image_path))} — Modelo predictivo RF"
                    _rf_line = ""
                    if tier == "T3":
                        _rf_line = f"\n- Modelo RF      : Random Forest (AUC: {rf_auc:.4f})"
                    _fuentes_line = (
                        f"- Imagen 1       : {os.path.basename(str(image_path))} — {_img1_label}"
                        f"{_img2_line}{_rf_line}"
                        f"\n- Presencias     : Registros GBIF limpios (Mesoamérica para RF, CR para mapa)"
                    )
                _modalidad = "Texto únicamente" if tier == "T0" else "Visión + Texto"
                encabezado_metadatos = f"""================================================================================
METADATOS DEL EXPERIMENTO (Arquitectura CR-BioLM — Tier {tier})
================================================================================
Modelo LLM       : {modelo_a_usar} ({_modalidad})
Especie          : {species_name}
Pregunta Usuario : {pregunta_texto}

FUENTES DE DATOS PROPORCIONADAS:
{_fuentes_line}
================================================================================

[ANÁLISIS HÍBRIDO GENERADO POR IA]
"""
                modelo_limpio = modelo_a_usar.replace('/', '_').replace('-', '_').replace(':', '_')
                ruta_salida_txt = os.path.join(output_dir, f"llm_profile_BIMODAL_{modelo_limpio}.txt")
                
                with open(ruta_salida_txt, "w", encoding="utf-8") as file:
                    file.write(encabezado_metadatos)
                    file.write(perfil_texto)
                    
                print(f"[EXITO] Perfil Multimodal guardado en: {ruta_salida_txt}")
                return True
            else:
                print(f"[ERROR] Falló la API de OpenRouter: {respuesta.text}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Falló la inferencia multimodal: {e}")
            return False