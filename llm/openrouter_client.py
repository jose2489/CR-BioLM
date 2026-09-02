import io
import os
import base64
import requests
from PIL import Image
# IMPORTANTE: Asegúrate de importar la función traducir_variable desde tu archivo de templates
from .prompt_templates import PROMPT_T0, PROMPT_T1, PROMPT_T3, _REGLA_STRICTA, traducir_variable, get_effective_prompts
from . import providers as _providers

# T2 removed — ficha leakage fix (texto_manual reached ground truth used in evaluation)
# Base fallback — actual prompts loaded at call time via get_effective_prompts()
TIER_PROMPTS = {"T0": PROMPT_T0, "T1": PROMPT_T1, "T3": PROMPT_T3}

MAX_IMG_PX = 1024  # max pixels on the longest edge before base64 encoding

class OpenRouterClient:
    """Cliente multimodal para cualquier endpoint compatible con OpenAI.

    A pesar del nombre (conservado para no romper los call sites existentes), el
    cliente resuelve el proveedor a partir del *model spec* mediante
    ``llm.providers``: acepta tanto modelos alojados en OpenRouter como modelos
    locales servidos por Ollama, con el MISMO payload (texto + ``image_url`` en
    base64).

        "openrouter:openai/gpt-4o"
        "ollama:qwen3-vl:8b-instruct"
        "openai/gpt-4o"                  -> OpenRouter (compatibilidad)
    """

    def __init__(self, api_key=None, model="openai/gpt-4o"):
        self.api_key = api_key
        self.model = model
        # Conservado por compatibilidad; la URL real se resuelve por spec en cada
        # llamada (ver llm/providers.py).
        self.url = _providers.resolve(model, api_key=api_key).url

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
        
        spec = model_override if model_override else self.model
        ok, _motivo = _providers.is_configured(spec, api_key=self.api_key)
        if not ok:
            print(f"[ERROR] Proveedor no configurado para '{spec}': {_motivo}")
            return False
        prov = _providers.resolve(spec, api_key=self.api_key)
        modelo_a_usar = prov.model
        print(f"[INFO] Preparando síntesis bimodal — proveedor={prov.name} | "
              f"modelo={prov.model} | {'LOCAL' if prov.is_local else 'API remota'}")

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

        # Seleccionar prompt según tier — carga overrides en runtime si existen
        _effective = get_effective_prompts()
        _regla_efectiva = _effective.get("_REGLA_STRICTA", _REGLA_STRICTA)
        template = _effective.get(tier) or TIER_PROMPTS.get(tier, PROMPT_T3)
        print(f"[INFO] Usando prompt {tier}")

        # T1 — solo imagen GBIF; T3 — mapa hábitat predicho + mapa RF + métricas SHAP/AUC
        format_kwargs = dict(
            species_name=species_name,
            instruccion_pregunta=f"PREGUNTA DEL USUARIO: {pregunta_texto}",
            _regla=_regla_efectiva,
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

        # El prompt T3 describe TRES fuentes, incluida la "Imagen 2" (mapa RF). Si esa
        # imagen no se adjunta, el modelo igual la da por vista y alucina una
        # comparación entre mapas (observado con qwen3-vl, 2026-09-01). Avisarlo de
        # forma explícita en vez de dejar que el prompt prometa algo que no llega.
        _tiene_img2 = bool(manual_image_path and os.path.isfile(str(manual_image_path)))
        if tier == "T3" and not _tiene_img2:
            prompt_listo += (
                "\n\nAVISO IMPORTANTE: la Imagen 2 (mapa predictivo climático RF) NO se "
                "adjuntó en esta llamada. Solo recibes la Imagen 1. NO afirmes nada sobre "
                "la Imagen 2 ni la compares con la Imagen 1; indica explícitamente que no "
                "está disponible."
            )

        # ==========================================================
        # 3. CODIFICAR IMAGEN(ES) Y ARMAR PAYLOAD PARA LA API
        # T0 es texto únicamente — no se envían imágenes
        # ==========================================================
        if tier == "T0":
            content_parts = [{"type": "text", "text": prompt_listo}]
            print("[INFO] T0: llamada texto-únicamente (sin imágenes)")
        else:
            if not image_path or not os.path.isfile(str(image_path)):
                print(f"[ERROR] Imagen principal no encontrada para tier {tier}: {image_path!r}")
                return False
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

        headers = prov.headers()

        payload = {
            "model": prov.model,
            "max_tokens": 4096,
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
            print(f"[LLM] Solicitando análisis a {prov.name} ({prov.url})...")
            # Los modelos locales son bastante más lentos (~18 s en caliente, ~47 s en
            # frío por la carga del modelo), de ahí el timeout más generoso.
            respuesta = requests.post(prov.url, headers=headers, json=payload,
                                      timeout=600 if prov.is_local else 180)
            
            if respuesta.status_code == 200:
                perfil_texto = respuesta.json()['choices'][0]['message']['content']
                
                # Metadata header: tier-aware labels
                if tier == "T0":
                    _fuentes_line = "- Fuentes        : Conocimiento de entrenamiento del LLM (sin mapas, sin datos instrumentales)"
                else:
                    _img1_label = {
                        "T1": "Distribución GBIF Mesoamérica (puntos de presencia)",
                        "T3": "Mapa de hábitat predicho (Manual + Regiones Botánicas + DEM + GBIF)",
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
Proveedor        : {prov.name} ({'local, pesos abiertos' if prov.is_local else 'API remota'})
Especie          : {species_name}
Pregunta Usuario : {pregunta_texto}

FUENTES DE DATOS PROPORCIONADAS:
{_fuentes_line}
================================================================================

[ANÁLISIS HÍBRIDO GENERADO POR IA]
"""
                # Incluye el proveedor en el nombre para que un mismo modelo servido
                # local y vía API no se pisen entre sí.
                modelo_limpio = _providers.slug(prov.spec)
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