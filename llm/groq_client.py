import os
from groq import Groq
from .base_llm import BaseLLM
from .prompt_templates import build_ecological_prompt
import config

class GroqAnalyst(BaseLLM):
    """
    Cliente de Groq modularizado. Permite la inyeccion dinamica de diferentes 
    modelos (Llama, Mixtral, Gemma) y delega la construccion del texto 
    al gestor de prompts.
    """
    def __init__(self, api_key=None, default_model="llama-3.3-70b-versatile"):
            # Le decimos que busque en config.GROQ_API_KEY como plan de respaldo
            self.api_key = api_key or os.environ.get("GROQ_API_KEY") or config.GROQ_API_KEY
            
            if not self.api_key or self.api_key == "API_KEY_SI_NO_ESTA_EN_ENV":
                raise ValueError("[ERROR] No se encontro GROQ_API_KEY. Configurala en config.py o en el entorno.")
                
            self.client = Groq(api_key=self.api_key)
            self.default_model = default_model

    def generate_profile(self, species_name, rf_metrics, shap_dict, output_dir, area_km2=None, user_question=None, model_override=None, contexto_conservacion=None):
        modelo_a_usar = model_override if model_override else self.default_model
        print(f"[INFO] Preparando síntesis para Groq ({modelo_a_usar})...")
        
        # 1. Construimos el prompt inyectando métricas, área y el contexto de conservación
        prompt = build_ecological_prompt(
            species_name=species_name, 
            rf_metrics=rf_metrics, 
            shap_dict=shap_dict, 
            area_km2=area_km2,
            user_question=user_question,
            contexto_conservacion=contexto_conservacion 
        )
        
        try:
            print(f"[INFO] Solicitando síntesis a {modelo_a_usar}...")
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "Eres un científico botanico encargado de redactar la discusión de resultados para el proyecto CR-BioLM. Tu lenguaje es académico, formal y directo."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                model=modelo_a_usar,
                temperature=0.2 # Ligeramente más alto que 0.1 para que la redacción fluya mejor, pero muy preciso.
            )
            
            perfil_texto = chat_completion.choices[0].message.content.strip()
            
            # =====================================================================
            # 2. CONSTRUCCIÓN DEL ENCABEZADO DE METADATOS (TRAZABILIDAD)
            # =====================================================================
            pregunta_texto = user_question if user_question else "Análisis general de la especie (Sin pregunta específica)"
            
            encabezado_metadatos = f"""================================================================================
METADATOS DEL EXPERIMENTO (Arquitectura CR-BioLM)
================================================================================
Modelo LLM       : {modelo_a_usar}
Especie          : {species_name}
Pregunta Usuario : {pregunta_texto}

FUENTES DE DATOS Y VARIABLES ESPACIALES INTEGRADAS:
- Presencias     : Registros limpios de GBIF (Global Biodiversity Information Facility).
- Mapa Experto   : Polígonos base de BIEN / Map of Life.
- Clima          : WorldClim V2.1 (Variables bioclimáticas continuas bio_1 a bio_19).
- Topografía     : WorldClim V2.1 (Modelo de Elevación Digital continuo - DEM).
- Conservación   : Geoportal SNIT / SINAC Costa Rica (Shapefile vectorial de Áreas Protegidas EPSG:4326).
================================================================================

[ANÁLISIS GENERADO POR IA]
"""
            
            # =====================================================================
            # 3. GUARDADO DE RESULTADOS
            # =====================================================================
            modelo_limpio = modelo_a_usar.replace('-', '_').replace('/', '_')
            nombre_archivo = f"llm_profile_{modelo_limpio}.txt"
            file_path = os.path.join(output_dir, nombre_archivo)
            
            with open(file_path, "w", encoding="utf-8") as file:
                # Escribimos el nuevo encabezado gigante seguido de la respuesta del LLM
                file.write(encabezado_metadatos)
                file.write(perfil_texto)
                
            return perfil_texto
            
        except Exception as e:
            print(f"[ERROR] Fallo la inferencia con {modelo_a_usar}: {e}")
            return None