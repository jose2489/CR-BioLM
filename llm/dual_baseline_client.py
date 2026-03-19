import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

class DualBaselineAnalyst:
    def __init__(self):
        load_dotenv()
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("[ERROR] No se encontró OPENROUTER_API_KEY en el archivo .env.")
            
        # Configuramos el cliente para usar OpenRouter
        self.client_vision = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        
        # Necesitamos el cliente de Groq para el Agente 2 (Texto)
        import config
        self.client_texto = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_profile(self, species_name, image_path, output_dir, user_question=None):
        print("[INFO] Ejecutando Baseline Dual (Visión + Texto)...")
        
        if not os.path.exists(image_path):
            print(f"[ERROR] No se encontró el mapa en {image_path} para el baseline.")
            return None

        base64_image = self.encode_image(image_path)

        # --- AGENTE 1: Extracción Visual (Vía OpenRouter) ---
        print("  -> Agente 1 (Visión) leyendo el mapa crudo...")
        vision_prompt = f"Eres un geógrafo experto en Costa Rica. Observa este mapa de distribución. Menciona detalladamente en qué regiones, cordilleras, elevaciones aproximadas o provincias se agrupan los puntos (círculos rojos) de la especie {species_name}."
        
        try:
            respuesta_vision = self.client_vision.chat.completions.create(
                model="openai/gpt-4o", # Modelo de visión  en OpenRouter
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                temperature=0.2
            ).choices[0].message.content

            # --- AGENTE 2: Síntesis y Respuesta (Vía Groq) ---
            print("  -> Agente 2 (Texto) respondiendo la pregunta del usuario...")
            
            instruccion_pregunta = ""
            if user_question:
                instruccion_pregunta = f"\nPREGUNTA DEL USUARIO A RESPONDER: '{user_question}'\nTu objetivo principal es responder a esta pregunta basándote en la ubicación geográfica descubierta."

            sintesis_prompt = f"""Eres un ecólogo tropical. Un analista espacial te reporta que la especie {species_name} se distribuye geográficamente así: 
'{respuesta_vision}'

Redacta un perfil ecológico formal sobre esta especie.{instruccion_pregunta}"""

            perfil_final = self.client_texto.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": sintesis_prompt}],
                temperature=0.2
            ).choices[0].message.content
            
            # =====================================================================
            # CONSTRUCCIÓN DEL ENCABEZADO DE METADATOS (TRAZABILIDAD)
            # =====================================================================
            pregunta_texto = user_question if user_question else "Análisis general del nicho ecológico (Sin pregunta específica)"
            modelo_nombre = "Baseline Dual (Visión OpenRouter + Texto Groq)"

            encabezado_metadatos = f"""================================================================================
METADATOS DEL EXPERIMENTO (Arquitectura CR-BioLM)
================================================================================
Modelo LLM       : {modelo_nombre}
Especie          : {species_name}
Pregunta Usuario : {pregunta_texto}

FUENTES DE DATOS Y VARIABLES ESPACIALES INTEGRADAS:
- Presencias     : Registros limpios de GBIF (Global Biodiversity Information Facility).
- Mapa Experto   : Polígonos base de BIEN / Map of Life.
- Clima          : WorldClim V2.1 (Variables bioclimáticas continuas bio_1 a bio_19).
- Topografía     : WorldClim V2.1 (Modelo de Elevación Digital continuo - DEM).
- Conservación   : Geoportal SNIT / SINAC Costa Rica (Shapefile vectorial de Áreas Protegidas EPSG:4326).
- Nota Metodológica: El Baseline infiere visualmente desde el mapa estático, sin recibir la matriz del Random Forest.
================================================================================

[ANÁLISIS GENERADO POR IA]
--- DIAGNÓSTICO VISUAL ESTÁTICO (Agente 1) ---
{respuesta_vision}

--- RESPUESTA FINAL ECOLÓGICA (Agente 2) ---
{perfil_final}
"""
            
            # =====================================================================
            # GUARDAR RESULTADO
            # =====================================================================
            nombre_archivo = "llm_profile_BASELINE_DUAL.txt"
            file_path = os.path.join(output_dir, nombre_archivo)
            
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(encabezado_metadatos)
                
            return perfil_final

        except Exception as e:
            print(f"[ERROR] Falló el Baseline Dual: {e}")
            return None