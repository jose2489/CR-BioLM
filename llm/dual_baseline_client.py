import os
from groq import Groq
from dotenv import load_dotenv

class DualBaselineAnalyst:
    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("[ERROR] No se encontró GROQ_API_KEY en el archivo .env.")

        self.client = Groq(api_key=api_key)

    def generate_profile(self, species_name, image_path, output_dir, user_question=None):
        print("[INFO] Ejecutando Baseline Dual (Texto + Inferencia espacial)...")

        if not os.path.exists(image_path):
            print(f"[ERROR] No se encontró el mapa en {image_path} para el baseline.")
            return None

        try:
            # ==========================================================
            # AGENTE 1: "VISIÓN SIMULADA" (DESCRIPCIÓN ESPACIAL)
            # ==========================================================
            print("  -> Agente 1 (pseudo-visión) infiriendo patrón espacial...")

            vision_prompt = f"""
            Eres un geógrafo experto en Costa Rica.

            Aunque no puedes ver la imagen directamente, asume que estás analizando
            un mapa de distribución de la especie {species_name} basado en registros reales.

            Describe de forma plausible:
            - Regiones del país donde se encontraría
            - Posibles cordilleras o elevaciones
            - Patrones típicos de distribución (ej: montaña, bosque nuboso, etc.)

            Mantén el análisis realista y conservador.
            """

            respuesta_vision = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": vision_prompt}],
                temperature=0.3
            ).choices[0].message.content

            # ==========================================================
            # AGENTE 2: SÍNTESIS ECOLÓGICA
            # ==========================================================
            print("  -> Agente 2 (texto) generando perfil ecológico...")

            instruccion_pregunta = ""
            if user_question:
                instruccion_pregunta = f"""
                PREGUNTA DEL USUARIO:
                {user_question}

                Debes responder explícitamente esta pregunta.
                """

            sintesis_prompt = f"""
            Eres un ecólogo tropical.

            Un analista espacial reporta lo siguiente sobre la especie {species_name}:
            "{respuesta_vision}"

            Redacta un perfil ecológico formal, coherente y científicamente plausible.

            {instruccion_pregunta}
            """

            perfil_final = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": sintesis_prompt}],
                temperature=0.2
            ).choices[0].message.content

            print("\n================ BASELINE DUAL OUTPUT ================\n")

            print("--- INFERENCIA ESPACIAL (Agente 1) ---\n")
            print(respuesta_vision)

            print("\n--- PERFIL ECOLÓGICO (Agente 2) ---\n")
            print(perfil_final)

            print("\n======================================================\n")

            # ==========================================================
            # METADATOS
            # ==========================================================
            pregunta_texto = user_question if user_question else "Análisis general (sin pregunta específica)"
            modelo_nombre = "Baseline Dual (Texto + Inferencia, Groq)"

            encabezado = f"""================================================================================
METADATOS DEL EXPERIMENTO (BASELINE)
================================================================================
Modelo LLM       : {modelo_nombre}
Especie          : {species_name}
Pregunta Usuario : {pregunta_texto}

NOTA:
- Este baseline NO utiliza datos estructurados ni variables ambientales.
- No usa Random Forest, SHAP ni contexto geoespacial.
- La inferencia es generativa y no guiada por datos.
================================================================================

[ANÁLISIS GENERADO POR IA]
--- INFERENCIA ESPACIAL (Agente 1) ---
{respuesta_vision}

--- PERFIL ECOLÓGICO (Agente 2) ---
{perfil_final}
"""

            # ==========================================================
            # GUARDAR
            # ==========================================================
            file_path = os.path.join(output_dir, "llm_profile_BASELINE_DUAL.txt")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(encabezado)

            print(f"[INFO] Baseline guardado en: {file_path}")

            return perfil_final

        except Exception as e:
            print(f"[ERROR] Falló el Baseline Dual: {e}")
            return None