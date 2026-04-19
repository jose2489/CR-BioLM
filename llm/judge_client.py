# llm/judge_client.py
#
# Módulo de evaluación automática (LLM-as-Judge) para CR-BioLM.
# Juez: google/gemini-2.0-flash-001 — familia distinta a los generadores,
# eliminando self-preference bias por diseño (Zheng et al., 2023).
#
# Rúbrica basada en G-Eval (Liu et al., 2023) con "Conditional Path Evaluation"
# (inspirado en MT-Bench-101 y "Rubric Is All You Need", 2025):
#
#   M5 — Profundidad analítica   (0–3)         todos los tiers (M5 primero)
#   M1 — Precisión geográfica    (0–3)         todos los tiers
#   M3 — Relevancia de respuesta (0–3)         todos los tiers
#   M2 — Uso del contexto altitud(0–2 / N/A)   N/A para T0/T1/T2; bonus para T3
#   M4 — Variable climática SHAP (0–2 / N/A)   N/A para T0/T1/T2; bonus para T3
#
# Fórmula de score compuesto:
#   T0/T1/T2 base            : (M1 + M3 + M5) / 9
#   T3 sin bonuses           : (M1 + M3 + M5) / 9
#   T3 + M2 solamente        : (M1 + M3 + M5 + M2) / 11
#   T3 + M4 solamente        : (M1 + M3 + M5 + M4) / 11
#   T3 + M2 + M4 (completo)  : (M1 + M3 + M5 + M2 + M4) / 13
#
# Principio clave: M2 y M4 son BONUS opcionales para T3, no penalizaciones.
# T0/T1/T2 no tienen el dato → N/A (excluidos del denominador).
# T3 tiene el dato (puede ser ruidoso) → recompensa por usarlo bien,
# 0 si no lo usa, penalización solo si contradice AMBAS fuentes (alucinación).
#
# M5 (Profundidad analítica) — mide calidad de razonamiento independiente de
# la exactitud factual. Evalúa PRIMERO para que no se vea influenciado por M1.
# En T0: una negativa bien calibrada puede obtener M5=3.

import os
import json
import requests

JUDGE_MODEL = "google/gemini-2.0-flash-001"

JUDGE_PROMPT = """Eres un evaluador experto en botánica costarricense y ecología de plantas tropicales.
Se te proporciona una PREGUNTA, una RESPUESTA generada por un sistema de IA, y una FICHA DE REFERENCIA
del Manual de Plantas de Costa Rica (ground truth de expertos).

Evalúa en CINCO dimensiones. Sigue el orden M5 → M1 → M3 → M2 → M4.
Para cada dimensión: cita textualmente el fragmento de la respuesta que fundamenta tu puntuación
(campo "cita_*"), luego explica brevemente tu razonamiento (campo "razonamiento_*"), luego asigna el número.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXTO DEL SISTEMA — TIER: {tier}
  T0 = Sin contexto (conocimiento previo del LLM únicamente — sin mapas, sin datos)
  T1 = Solo mapa GBIF Mesoamérica          → sin datos de altitud, sin RF/SHAP
  T2 = Mapa hábitat botánico + GBIF         → sin datos de altitud, sin RF/SHAP
  T3 = Ambos mapas + RF (AUC/SHAP) + altitud GBIF + altitud del Manual

PREGUNTA:
{pregunta}

RESPUESTA A EVALUAR:
{respuesta}

FICHA DE REFERENCIA (Manual de Plantas de Costa Rica):
{ficha_mdp}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DIMENSIONES DE EVALUACIÓN (en este orden):

──────────────────────────────────────
M5 — PROFUNDIDAD ANALÍTICA (0–3)  [aplica a TODOS los tiers — evaluar PRIMERO]
Mide la calidad del razonamiento, independientemente de la exactitud factual.
Longitud ≠ profundidad. Una respuesta corta con razonamiento sólido puede obtener M5=3.
Una recitación larga de datos sin interpretación obtiene M5=0.

  0 = Superficial: recita o parafrasea los datos proporcionados sin interpretación.
      No aborda limitaciones, tensiones entre fuentes ni incertidumbre.
      Para T0: el sistema produce afirmaciones confiadas sin reconocer la falta de datos.

  1 = Mínimo: menciona al menos una limitación, incertidumbre o calificación
      (ej.: "los registros GBIF pueden incluir ocurrencias fuera del rango natural").
      No integra fuentes ni reconcilia tensiones.
      Para T0: reconoce la ausencia de datos y se muestra cauteloso, pero da algún contexto general útil.

  2 = Reconciliador: identifica y trata de reconciliar tensiones entre fuentes,
      o señala discrepancias explícitas entre lo observado y lo esperado.
      Para T0: declara explícitamente que no puede responder con precisión sin datos
      adicionales Y describe qué información específica resolvería la pregunta.

  3 = Integrativo: sintetiza múltiples fuentes de evidencia críticamente,
      distingue afirmaciones respaldadas de especulativas, y enmarca conclusiones
      con cobertura epistémica apropiada.
      Para T0: negativa bien calibrada O respuesta general bien acotada que un autor
      del Manual podría avalar, incluyendo explícitamente qué no se puede saber sin datos.

──────────────────────────────────────
M1 — PRECISIÓN GEOGRÁFICA (0–3)  [aplica a todos los tiers]
¿Las zonas geográficas (vertiente, cordillera, región, cantón, parque) son consistentes con el Manual?

  Para T0: si la respuesta es una negativa calibrada, evalúa si la negativa es CORRECTA
  (identifica correctamente que no puede determinar la distribución específica sin datos).
  Una negativa correcta y honesta obtiene M1=2. Una negativa que afirma erróneamente
  que "la especie está en todo Costa Rica" obtiene M1=0.

  0 = Zonas incorrectas, contradicen el Manual, o afirmaciones geográficas fabricadas
  1 = Una zona correcta, incompleta o con errores notables
  2 = Zonas correctas pero sin especificidad suficiente, o negativa T0 correcta
  3 = Zonas completas, específicas y consistentes con el Manual

──────────────────────────────────────
M3 — RELEVANCIA DE RESPUESTA (0–3)  [aplica a todos los tiers]
¿La respuesta contesta directamente la pregunta con información específica de la especie?

  Para T0: una negativa que explica POR QUÉ no puede responder es más relevante
  que una que simplemente dice "no sé". Una negativa informativa puede obtener M3=2–3.

  0 = No responde la pregunta o es completamente genérica
  1 = Responde parcialmente, sin especificidad suficiente
  2 = Responde la pregunta con digresiones o parcialmente
  3 = Responde directo, conciso, y con datos específicos de la especie (o, para T0:
      negativa directa que especifica qué datos resolverían exactamente la pregunta)

──────────────────────────────────────
M2 — USO DEL CONTEXTO ALTITUDINAL (0–2 o "N/A")  [BONUS para T3 — N/A obligatorio para T0/T1/T2]

▸ Si TIER = T0, T1 o T2: asigna "N/A" siempre. El sistema no proporcionó datos de altitud.
▸ Si la ficha de referencia no contiene datos de altitud: asigna "N/A".
▸ Si TIER = T3: evalúa si el modelo usó bien el contexto altitudinal que SÍ recibió.
  Esta métrica es un BONUS — reconoce el uso correcto de información adicional.

  Para T3:
  0 = No menciona altitud a pesar de haberla recibido, O reporta valores que
      contradicen TANTO el rango GBIF proporcionado COMO el Manual (alucinación)
  1 = Menciona altitud pero con imprecisión notable (error > 400 m vs Manual),
      o usa el dato GBIF sin reconocer posible ruido
  2 = Usa el dato altitudinal de forma responsable: cita el rango disponible,
      lo compara con el Manual o reconoce posible ruido GBIF, y la estimación
      final es consistente con el Manual (error ≤ 400 m)

──────────────────────────────────────
M4 — USO DEL CONTEXTO CLIMÁTICO/RF (0–2 o "N/A")  [BONUS para T3 — N/A obligatorio para T0/T1/T2]

▸ Si TIER = T0, T1 o T2: asigna "N/A" siempre. No se proporcionaron métricas RF ni SHAP.
▸ Si TIER = T3: evalúa si el modelo usó bien las métricas RF/SHAP que recibió.

  Para T3:
  0 = No identifica ninguna variable climática a pesar de tener datos SHAP, O
      menciona una variable que contradice el hábitat descrito en el Manual
  1 = Identifica una variable climática plausible pero no la más relevante
      según el hábitat del Manual, o sin explicación mecanística
  2 = Identifica correctamente la variable más limitante, consistente con el
      hábitat del Manual, y explica el mecanismo ecológico

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EJEMPLOS DE PUNTUACIÓN M5 (orientación):

[T0 — M5=3] "No puedo determinar el rango altitudinal exacto de {{especie}} sin acceso a datos
de campo o literatura especializada. Para responder con precisión serían necesarios registros
georreferenciados con altitud o el Manual de Plantas de Costa Rica. Puedo decir que la familia
Melastomataceae tiende a ocupar bosques montanos húmedos, pero no puedo confirmar rangos para
esta especie en particular."
→ M5=3: negativa calibrada con familia de contexto general honesto, identifica qué datos faltan.

[T1 — M5=0] "Los puntos GBIF muestran presencia en la Cordillera de Talamanca, Cordillera Central
y vertiente Caribe. La especie prefiere elevaciones medias. Se distribuye ampliamente en Costa Rica."
→ M5=0: recitación de datos visibles sin interpretación; no reconcilia nada, no señala limitaciones.

[T2 — M5=2] "El mapa muestra hábitat óptimo (cyan) en la Cordillera Central, pero los puntos GBIF
aparecen tanto en zonas cyan como en zonas muted, sugiriendo que la especie tolera condiciones
subóptimas. Esta discrepancia puede reflejar plasticidad ecológica o errores en los registros."
→ M5=2: identifica tensión entre hábitat óptimo y registros reales, propone interpretación.

[T3 — M5=3] "El modelo (AUC=0.89) y el Manual concuerdan en la Cordillera Central como núcleo
de distribución. Sin embargo, la variable limitante (bio_19, precipitación trimestre más seco)
tiene impacto positivo, sugiriendo que la especie requiere humedad mínima estacional. Esto es
coherente con el hábitat de bosque pluvial del Manual pero implica vulnerabilidad ante sequías
proyectadas. Los puntos GBIF en zonas de baja idoneidad predicha podrían ser registros históricos
o errores de georreferenciación."
→ M5=3: síntesis crítica de tres fuentes, distinción respaldado/especulativo, cobertura epistémica.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANTE:
- Evalúa M5 ANTES de M1 para evitar que la exactitud factual sesga la profundidad analítica.
- No favorezcas respuestas más largas. Evalúa solo calidad del razonamiento y precisión.
- Para M2 y M4 en T3: el intento de usar el contexto adicional (aunque con ruido)
  ya es meritorio. Solo penaliza con 0 si hay alucinación o ausencia total del uso del dato.
- Responde EXCLUSIVAMENTE en JSON (sin texto fuera del JSON):

{{
  "cita_M5": "<fragmento textual de la respuesta que sustenta M5>",
  "razonamiento_M5": "<razonamiento breve>",
  "M5_profundidad_analitica": <0|1|2|3>,
  "cita_M1": "<fragmento textual de la respuesta que sustenta M1>",
  "razonamiento_M1": "<razonamiento breve>",
  "M1_precision_geografica": <0|1|2|3>,
  "cita_M3": "<fragmento textual de la respuesta que sustenta M3>",
  "razonamiento_M3": "<razonamiento breve>",
  "M3_relevancia_respuesta": <0|1|2|3>,
  "razonamiento_M2": "<razonamiento breve>",
  "M2_precision_altitudinal": <0|1|2|"N/A">,
  "razonamiento_M4": "<razonamiento breve>",
  "M4_variable_climatica": <0|1|2|"N/A">
}}
"""


def _score_compuesto(scores: dict, perfil: str, tier: str = "T3") -> float:
    """
    Calcula el score compuesto normalizado [0, 1].

    Diseño de Conditional Path Evaluation con M5:
    - M1, M3, M5 son universales (todos los tiers T0/T1/T2/T3, siempre en denominador).
    - M2 y M4 son BONUS para T3:
        * T0/T1/T2 → N/A, excluidos del denominador
        * T3        → incluidos en denominador, recompensan uso correcto del contexto extra

    Fórmula:
        T0/T1/T2 base     : (M1+M3+M5) / 9
        T3 sin bonuses    : (M1+M3+M5) / 9
        T3 + M2 solo      : (M1+M3+M5+M2) / 11
        T3 + M4 solo      : (M1+M3+M5+M4) / 11
        T3 + M2 + M4      : (M1+M3+M5+M2+M4) / 13
    """
    m1 = scores.get("M1_precision_geografica", 0)
    m2 = scores.get("M2_precision_altitudinal", "N/A")
    m3 = scores.get("M3_relevancia_respuesta", 0)
    m4 = scores.get("M4_variable_climatica", "N/A")
    m5 = scores.get("M5_profundidad_analitica", 0)

    total = (
        (m1 if isinstance(m1, (int, float)) else 0) +
        (m3 if isinstance(m3, (int, float)) else 0) +
        (m5 if isinstance(m5, (int, float)) else 0)
    )
    denom = 9  # M1(3) + M3(3) + M5(3)

    # M2: bonus solo si T3 y la ficha tiene datos de altitud (m2 != "N/A")
    if tier == "T3" and m2 != "N/A":
        total += (m2 if isinstance(m2, (int, float)) else 0)
        denom += 2

    # M4: bonus solo si T3 y perfil botánico y pregunta climática (m4 != "N/A")
    if tier == "T3" and perfil == "botanico" and m4 != "N/A":
        total += (m4 if isinstance(m4, (int, float)) else 0)
        denom += 2

    return round(total / denom, 4) if denom > 0 else 0.0


class JudgeClient:
    """
    Cliente Gemini-as-Judge para evaluar respuestas del pipeline CR-BioLM.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = JUDGE_MODEL

    def evaluar(
        self,
        pregunta: str,
        respuesta: str,
        ficha_mdp: str,
        perfil: str,           # "turista" | "botanico" | "municipalidad"
        especie: str,
        tier: str,             # "T0" | "T1" | "T2" | "T3"
        modelo_generador: str,
        output_dir: str = None,
    ) -> dict | None:
        """
        Evalúa una respuesta contra el ground truth (ficha MdP).
        Retorna dict con scores y score compuesto, o None si falla.
        """
        prompt = JUDGE_PROMPT.format(
            tier=tier,
            pregunta=pregunta,
            respuesta=respuesta,
            ficha_mdp=ficha_mdp,
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }

        try:
            resp = requests.post(self.url, headers=headers, json=payload, timeout=60)
            if resp.status_code != 200:
                print(f"[JUDGE ERROR] HTTP {resp.status_code}: {resp.text[:200]}")
                return None

            content = resp.json()["choices"][0]["message"]["content"].strip()

            # Limpiar posibles bloques de código markdown
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            scores = json.loads(content)
            scores["score_compuesto"] = _score_compuesto(scores, perfil, tier)
            scores["especie"]          = especie
            scores["tier"]             = tier
            scores["perfil"]           = perfil
            scores["modelo_generador"] = modelo_generador
            scores["modelo_juez"]      = self.model

            if output_dir:
                _guardar_evaluacion(scores, output_dir, tier, modelo_generador, perfil)

            return scores

        except json.JSONDecodeError as e:
            print(f"[JUDGE ERROR] No se pudo parsear JSON del juez: {e}")
            print(f"  Contenido recibido: {content[:300]}")
            return None
        except Exception as e:
            print(f"[JUDGE ERROR] {e}")
            return None


def _guardar_evaluacion(scores: dict, output_dir: str, tier: str, modelo: str, perfil: str):
    """Guarda el resultado de evaluación como JSON en el directorio de la especie."""
    modelo_limpio = modelo.replace("/", "_").replace("-", "_")
    nombre = f"eval_{tier}_{modelo_limpio}_{perfil}.json"
    ruta = os.path.join(output_dir, nombre)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print(f"[JUDGE] Evaluación guardada: {ruta}")


def ficha_summary(ficha_texto: str) -> str:
    """
    Extrae un resumen compacto de la ficha MdP para el juez.
    Solo incluye: distribución geográfica, rango altitudinal y tipo de hábitat.
    Reduce tokens del juez sin perder la información necesaria para M1/M3/M5.
    """
    lines = ficha_texto.splitlines()
    relevant = []
    capture = False
    target_sections = {
        "distribución geográfica", "rango altitudinal",
        "tipo de hábitat", "habitat", "altitud",
    }

    i = 0
    while i < len(lines):
        line = lines[i]
        line_lower = line.lower().strip().rstrip(":")

        # Detect section headers (lines ending with : or containing key terms)
        if any(sec in line_lower for sec in target_sections):
            capture = True
            relevant.append(line)
            i += 1
            # Capture following content lines until blank or next header
            while i < len(lines) and lines[i].strip():
                relevant.append(lines[i])
                i += 1
            relevant.append("")
            capture = False
        else:
            i += 1

    if not relevant:
        # Fallback: return first 400 chars
        return ficha_texto[:400]

    return "\n".join(relevant).strip()


def evaluar_directorio(output_dir: str, ficha_mdp: str, pregunta: str,
                       perfil: str, api_key: str) -> list[dict]:
    """
    Evalúa todos los archivos llm_profile_BIMODAL_*.txt encontrados en output_dir.
    Útil para post-procesar un run ya completado.
    """
    import glob
    judge = JudgeClient(api_key)
    resultados = []

    archivos = glob.glob(os.path.join(output_dir, "llm_profile_BIMODAL_*.txt"))
    for archivo in archivos:
        nombre = os.path.basename(archivo)

        # Extraer tier y modelo del nombre del directorio/archivo
        tier = "T3"  # default; override si hay convención en el path
        modelo_gen = nombre.replace("llm_profile_BIMODAL_", "").replace(".txt", "")

        with open(archivo, encoding="utf-8") as f:
            contenido = f.read()

        # Extraer solo la sección de análisis (después del encabezado)
        if "[ANÁLISIS HÍBRIDO GENERADO POR IA]" in contenido:
            respuesta = contenido.split("[ANÁLISIS HÍBRIDO GENERADO POR IA]")[1].strip()
        else:
            respuesta = contenido

        especie = os.path.basename(os.path.dirname(os.path.dirname(output_dir)))

        resultado = judge.evaluar(
            pregunta=pregunta,
            respuesta=respuesta,
            ficha_mdp=ficha_mdp,
            perfil=perfil,
            especie=especie,
            tier=tier,
            modelo_generador=modelo_gen,
            output_dir=output_dir,
        )
        if resultado:
            resultados.append(resultado)

    return resultados
