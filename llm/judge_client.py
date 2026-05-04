# llm/judge_client.py
#
# Módulo de evaluación automática (LLM-as-Judge) para CR-BioLM.
#
# Arquitectura: ensemble de 2 jueces con tie-breaker Groq.
#   Judge A: google/gemini-2.0-flash-001 via OpenRouter
#   Judge B: openai/gpt-4o-mini via OpenRouter
#   Judge C: meta-llama/llama-3.3-70b-versatile via Groq (tie-breaker, gratis)
#
# Reglas:
#   - A y B puntúan cada respuesta de forma independiente.
#   - Self-enhancement guard: si la familia del generador coincide con un juez, se reemplaza por C.
#   - Score final por métrica: media de A y B (o A y C / B y C si se aplicó el guard).
#   - Disagreement flag: |A − B| >= 2 en cualquier métrica → invocar C + flag para revisión humana.
#   - Todos los scores individuales se persisten en eval_*.json para auditoría.
#
# Rúbrica basada en G-Eval (Liu et al., 2023) con "Conditional Path Evaluation":
#   M5 — Profundidad analítica   (0–3)         todos los tiers (M5 primero)
#   M1 — Precisión geográfica    (0–3)         todos los tiers
#   M3 — Relevancia de respuesta (0–3)         todos los tiers
#   M2 — Uso del contexto altitud(0–2 / N/A)   N/A para T0/T1; bonus para T3
#   M4 — Variable climática SHAP (0–2 / N/A)   N/A para T0/T1; bonus para T3
#
# Score compuesto (ver _score_compuesto):
#   T0/T1 base           : (M1 + M3 + M5) / 9
#   T3 + M2              : (M1 + M3 + M5 + M2) / 11
#   T3 + M4 (botanico)   : (M1 + M3 + M5 + M4) / 11
#   T3 + M2 + M4         : (M1 + M3 + M5 + M2 + M4) / 13
#
# Caps (stackable, applied as min):
#   D1 taxonomy cap: taxonomy_valid == False → score ≤ 0.1
#   M3 empty-answer cap: M3 ≤ 1 → score ≤ 0.2

import os
import json
import time
import requests

# ── Judge model configuration ─────────────────────────────────────────────────

JUDGE_A = {
    "model":    "google/gemini-2.0-flash-001",
    "base_url": "https://openrouter.ai/api/v1/chat/completions",
    "family":   "gemini",
}
JUDGE_B = {
    "model":    "openai/gpt-4o-mini",
    "base_url": "https://openrouter.ai/api/v1/chat/completions",
    "family":   "gpt",
}
JUDGE_C = {
    "model":    "llama-3.3-70b-versatile",
    "base_url": "https://api.groq.com/openai/v1/chat/completions",
    "family":   "llama",
}

# Generator family tags — used for self-enhancement guard
_FAMILY_MAP = {
    "openai":     "gpt",
    "gpt":        "gpt",
    "anthropic":  "claude",
    "claude":     "claude",
    "google":     "gemini",
    "gemini":     "gemini",
    "llama":      "llama",
    "meta-llama": "llama",
    "deepseek":   "deepseek",
}

# ── CR-specific context block injected into every judge prompt ────────────────
# (~300 tokens, negligible cost, large quality gain per §6 of LLM-judge research)

_CR_CONTEXT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXTO BIOGEOGRÁFICO DE COSTA RICA (referencia para el juez)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Regiones y zonas clave que el LLM evaluado puede mencionar:
- Vertientes: Pacífico (seco en Guanacaste, húmedo en Osa/Quepos), Caribe (húmedo, lluvias todo el año)
- Cordilleras: Volcánica Central (Poás, Barva, Irazú, Turrialba), Talamanca (Chirripó, 3820 m), Guanacaste, Tilarán
- Llanuras: Llanura del Norte (San Carlos), Llanuras de Tortuguero, Valle Central
- Zonas biogeográficas clave: Península de Osa, Zona Sur, Valle de El General, Turrialba, Sarapiquí
- Unidades fitogeográficas (Hammel 2014): bosque lluvioso de bajura Caribe, bosque premontano,
  bosque montano, bosque seco Guanacaste, manglar, páramo (> 3000 m)

Convenciones del Manual de Plantas de Costa Rica:
- Notación altitudinal: "500–1500 (–1800) m" = rango normal 500–1500 m, registros raros hasta 1800 m
- Las fichas pueden omitir la vertiente si la especie es ubicua; no penalices por ello si hay razón
- Vocabulary morfológico esperado cuando sea relevante: estípulas, pecíolos, inflorescencias, brácteas

INSTRUCCIÓN ANTI-VERBOSIDAD:
No premies respuestas más largas. Evalúa solo precisión técnica:
metros exactos citados, nombres de zonas exactos, variables climáticas correctas.
Una respuesta concisa y precisa supera a una larga pero vaga.
"""

JUDGE_PROMPT = """Eres un evaluador experto en botánica costarricense y ecología de plantas tropicales.
Se te proporciona una PREGUNTA, una RESPUESTA generada por un sistema de IA, y una FICHA DE REFERENCIA
del Manual de Plantas de Costa Rica (ground truth de expertos).

Evalúa en CINCO dimensiones. Sigue el orden M5 → M1 → M3 → M2 → M4.
Para cada dimensión: cita textualmente el fragmento de la respuesta que fundamenta tu puntuación
(campo "cita_*"), luego explica brevemente tu razonamiento (campo "razonamiento_*"), luego asigna el número.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXTO DEL SISTEMA — TIER: {tier}
  T0 = Conocimiento previo del LLM únicamente — sin mapas, sin datos instrumentales.
       El modelo respondió como experto botánico sin ningún contexto adicional.
       Evalúa la calidad factual y analítica de la respuesta contra la ficha de referencia.
  T1 = Solo mapa GBIF Mesoamérica          → sin datos de altitud, sin RF/SHAP
  T3 = Ambos mapas + RF (AUC/SHAP) + altitud GBIF + altitud del Manual

PREGUNTA:
{pregunta}

RESPUESTA A EVALUAR:
{respuesta}

FICHA DE REFERENCIA (Manual de Plantas de Costa Rica):
{ficha_mdp}
{cr_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DIMENSIONES DE EVALUACIÓN (en este orden):

──────────────────────────────────────
M5 — PROFUNDIDAD ANALÍTICA (0–3)  [aplica a TODOS los tiers — evaluar PRIMERO]
Mide la calidad del razonamiento, independientemente de la exactitud factual.
Longitud ≠ profundidad. Una respuesta corta con razonamiento sólido puede obtener M5=3.
Una recitación larga de datos sin interpretación obtiene M5=0.

  0 = Superficial: recita o enumera datos sin ninguna interpretación.
      No aborda limitaciones, tensiones entre fuentes ni incertidumbre.

  1 = Mínimo: menciona al menos una limitación, incertidumbre o calificación relevante.
      No integra fuentes ni reconcilia tensiones.

  2 = Reconciliador: identifica y trata de reconciliar tensiones entre fuentes,
      o señala discrepancias explícitas entre lo observado y lo esperado.

  3 = Integrativo: sintetiza múltiples líneas de evidencia críticamente,
      distingue afirmaciones respaldadas de especulativas, y enmarca conclusiones
      con cobertura epistémica apropiada.

──────────────────────────────────────
M1 — PRECISIÓN GEOGRÁFICA (0–3)  [aplica a todos los tiers]
¿Las zonas geográficas mencionadas (vertiente, cordillera, región, cantón, parque)
son consistentes con el Manual de Plantas de Costa Rica?
Para T0: evalúa directamente si las afirmaciones geográficas son correctas o incorrectas
comparadas con la ficha — hallucinations score M1=0.

  0 = Zonas incorrectas, contradicen el Manual, o afirmaciones geográficas fabricadas
  1 = Una zona correcta pero incompleta, o con errores notables
  2 = Zonas correctas pero sin especificidad suficiente
  3 = Zonas completas, específicas y consistentes con el Manual

──────────────────────────────────────
M3 — RELEVANCIA DE RESPUESTA (0–3)  [aplica a todos los tiers]
¿La respuesta contesta directamente la pregunta con información específica de la especie?

  0 = No responde la pregunta, es completamente genérica, o esquiva con vaguedades
  1 = Responde parcialmente, sin especificidad suficiente para la especie
  2 = Responde la pregunta pero con digresiones o solo parcialmente
  3 = Responde directo, conciso, y con datos específicos de la especie

──────────────────────────────────────
M2 — USO DEL CONTEXTO ALTITUDINAL (0–2 o "N/A")  [BONUS para T3 — N/A obligatorio para T0/T1]

▸ Si TIER = T0 o T1: asigna "N/A" siempre. El sistema no proporcionó datos de altitud.
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
M4 — USO DEL CONTEXTO CLIMÁTICO/RF (0–2 o "N/A")  [BONUS para T3 — N/A obligatorio para T0/T1]

▸ Si TIER = T0 o T1: asigna "N/A" siempre. No se proporcionaron métricas RF ni SHAP.
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

[T0 — M5=3] "Blakea gracilis florece principalmente entre enero y mayo en la vertiente Caribe,
con un segundo pico entre septiembre y noviembre. Como melastomatácea de bosque montano, su
fenología está ligada a la humedad estacional — los picos de floración coinciden con el inicio
de lluvias. La variación altitudinal (800–2350 m) introduce un desfase de 2–4 semanas entre
poblaciones de baja y alta elevación."
→ M5=3: da datos concretos, explica el mecanismo ecológico, señala variación intra-específica.

[T1 — M5=0] "Los puntos GBIF muestran presencia en la Cordillera de Talamanca, Cordillera Central
y vertiente Caribe. La especie prefiere elevaciones medias. Se distribuye ampliamente en Costa Rica."
→ M5=0: recitación de datos visibles sin interpretación; no reconcilia nada, no señala limitaciones.

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


# ── Score computation ─────────────────────────────────────────────────────────

def _score_compuesto(scores: dict, perfil: str, tier: str = "T3",
                     taxonomy_valid: bool = True) -> float:
    """
    Calcula el score compuesto normalizado [0, 1].

    Caps (stackable via min):
      - D1 taxonomy cap: taxonomy_valid == False → score ≤ 0.1
      - M3 empty-answer cap: M3 ≤ 1 → score ≤ 0.2
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

    if tier == "T3" and m2 != "N/A":
        total += (m2 if isinstance(m2, (int, float)) else 0)
        denom += 2

    if tier == "T3" and perfil == "botanico" and m4 != "N/A":
        total += (m4 if isinstance(m4, (int, float)) else 0)
        denom += 2

    raw = round(total / denom, 4) if denom > 0 else 0.0

    # Apply caps
    cap = 1.0
    if not taxonomy_valid:
        cap = min(cap, 0.1)
    m3_val = m3 if isinstance(m3, (int, float)) else 0
    if m3_val <= 1:
        cap = min(cap, 0.2)

    return round(min(raw, cap), 4)


def _generator_family(modelo_generador: str) -> str:
    """Extrae la familia del modelo generador para el self-enhancement guard."""
    modelo_lower = modelo_generador.lower()
    for prefix, family in _FAMILY_MAP.items():
        if prefix in modelo_lower:
            return family
    return "unknown"


# ── Single judge client ───────────────────────────────────────────────────────

class JudgeClient:
    """
    Cliente proveedor-agnóstico para un único juez LLM.
    Compatible con OpenRouter y Groq (ambos tienen endpoints OpenAI-compatible).
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model    = model
        self.api_key  = api_key
        self.base_url = base_url

    def _call(self, prompt: str, max_retries: int = 3) -> dict | None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       self.model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        for attempt in range(max_retries):
            try:
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=60)

                if resp.status_code == 429:
                    wait = 15 * (attempt + 1)
                    print(f"[JUDGE] {self.model} rate-limited — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    print(f"[JUDGE ERROR] {self.model} HTTP {resp.status_code}: {resp.text[:200]}")
                    return None

                content = resp.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]

                return json.loads(content)

            except json.JSONDecodeError as e:
                print(f"[JUDGE ERROR] {self.model} — no se pudo parsear JSON: {e}")
                return None
            except Exception as e:
                print(f"[JUDGE ERROR] {self.model} — {e}")
                return None

        print(f"[JUDGE ERROR] {self.model} — agotados {max_retries} intentos (rate limit persistente).")
        return None

    def evaluar(self, pregunta: str, respuesta: str, ficha_mdp: str, tier: str) -> dict | None:
        prompt = JUDGE_PROMPT.format(
            tier=tier,
            pregunta=pregunta,
            respuesta=respuesta,
            ficha_mdp=ficha_mdp,
            cr_context=_CR_CONTEXT,
        )
        return self._call(prompt)


# ── Ensemble judge ────────────────────────────────────────────────────────────

class EnsembleJudge:
    """
    Ensemble de 2 jueces (A + B) con tie-breaker C (Groq, gratis).
    Implementa self-enhancement guard y disagreement flag.
    """

    DISAGREE_THRESHOLD = 2  # |A − B| >= threshold en cualquier métrica → invocar C

    def __init__(self, openrouter_api_key: str, groq_api_key: str | None = None):
        self._or_key   = openrouter_api_key
        self._groq_key = groq_api_key

        self._judge_a = JudgeClient(
            model=JUDGE_A["model"],
            api_key=openrouter_api_key,
            base_url=JUDGE_A["base_url"],
        )
        self._judge_b = JudgeClient(
            model=JUDGE_B["model"],
            api_key=openrouter_api_key,
            base_url=JUDGE_B["base_url"],
        )
        self._judge_c = JudgeClient(
            model=JUDGE_C["model"],
            api_key=groq_api_key or "",
            base_url=JUDGE_C["base_url"],
        ) if groq_api_key else None

    def _select_judges(self, modelo_generador: str):
        """
        Aplica self-enhancement guard: si la familia del generador coincide
        con un juez, ese juez se reemplaza por C.
        Retorna (judge_primary, judge_secondary, label_primary, label_secondary).
        """
        gen_family = _generator_family(modelo_generador)
        ja, jb = self._judge_a, self._judge_b
        la, lb = "A", "B"

        if gen_family == JUDGE_A["family"] and self._judge_c:
            print(f"[ENSEMBLE] Self-enhancement guard: reemplazando Judge A ({JUDGE_A['model']}) por C (familia={gen_family})")
            ja, la = self._judge_c, "C"
        elif gen_family == JUDGE_B["family"] and self._judge_c:
            print(f"[ENSEMBLE] Self-enhancement guard: reemplazando Judge B ({JUDGE_B['model']}) por C (familia={gen_family})")
            jb, lb = self._judge_c, "C"

        return ja, jb, la, lb

    def _avg_scores(self, s1: dict, s2: dict) -> dict:
        """Promedia las métricas numéricas de dos dicts de scores."""
        metrics = ["M1_precision_geografica", "M2_precision_altitudinal",
                   "M3_relevancia_respuesta", "M4_variable_climatica", "M5_profundidad_analitica"]
        avg = {}
        for m in metrics:
            v1, v2 = s1.get(m), s2.get(m)
            if v1 == "N/A" or v2 == "N/A":
                avg[m] = "N/A"
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                avg[m] = round((v1 + v2) / 2, 2)
            else:
                avg[m] = v1  # fallback
        return avg

    def _max_disagreement(self, s1: dict, s2: dict) -> float:
        """Retorna el mayor |A − B| entre métricas numéricas."""
        metrics = ["M1_precision_geografica", "M3_relevancia_respuesta", "M5_profundidad_analitica",
                   "M2_precision_altitudinal", "M4_variable_climatica"]
        max_diff = 0.0
        for m in metrics:
            v1, v2 = s1.get(m), s2.get(m)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                max_diff = max(max_diff, abs(v1 - v2))
        return max_diff

    def evaluar(
        self,
        pregunta:        str,
        respuesta:       str,
        ficha_mdp:       str,
        perfil:          str,
        especie:         str,
        tier:            str,
        modelo_generador: str,
        taxonomy_valid:  bool = True,
        output_dir:      str  = None,
    ) -> dict | None:
        """
        Evalúa una respuesta con el ensemble.
        Retorna dict con scores individuales, agregado, score compuesto, y flags.
        """
        ja, jb, la, lb = self._select_judges(modelo_generador)

        print(f"[ENSEMBLE] Evaluando con Judge {la} ({ja.model})...")
        scores_a = ja.evaluar(pregunta, respuesta, ficha_mdp, tier)

        print(f"[ENSEMBLE] Evaluando con Judge {lb} ({jb.model})...")
        scores_b = jb.evaluar(pregunta, respuesta, ficha_mdp, tier)

        if not scores_a and not scores_b:
            print("[ENSEMBLE] Ambos jueces fallaron.")
            return None

        # Si uno falla, usar solo el exitoso
        if not scores_a:
            scores_a = scores_b
            la = lb
        if not scores_b:
            scores_b = scores_a
            lb = la

        max_diff    = self._max_disagreement(scores_a, scores_b)
        disagree    = max_diff >= self.DISAGREE_THRESHOLD
        scores_c    = None
        used_tiebreaker = False

        if disagree and self._judge_c:
            print(f"[ENSEMBLE] Desacuerdo (max_diff={max_diff:.1f}) — invocando tie-breaker C ({self._judge_c.model})...")
            scores_c = self._judge_c.evaluar(pregunta, respuesta, ficha_mdp, tier)
            used_tiebreaker = True

        # Aggregate: mean of the two primary judges (A and B after guard)
        scores_agg = self._avg_scores(scores_a, scores_b)

        score_comp = _score_compuesto(scores_agg, perfil, tier, taxonomy_valid)

        result = {
            "especie":           especie,
            "tier":              tier,
            "perfil":            perfil,
            "modelo_generador":  modelo_generador,
            "modelo_juez_A":     ja.model,
            "modelo_juez_B":     jb.model,
            "judge_A_label":     la,
            "judge_B_label":     lb,
            "judge_A_scores":    scores_a,
            "judge_B_scores":    scores_b,
            "judge_C_scores":    scores_c,
            "judge_aggregate":   scores_agg,
            "score_compuesto":   score_comp,
            "taxonomy_valid":    taxonomy_valid,
            "disagree_flag":     disagree,
            "max_metric_diff":   max_diff,
            "used_tiebreaker":   used_tiebreaker,
            "needs_human_review": disagree,
        }

        if output_dir:
            _guardar_evaluacion(result, output_dir, tier, modelo_generador, perfil)

        return result


# ── Legacy single-judge client (kept for backwards compat / unit tests) ───────

class JudgeClientLegacy:
    """
    Cliente Gemini-as-Judge original (single judge).
    Mantenido para compatibilidad y unit tests de rúbrica.
    Para producción usar EnsembleJudge.
    """

    def __init__(self, api_key: str):
        self._client = JudgeClient(
            model=JUDGE_A["model"],
            api_key=api_key,
            base_url=JUDGE_A["base_url"],
        )

    def evaluar(self, pregunta, respuesta, ficha_mdp, perfil, especie, tier,
                modelo_generador, output_dir=None, taxonomy_valid=True):
        scores = self._client.evaluar(pregunta, respuesta, ficha_mdp, tier)
        if not scores:
            return None
        scores["score_compuesto"] = _score_compuesto(scores, perfil, tier, taxonomy_valid)
        scores["especie"]          = especie
        scores["tier"]             = tier
        scores["perfil"]           = perfil
        scores["modelo_generador"] = modelo_generador
        scores["modelo_juez"]      = self._client.model
        if output_dir:
            _guardar_evaluacion(scores, output_dir, tier, modelo_generador, perfil)
        return scores


# ── Persist ───────────────────────────────────────────────────────────────────

def _guardar_evaluacion(scores: dict, output_dir: str, tier: str, modelo: str, perfil: str):
    modelo_limpio = modelo.replace("/", "_").replace("-", "_")
    nombre = f"eval_{tier}_{modelo_limpio}_{perfil}.json"
    ruta = os.path.join(output_dir, nombre)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print(f"[JUDGE] Evaluación guardada: {ruta}")


# ── Utilities ─────────────────────────────────────────────────────────────────

def ficha_summary(ficha_texto: str) -> str:
    """
    Extrae un resumen compacto de la ficha MdP para el juez.
    Solo incluye: distribución geográfica, rango altitudinal y tipo de hábitat.
    """
    lines = ficha_texto.splitlines()
    relevant = []
    target_sections = {
        "distribución geográfica", "rango altitudinal",
        "tipo de hábitat", "habitat", "altitud",
    }

    i = 0
    while i < len(lines):
        line = lines[i]
        line_lower = line.lower().strip().rstrip(":")

        if any(sec in line_lower for sec in target_sections):
            relevant.append(line)
            i += 1
            while i < len(lines) and lines[i].strip():
                relevant.append(lines[i])
                i += 1
            relevant.append("")
        else:
            i += 1

    if not relevant:
        return ficha_texto[:400]
    return "\n".join(relevant).strip()


def evaluar_directorio(output_dir: str, ficha_mdp: str, pregunta: str,
                       perfil: str, openrouter_api_key: str,
                       groq_api_key: str = None) -> list[dict]:
    """
    Evalúa todos los archivos llm_profile_BIMODAL_*.txt encontrados en output_dir.
    """
    import glob
    judge = EnsembleJudge(openrouter_api_key, groq_api_key)
    resultados = []

    archivos = glob.glob(os.path.join(output_dir, "llm_profile_BIMODAL_*.txt"))
    for archivo in archivos:
        nombre      = os.path.basename(archivo)
        tier        = "T3"
        modelo_gen  = nombre.replace("llm_profile_BIMODAL_", "").replace(".txt", "")

        with open(archivo, encoding="utf-8") as f:
            contenido = f.read()

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
