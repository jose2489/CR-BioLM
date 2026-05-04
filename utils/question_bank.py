# utils/question_bank.py
#
# Banco de preguntas para el pipeline CR-BioLM.
# Organizadas por perfil de usuario. Se inyectan al LLM como "user_question".
#
# Estructura de cada pregunta:
#   "q"                — texto de la pregunta
#   "tier_min"         — tier mínimo para que sea respondible con el contexto disponible
#                        T1 = solo GBIF | T3 = + RF/SHAP/altitud
#   "stratum"          — A (tier_min=T1), C (tier_min=T3)
#   "stratum_rationale"— justificación del stratum para el apéndice de la tesis
#
# T0 NO tiene tier_min propio — recibe todas las preguntas como baseline de conocimiento previo.
# Una respuesta correcta de rechazo en T0 ante una pregunta stratum C es el comportamiento esperado.
#
# Strata: A = answerable with GBIF alone (T1+), C = requires RF/SHAP/altitude (T3 only)
# T2 fue eliminado — no era ablación limpia (contenía texto del Manual = ground truth leakage).
#
# Uso desde CLI (main.py):
#   --persona turista                          → pregunta aleatoria del perfil
#   --persona botanico                         → pregunta aleatoria del perfil
#   --question "texto libre"                   → pregunta explícita (sin banco)

import random

# Semilla fija para reproducibilidad (commit en experiment_meta.json)
_SEED = 42

# ------------------------------------------------------------------
# BANCO DE PREGUNTAS
# stratum A = tier_min T1 (all tiers)
# stratum C = tier_min T3 (T3 only; T0/T1 expected to refuse or underperform)
# ------------------------------------------------------------------
QUESTION_BANK = {

    # ---- TURISTA / NATURALISTA CASUAL ----
    "turista": [
        {
            "q": "Si visito Playa Panamá, ¿puedo encontrarme con esta especie?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "La presencia en una región costera específica es evaluable con puntos GBIF sobre mapa Mesoamérica.",
        },
        {
            "q": "Si hago senderismo en el Parque Nacional Braulio Carrillo, ¿es probable que vea esta planta?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "La probabilidad de avistamiento en un parque concreto se puede estimar a partir de la distribución geográfica GBIF.",
        },
        {
            "q": "¿En qué zonas geográficas es más probable observarla en floración?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "La fenología y su variación geográfica pueden inferirse de los metadatos temporales y espaciales de los registros GBIF disponibles.",
        },
        {
            "q": "¿Esta planta se puede ver en jardines o parques urbanos de San José?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "La presencia urbana es evaluable observando si hay registros GBIF cerca de la Gran Área Metropolitana.",
        },
        {
            "q": "¿A qué altura tengo que subir para encontrar esta especie y en qué cordillera o región de Costa Rica?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "El rango altitudinal óptimo requiere datos GBIF con altitud + rango del Manual (disponibles en T3); GBIF solo no provee referencia altitudinal Manual.",
        },
        {
            "q": "¿Esta especie es fácil de identificar a simple vista en el bosque?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "Pregunta sobre rasgos morfológicos diagnósticos que puede responderse desde conocimiento general de la especie.",
        },
        {
            "q": "¿Dónde en Costa Rica tengo más probabilidad de encontrar esta planta?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "La concentración geográfica de registros GBIF identifica las zonas de mayor densidad de presencia.",
        },
        {
            "q": "¿Esta especie existe en la Península de Osa?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "Presencia/ausencia en una región geográfica específica es directamente evaluable con los puntos GBIF.",
        },
        {
            "q": "¿Puedo ver esta especie en el Parque Nacional Manuel Antonio?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "Igual que Braulio Carrillo — presencia en un parque específico derivable de la distribución GBIF.",
        },
        {
            "q": "¿Esta planta crece cerca del mar o solo tierra adentro?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "La distribución costera vs interior es visible en el patrón espacial de puntos GBIF sobre mapa Mesoamérica.",
        },
    ],

    # ---- BOTÁNICO / ECÓLOGO ----
    "botanico": [
        {
            "q": "¿Cuál es la variable climática más determinante para la distribución de esta especie y por qué?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "Requiere valores SHAP del modelo RF para identificar la variable climática más limitante; no es inferible solo de GBIF.",
        },
        {
            "q": "¿Cuál es el rango altitudinal óptimo de esta especie y cómo se compara con lo reportado en el Manual de Plantas de Costa Rica?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "Requiere el rango altitudinal GBIF detectado + referencia del Manual, ambos disponibles solo en T3; T1 solo tiene coordenadas GBIF sin referencia Manual.",
        },
        {
            "q": "¿Existe coherencia entre el modelo climático predictivo y la distribución descrita en el Manual de Plantas?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "Requiere el mapa predictivo RF (segunda imagen T3) para compararlo con el mapa de hábitat predicho; no existe en T1.",
        },
        {
            "q": "¿Qué tipo de bosque define el nicho realizado de esta especie y en qué vertiente o cordillera se concentra?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "El tipo de bosque y la vertiente predominante en las zonas con registros GBIF puede inferirse del contexto geográfico visible en el mapa Mesoamérica.",
        },
        {
            "q": "¿Cómo influye la precipitación del trimestre más seco en la idoneidad del hábitat de esta especie?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "bio_17 (precipitación trimestre más seco) es una variable bioclimática WorldClim solo disponible en T3 vía SHAP/RF.",
        },
        {
            "q": "¿Hay zonas de Costa Rica donde el modelo predice alta idoneidad pero no hay registros GBIF confirmados?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "Requiere superponer el mapa predictivo RF (T3) con los puntos GBIF; no existe modelo predictivo en T1.",
        },
        {
            "q": "¿Esta especie muestra preferencia por alguna vertiente (Caribe vs Pacífico)?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "La distribución por vertiente (Caribe/Pacífico) es directamente visible en el patrón espacial de puntos GBIF.",
        },
        {
            "q": "¿Qué tan sensible es esta especie al cambio en temperatura media anual y en qué zonas de Costa Rica se manifiesta esa sensibilidad?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "bio_1 (temperatura media anual) y su importancia SHAP solo están disponibles en T3; la dimensión espacial requiere cruzar con el mapa RF.",
        },
        {
            "q": "¿El rango de distribución predicho en Mesoamérica es coherente con su ecología conocida?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "El mapa predictivo Mesoamericano (imagen 2 en T3) es el único que muestra rango predicho; no existe en T1.",
        },
        {
            "q": "¿Los registros GBIF confirman o contradicen el hábitat descrito en el Manual de Plantas?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "Los registros GBIF (T1) pueden compararse con la descripción textual del Manual sin necesitar datos RF.",
        },
        {
            "q": "¿Qué factores climáticos secundarios modulan la distribución de esta especie más allá de la variable principal y en qué subregiones de Costa Rica tienen más peso?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "Las variables secundarias SHAP (bio_X, bio_Y) y su expresión geográfica solo están disponibles en T3 como parte del output del modelo RF.",
        },
        # ── Nuevas preguntas — basadas en patrones de investigación de ecólogos tropicales ──
        {
            "q": "¿Dónde se localizan los límites altitudinales y geográficos de distribución de esta especie en Costa Rica, y qué factores climáticos los explican?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "Detectar límites de rango requiere cruzar altitud GBIF + rango Manual + variables limitantes SHAP, todos disponibles solo en T3.",
        },
        {
            "q": "¿Existen disyunciones notables en la distribución conocida de esta especie y dónde se ubican geográficamente?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "Las disyunciones (gaps en la distribución) son visibles en el patrón espacial de puntos GBIF sin necesitar el modelo RF.",
        },
        {
            "q": "¿Qué zonas actuales de presencia de esta especie son más vulnerables a desplazamiento bajo escenarios de aumento de temperatura?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "La vulnerabilidad climática espacial requiere cruzar la variable limitante SHAP (temperatura) con las zonas de presencia actuales del mapa RF.",
        },
        {
            "q": "¿En qué zonas de Costa Rica el hábitat predicho de esta especie se solapa con áreas de bosque ya intervenido o fragmentado?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "Requiere el mapa predictivo RF (T3) para identificar zonas de hábitat idóneo; T1 solo muestra presencias observadas.",
        },
        {
            "q": "¿La distribución observada en GBIF refleja un sesgo de muestreo geográfico (sobre-representación cerca de carreteras o estaciones biológicas)?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "El sesgo de muestreo GBIF es evaluable directamente del patrón espacial de puntos sin necesitar el modelo RF.",
        },
        {
            "q": "¿La especie muestra preferencia por alguna posición topográfica específica (laderas, cumbres, fondos de valle) según la distribución de elevación de sus registros?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "La especificidad topográfica requiere los datos de altitud de los registros GBIF + rango del Manual, disponibles en T3.",
        },
    ],
}

_TIER_ORDER = {"T1": 1, "T3": 3}

ALL_QUESTIONS = [entry for qs in QUESTION_BANK.values() for entry in qs]


def get_questions(persona=None, tier=None):
    """
    Retorna las preguntas (como strings) para un perfil, opcionalmente
    filtradas por tier: solo preguntas con tier_min <= tier dado.
    """
    entries = QUESTION_BANK.get(persona, ALL_QUESTIONS) if persona else ALL_QUESTIONS
    if persona and persona not in QUESTION_BANK:
        raise ValueError(f"Perfil '{persona}' no existe. Opciones: {list(QUESTION_BANK.keys())}")
    if tier:
        tier_val = _TIER_ORDER.get(tier, 99)
        entries = [e for e in entries if _TIER_ORDER.get(e["tier_min"], 1) <= tier_val]
    return [e["q"] for e in entries]


def get_question_meta(persona=None, tier=None):
    """
    Retorna lista de dicts completos (q, tier_min, stratum, stratum_rationale)
    filtrada por tier_min <= tier dado.
    """
    entries = QUESTION_BANK.get(persona, ALL_QUESTIONS) if persona else ALL_QUESTIONS
    if persona and persona not in QUESTION_BANK:
        raise ValueError(f"Perfil '{persona}' no existe.")
    if tier:
        tier_val = _TIER_ORDER.get(tier, 99)
        entries = [e for e in entries if _TIER_ORDER.get(e["tier_min"], 1) <= tier_val]
    return entries


def get_random_question(persona=None, tier=None, seed=_SEED, return_meta=False):
    """
    Retorna una pregunta aleatoria del banco, compatible con el tier dado.

    tier: "T1" | "T3" — si se proporciona, solo preguntas con tier_min <= tier.
    seed: semilla reproducible (default 42). Fijar por especie para consistencia.
    return_meta: si True, retorna (pregunta_str, stratum_str) en lugar de solo pregunta_str.
    """
    entries = get_question_meta(persona, tier=tier)
    if not entries:
        raise ValueError(f"No hay preguntas para persona='{persona}' con tier='{tier}'")

    rng = random.Random(seed)
    entry = rng.choice(entries)
    pregunta = entry["q"]
    stratum  = entry.get("stratum", "A")

    if return_meta:
        return pregunta, stratum
    return pregunta


if __name__ == "__main__":
    print("=== BANCO DE PREGUNTAS CR-BioLM ===\n")
    for perfil, entries in QUESTION_BANK.items():
        print(f"[{perfil.upper()}] ({len(entries)} preguntas)")
        for i, e in enumerate(entries, 1):
            print(f"  {i}. [{e['tier_min']}|{e['stratum']}] {e['q']}")
            print(f"       → {e['stratum_rationale']}")
        print()

    print("=== EJEMPLOS ALEATORIOS POR TIER ===")
    for tier in ["T1", "T3"]:
        q, s = get_random_question("botanico", tier=tier, return_meta=True)
        print(f"  [botanico/{tier}/stratum={s}] {q}")
