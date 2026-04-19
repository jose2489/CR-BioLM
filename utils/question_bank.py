# utils/question_bank.py
#
# Banco de preguntas para el pipeline CR-BioLM.
# Organizadas por perfil de usuario. Se inyectan al LLM como "user_question".
#
# Estructura de cada pregunta:
#   "q"                — texto de la pregunta (con placeholders {canton}/{proyecto} para municipalidad)
#   "tier_min"         — tier mínimo para que sea respondible con el contexto disponible
#                        T1 = solo GBIF | T2 = + mapa botánico | T3 = + RF/SHAP/altitud
#   "stratum"          — A (tier_min=T1), B (tier_min=T2), C (tier_min=T3)
#   "stratum_rationale"— justificación del stratum para el apéndice de la tesis
#
# T0 NO tiene tier_min propio — recibe todas las preguntas como baseline de conocimiento previo.
# Una respuesta correcta de rechazo en T0 ante una pregunta stratum C es el comportamiento esperado.
#
# Uso desde CLI (main.py):
#   --persona turista                          → pregunta aleatoria del perfil
#   --persona botanico                         → pregunta aleatoria del perfil
#   --persona municipalidad --canton "Nicoya" --proyecto "carretera costanera"
#   --question "texto libre"                   → pregunta explícita (sin banco)

import random

# Semilla fija para reproducibilidad (commit en experiment_meta.json)
_SEED = 42

# ------------------------------------------------------------------
# CANTONES Y PROYECTOS — para parametrizar preguntas de municipalidad
# ------------------------------------------------------------------
CANTONES_CR = [
    "Nicoya", "Liberia", "Santa Cruz", "Bagaces", "Carrillo",
    "Cañas", "Abangares", "Tilarán", "Nandayure", "La Cruz", "Hojancha",
    "San José", "Escazú", "Desamparados", "Puriscal", "Tarrazú",
    "Aserrí", "Mora", "Goicoechea", "Santa Ana", "Alajuelita",
    "Vázquez de Coronado", "Acosta", "Tibás", "Moravia", "Montes de Oca",
    "Turrubares", "Dota", "Curridabat", "Pérez Zeledón", "León Cortés",
    "Alajuela", "San Ramón", "Grecia", "San Mateo", "Atenas", "Naranjo",
    "Palmares", "Poás", "Orotina", "San Carlos", "Zarcero", "Valverde Vega",
    "Upala", "Los Chiles", "Guatuso", "Río Cuarto",
    "Cartago", "Paraíso", "La Unión", "Jiménez", "Turrialba",
    "Alvarado", "Oreamuno", "El Guarco",
    "Heredia", "Barva", "Santo Domingo", "Santa Bárbara", "San Rafael",
    "San Isidro", "Belén", "Flores", "San Pablo", "Sarapiquí",
    "Limón", "Pococí", "Siquirres", "Talamanca", "Matina", "Guácimo",
    "Puntarenas", "Esparza", "Buenos Aires", "Montes de Oro", "Osa",
    "Quepos", "Golfito", "Coto Brus", "Parrita", "Corredores", "Garabito",
]

PROYECTOS_TIPO = [
    "proyecto de carretera",
    "desarrollo turístico costero",
    "proyecto hidroeléctrico",
    "expansión de zona franca",
    "proyecto de urbanización",
    "concesión minera",
    "proyecto agropecuario",
    "construcción de oleoducto",
    "proyecto de reforestación",
    "ampliación de puerto",
]

# ------------------------------------------------------------------
# BANCO DE PREGUNTAS
# stratum A = tier_min T1 (all tiers)
# stratum B = tier_min T2 (T2, T3)
# stratum C = tier_min T3 (T3 only; T0/T1/T2 expected to refuse or underperform)
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
            "q": "¿En qué época del año florece esta especie en Costa Rica?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "La fenología puede inferirse de los metadatos temporales de los registros GBIF disponibles.",
        },
        {
            "q": "¿Esta planta se puede ver en jardines o parques urbanos de San José?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "La presencia urbana es evaluable observando si hay registros GBIF cerca de la Gran Área Metropolitana.",
        },
        {
            "q": "¿A qué altura tengo que subir para encontrar esta especie?",
            "tier_min": "T2",
            "stratum": "B",
            "stratum_rationale": "El rango altitudinal requiere el mapa botánico con coloración por elevación (Manual + DEM); GBIF solo no es suficiente.",
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
            "stratum_rationale": "Requiere valores SHAP del modelo RF para identificar la variable climática más limitante; no es inferible solo de GBIF o del mapa botánico.",
        },
        {
            "q": "¿Cuál es el rango altitudinal óptimo de esta especie y cómo se compara con lo reportado en el Manual de Plantas de Costa Rica?",
            "tier_min": "T2",
            "stratum": "B",
            "stratum_rationale": "El mapa botánico T2 codifica altitud por color (óptimo vs fuera de rango); T1 solo tiene coordenadas GBIF sin referencia altitudinal Manual.",
        },
        {
            "q": "¿Existe coherencia entre el modelo climático predictivo y la distribución descrita en el Manual de Plantas?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "Requiere el mapa predictivo RF (segunda imagen T3) para compararlo con el mapa botánico Manual; no existe en T1/T2.",
        },
        {
            "q": "¿Qué tipo de bosque define el nicho realizado de esta especie según los registros de presencia?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "El tipo de bosque predominante en las zonas con registros GBIF puede inferirse del contexto geográfico visible en el mapa Mesoamérica.",
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
            "stratum_rationale": "Requiere superponer el mapa predictivo RF (T3) con los puntos GBIF; no existe modelo predictivo en T1/T2.",
        },
        {
            "q": "¿Esta especie muestra preferencia por alguna vertiente (Caribe vs Pacífico)?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "La distribución por vertiente (Caribe/Pacífico) es directamente visible en el patrón espacial de puntos GBIF.",
        },
        {
            "q": "¿Qué tan sensible es esta especie al cambio en temperatura media anual?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "bio_1 (temperatura media anual) y su importancia SHAP solo están disponibles en T3; no evaluable en T1/T2.",
        },
        {
            "q": "¿El rango de distribución predicho en Mesoamérica es coherente con su ecología conocida?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "El mapa predictivo Mesoamericano (imagen 1 en T3, imagen 2 en T3) es el único que muestra rango predicho; no existe en T1/T2.",
        },
        {
            "q": "¿Los registros GBIF confirman o contradicen el hábitat descrito en el Manual de Plantas?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "Los registros GBIF (T1) pueden compararse con la descripción textual del Manual sin necesitar el mapa botánico T2.",
        },
        {
            "q": "¿Qué factores climáticos secundarios modulan la distribución de esta especie más allá de la variable principal?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "Las variables secundarias SHAP (bio_X, bio_Y) solo están disponibles en T3 como parte del output del modelo RF.",
        },
    ],

    # ---- MUNICIPALIDAD / GESTOR TERRITORIAL ----
    "municipalidad": [
        {
            "q": "¿Esta especie tiene presencia registrada en el cantón de {canton}? ¿Debería considerarse en el plan regulador?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "La presencia en un cantón es directamente evaluable con los puntos GBIF georreferenciados.",
        },
        {
            "q": "Si se aprueba un {proyecto} en el cantón de {canton}, ¿habría impacto sobre el hábitat de esta especie?",
            "tier_min": "T2",
            "stratum": "B",
            "stratum_rationale": "El impacto requiere conocer el hábitat óptimo (T2 mapa botánico); solo con GBIF no se puede establecer si el hábitat es crítico.",
        },
        {
            "q": "¿Qué porcentaje del hábitat idóneo de esta especie se encuentra dentro de áreas protegidas de Costa Rica?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "Requiere el mapa predictivo RF (T3) para estimar hábitat idóneo vs áreas protegidas; T2 solo muestra hábitat Manual, no idoneidad climática.",
        },
        {
            "q": "¿La distribución de esta especie coincide con zonas de amortiguamiento de algún parque nacional cercano a {canton}?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "La superposición geográfica con zonas de amortiguamiento puede evaluarse directamente con los puntos GBIF.",
        },
        {
            "q": "¿Es esta especie indicadora de algún tipo de ecosistema que debería protegerse en el plan de uso del suelo del cantón {canton}?",
            "tier_min": "T2",
            "stratum": "B",
            "stratum_rationale": "Identificar el ecosistema indicado requiere el mapa botánico T2 que cruza el Manual con unidades fitogeográficas.",
        },
        {
            "q": "¿Cuál es el riesgo de pérdida de hábitat para esta especie si se desarrolla un {proyecto} en {canton}?",
            "tier_min": "T3",
            "stratum": "C",
            "stratum_rationale": "El riesgo de pérdida de hábitat requiere el mapa predictivo RF (idoneidad climática) y las variables limitantes SHAP, disponibles solo en T3.",
        },
        {
            "q": "¿Existe presencia de esta especie en corredores biológicos activos que atraviesen el cantón de {canton}?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "La presencia en corredores biológicos es evaluable con la distribución geográfica de registros GBIF.",
        },
        {
            "q": "¿Debería el cantón de {canton} incluir esta especie en su inventario de biodiversidad municipal?",
            "tier_min": "T1",
            "stratum": "A",
            "stratum_rationale": "La recomendación de incluir en inventario depende de si hay registros en el cantón, evaluable con GBIF.",
        },
        {
            "q": "¿Un {proyecto} en la región de {canton} requeriría una evaluación de impacto ambiental que considere esta especie?",
            "tier_min": "T2",
            "stratum": "B",
            "stratum_rationale": "Determinar si se requiere EIA requiere conocer si el hábitat óptimo (T2) coincide con el área del proyecto.",
        },
    ],
}

_TIER_ORDER = {"T1": 1, "T2": 2, "T3": 3}

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


def get_random_question(persona=None, canton=None, proyecto=None, tier=None,
                        seed=_SEED, return_meta=False):
    """
    Retorna una pregunta aleatoria del banco, compatible con el tier dado.
    Para persona='municipalidad', rellena {canton} y {proyecto} con valores
    aleatorios si no se proporcionan.

    tier: "T1" | "T2" | "T3" — si se proporciona, solo preguntas con
          tier_min <= tier.
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

    # Rellenar placeholders de municipalidad
    if "{canton}" in pregunta or "{proyecto}" in pregunta:
        canton_val   = canton   if canton   else rng.choice(CANTONES_CR)
        proyecto_val = proyecto if proyecto else rng.choice(PROYECTOS_TIPO)
        pregunta = pregunta.format(canton=canton_val, proyecto=proyecto_val)

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
    for tier in ["T1", "T2", "T3"]:
        q, s = get_random_question("botanico", tier=tier, return_meta=True)
        print(f"  [botanico/{tier}/stratum={s}] {q}")
