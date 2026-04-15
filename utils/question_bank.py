# utils/question_bank.py
#
# Banco de preguntas para el pipeline CR-BioLM.
# Organizadas por perfil de usuario. Se inyectan al LLM como "user_question".
#
# Uso desde CLI (main.py):
#   --persona turista                          → pregunta aleatoria del perfil
#   --persona botanico                         → pregunta aleatoria del perfil
#   --persona municipalidad --canton "Nicoya" --proyecto "carretera costanera"
#   --question "texto libre"                   → pregunta explícita (sin banco)

import random

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
# Las preguntas de municipalidad usan {canton} y {proyecto} como placeholders.
# ------------------------------------------------------------------
QUESTION_BANK = {

    # ---- TURISTA / NATURALISTA CASUAL ----
    "turista": [
        "Si visito Playa Panamá, ¿puedo encontrarme con esta especie?",
        "Si hago senderismo en el Parque Nacional Braulio Carrillo, ¿es probable que vea esta planta?",
        "¿En qué época del año florece esta especie en Costa Rica?",
        "¿Esta planta se puede ver en jardines o parques urbanos de San José?",
        "¿A qué altura tengo que subir para encontrar esta especie?",
        "¿Esta especie es fácil de identificar a simple vista en el bosque?",
        "¿Dónde en Costa Rica tengo más probabilidad de encontrar esta planta?",
        "¿Esta especie existe en la Península de Osa?",
        "¿Puedo ver esta especie en el Parque Nacional Manuel Antonio?",
        "¿Esta planta crece cerca del mar o solo tierra adentro?",
    ],

    # ---- BOTÁNICO / ECÓLOGO ----
    "botanico": [
        "¿Cuál es la variable climática más determinante para la distribución de esta especie y por qué?",
        "¿Cuál es el rango altitudinal óptimo de esta especie y cómo se compara con lo reportado en el Manual de Plantas de Costa Rica?",
        "¿Existe coherencia entre el modelo climático predictivo y la distribución descrita en el Manual de Plantas?",
        "¿Qué tipo de bosque define el nicho realizado de esta especie según los registros de presencia?",
        "¿Cómo influye la precipitación del trimestre más seco en la idoneidad del hábitat de esta especie?",
        "¿Hay zonas de Costa Rica donde el modelo predice alta idoneidad pero no hay registros GBIF confirmados?",
        "¿Esta especie muestra preferencia por alguna vertiente (Caribe vs Pacífico)?",
        "¿Qué tan sensible es esta especie al cambio en temperatura media anual?",
        "¿El rango de distribución predicho en Mesoamérica es coherente con su ecología conocida?",
        "¿Los registros GBIF confirman o contradicen el hábitat descrito en el Manual de Plantas?",
        "¿Qué factores climáticos secundarios modulan la distribución de esta especie más allá de la variable principal?",
    ],

    # ---- MUNICIPALIDAD / GESTOR TERRITORIAL ----
    # Usan {canton} y {proyecto} como placeholders
    "municipalidad": [
        "¿Esta especie tiene presencia registrada en el cantón de {canton}? ¿Debería considerarse en el plan regulador?",
        "Si se aprueba un {proyecto} en el cantón de {canton}, ¿habría impacto sobre el hábitat de esta especie?",
        "¿Qué porcentaje del hábitat idóneo de esta especie se encuentra dentro de áreas protegidas de Costa Rica?",
        "¿La distribución de esta especie coincide con zonas de amortiguamiento de algún parque nacional cercano a {canton}?",
        "¿Es esta especie indicadora de algún tipo de ecosistema que debería protegerse en el plan de uso del suelo del cantón {canton}?",
        "¿Cuál es el riesgo de pérdida de hábitat para esta especie si se desarrolla un {proyecto} en {canton}?",
        "¿Existe presencia de esta especie en corredores biológicos activos que atraviesen el cantón de {canton}?",
        "¿Debería el cantón de {canton} incluir esta especie en su inventario de biodiversidad municipal?",
        "¿Un {proyecto} en la región de {canton} requeriría una evaluación de impacto ambiental que considere esta especie?",
    ],
}

ALL_QUESTIONS = [q for qs in QUESTION_BANK.values() for q in qs]


def get_questions(persona=None):
    """Retorna preguntas para un perfil. Si persona=None, retorna todas."""
    if persona is None:
        return ALL_QUESTIONS
    if persona not in QUESTION_BANK:
        raise ValueError(f"Perfil '{persona}' no existe. Opciones: {list(QUESTION_BANK.keys())}")
    return QUESTION_BANK[persona]


def get_random_question(persona=None, canton=None, proyecto=None):
    """
    Retorna una pregunta aleatoria del banco.
    Para persona='municipalidad', rellena {canton} y {proyecto} con valores
    aleatorios de las listas si no se proporcionan explícitamente.
    """
    pregunta = random.choice(get_questions(persona))

    # Rellenar placeholders de municipalidad
    if "{canton}" in pregunta or "{proyecto}" in pregunta:
        canton_val = canton if canton else random.choice(CANTONES_CR)
        proyecto_val = proyecto if proyecto else random.choice(PROYECTOS_TIPO)
        pregunta = pregunta.format(canton=canton_val, proyecto=proyecto_val)

    return pregunta


if __name__ == "__main__":
    print("=== BANCO DE PREGUNTAS CR-BioLM ===\n")
    for perfil, preguntas in QUESTION_BANK.items():
        print(f"[{perfil.upper()}] ({len(preguntas)} preguntas)")
        for i, p in enumerate(preguntas, 1):
            print(f"  {i}. {p}")
        print()

    print("=== EJEMPLOS ALEATORIOS ===")
    for perfil in QUESTION_BANK:
        q = get_random_question(perfil)
        print(f"  [{perfil}] {q}")
