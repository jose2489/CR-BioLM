import os
from utils.translator import traducir_variable

# ─────────────────────────────────────────────────────────────────────────────
# REGLA COMÚN — incluida en todos los prompts (T1/T3)
# ─────────────────────────────────────────────────────────────────────────────
_REGLA_STRICTA = """REGLAS ESTRICTAS:
- Razona EXCLUSIVAMENTE a partir de la información proporcionada en este prompt y la imagen adjunta.
- NO uses conocimiento previo sobre la especie, su taxonomía, ni su distribución.
- Si la imagen o los datos no son suficientes para responder algo, dilo explícitamente en lugar de inferirlo.
- NO menciones herramientas de software, nombres de modelos, ni fuentes externas."""

# ─────────────────────────────────────────────────────────────────────────────
# T0 — Baseline de conocimiento previo: sin contexto, sin imágenes
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_T0 = """Eres un experto en botánica y ecología de plantas tropicales de Costa Rica.
Responde la siguiente pregunta sobre {species_name} de la forma más completa y precisa posible,
basándote en tu conocimiento.

{instruccion_pregunta}
"""

# ─────────────────────────────────────────────────────────────────────────────
# T1 — Baseline: solo registros GBIF sobre mapa Mesoamericano
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_T1 = """Eres un evaluador ecológico imparcial. Tu tarea es responder una pregunta concreta sobre la especie {species_name} basándote ÚNICAMENTE en la imagen adjunta.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUENTE DISPONIBLE — REGISTROS DE PRESENCIA GBIF (Imagen adjunta)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
La imagen muestra los registros de presencia georreferenciados de la especie en Mesoamérica obtenidos de GBIF. Cada punto representa un espécimen colectado o observado con coordenadas validadas. No se ha aplicado ningún modelo ni interpretación adicional.

{_regla}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{instruccion_pregunta}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FORMATO DE RESPUESTA OBLIGATORIO:

## Razonamiento
Máximo 5 viñetas. Describe solo lo que ves en la imagen: distribución espacial de los puntos, concentraciones, ausencias notables. Si no puedes responder algún aspecto de la pregunta con esta imagen, indícalo explícitamente.

## Respuesta
3 a 4 oraciones. Responde la pregunta basándote únicamente en el patrón espacial de los registros GBIF visibles. Si la imagen no permite responder con certeza, indícalo.
"""

# ─────────────────────────────────────────────────────────────────────────────
# T2 — ARCHIVADO: tier eliminado (contenía fuente_manual → leakage vs ground truth)
# Mantenido solo para referencia histórica. No usar en experimentos nuevos.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# T3 — Sistema completo: mapa hábitat predicho + mapa RF + métricas SHAP/AUC
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_T3 = """Eres un evaluador ecológico imparcial. Tu tarea es responder una pregunta concreta sobre la especie {species_name} cruzando ÚNICAMENTE las tres fuentes de evidencia proporcionadas a continuación.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUENTE 1 — MODELO CLIMÁTICO (datos cuantitativos)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Precisión del modelo (AUC, validación por bloques espaciales): {rf_auc:.4f}
- Rango altitudinal observado en CR: {info_altitud}
- Variable climática más limitante: {var_humana} (impacto {direccion} sobre la idoneidad)
- Ecosistema de mayor idoneidad: {zona_humana}
- Variables secundarias: {secundaria_1}, {secundaria_2}
- {envolvente_climatica}
  IMPORTANTE: para describir el clima usa estos VALORES medidos (con sus unidades),
  no el nombre de la variable. "Precipitación anual mediana de 3200 mm" es una
  respuesta; "precipitación del trimestre frío" es solo una etiqueta.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUENTE 2 — MAPA DE HÁBITAT PREDICHO (Manual + Regiones Botánicas + DEM) (Imagen 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generado cruzando el Manual de Plantas de Costa Rica con las Regiones Botánicas de
Costa Rica (José Araya, 2026) y un modelo de elevación (DEM). Leyenda del mapa:
- RELLENO CYAN = hábitat potencial: la intersección de las regiones citadas en el
  Manual con el rango altitudinal de la especie. Es el ÚNICO relleno de color.
- CYAN CON TRAMA DIAGONAL = banda altitudinal atípica (registros ocasionales).
- CONTORNO BLANCO SIN RELLENO = región botánica citada en el texto del Manual.
- CONTORNO ÁMBAR PUNTEADO = región NO citada en el Manual pero con >=5 registros
  GBIF dentro del rango altitudinal (extensión inferida por evidencia de ocurrencia).
- CONTORNO NARANJA = parque nacional citado en el Manual, filtrado por elevación.
- CONTORNO ÁMBAR FINO = áreas silvestres protegidas dentro del rango altitudinal.
- PUNTOS ROJOS = presencias GBIF en Costa Rica (coordenadas limpias).
- GRIS OSCURO = fuera del rango geográfico indicado por el Manual.
Nota: si la especie tiene una elevación de punto único en el Manual (p. ej. "ca. 2950 m"),
la máscara de elevación queda vacía y NO habrá relleno cyan. En ese caso dilo.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUENTE 3 — MAPA PREDICTIVO CLIMÁTICO RF (Imagen 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Distribución modelada a partir de variables bioclimáticas WorldClim y presencias Mesoamericanas. Las zonas más oscuras indican mayor idoneidad climática predicha.

{_regla}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{instruccion_pregunta}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FORMATO DE RESPUESTA OBLIGATORIO:

## Razonamiento
Máximo 5 viñetas. Describe la POSICIÓN de lo que observas en las imágenes, no las etiquetas de la leyenda: ¿el relleno cyan está en la mitad norte o sur del país?, ¿en la vertiente Pacífica o Caribe?, ¿costa o interior?, ¿tierras bajas o altas?, ¿es una mancha continua o está fragmentado en bloques separados? ¿Los puntos rojos caen DENTRO o FUERA del cyan? ¿Coincide el mapa de hábitat (Imagen 1) con el mapa climático RF (Imagen 2)? Si hay discrepancias entre fuentes, señálalas. Si algo no está en los datos, no lo infieras.

## Respuesta
3 a 4 oraciones. Responde la pregunta usando únicamente la evidencia cruzada de las tres fuentes. Describe la distribución en términos POSICIONALES (norte/sur, vertiente Pacífica/Caribe, costa/interior, tierras bajas/altas) antes de nombrar cualquier región. Si los datos no permiten responder algún aspecto, indícalo.
"""

# Alias para compatibilidad con código existente (main.py usa BIMODAL_PROMPT para T3)
BIMODAL_PROMPT = PROMPT_T3

# Active tier prompts — T2 dropped (ficha leakage fix)
TIER_PROMPTS_ALL = {"T0": PROMPT_T0, "T1": PROMPT_T1, "T3": PROMPT_T3}

# Defaults dict used for reset operations
_DEFAULTS = {
    "T0": PROMPT_T0,
    "T1": PROMPT_T1,
    "T3": PROMPT_T3,
    "_REGLA_STRICTA": _REGLA_STRICTA,
}

_OVERRIDE_PATH = os.path.join(os.path.dirname(__file__), "prompts_override.json")


def get_effective_prompts() -> dict:
    """Returns the active prompts, applying any saved overrides."""
    prompts = dict(_DEFAULTS)
    if os.path.isfile(_OVERRIDE_PATH):
        try:
            import json
            overrides = json.loads(open(_OVERRIDE_PATH, encoding="utf-8").read())
            prompts.update({k: v for k, v in overrides.items() if k in prompts and v})
        except Exception:
            pass
    return prompts