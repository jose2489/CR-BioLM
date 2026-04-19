from utils.translator import traducir_variable

# ─────────────────────────────────────────────────────────────────────────────
# REGLA COMÚN — incluida en todos los prompts (T1/T2/T3)
# ─────────────────────────────────────────────────────────────────────────────
_REGLA_STRICTA = """REGLAS ESTRICTAS:
- Razona EXCLUSIVAMENTE a partir de la información proporcionada en este prompt y la imagen adjunta.
- NO uses conocimiento previo sobre la especie, su taxonomía, ni su distribución.
- Si la imagen o los datos no son suficientes para responder algo, dilo explícitamente en lugar de inferirlo.
- NO menciones herramientas de software, nombres de modelos, ni fuentes externas."""

# ─────────────────────────────────────────────────────────────────────────────
# T0 — Baseline de conocimiento previo: sin contexto, sin imágenes
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_T0 = """Eres un evaluador ecológico imparcial. Tu tarea es responder una pregunta concreta sobre la especie {species_name}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONDICIÓN DE ESTA EVALUACIÓN — SIN CONTEXTO (T0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
No se te han proporcionado mapas, datos de campo, registros de presencia, ni material de referencia de ningún tipo.
Responde ÚNICAMENTE a partir del conocimiento que ya posees sobre esta especie.

REGLAS ESTRICTAS:
- Si puedes responder con confianza razonable basándote en conocimiento ya establecido, hazlo — pero indica el nivel de certeza.
- Si NO puedes responder con confianza sin datos adicionales, declara explícitamente que no puedes responder sin esa información y describe qué datos serían necesarios.
- NO fabriques hechos específicos sobre la especie (distribución exacta, rangos altitudinales precisos, etc.) si no los conoces con certeza.
- NO menciones herramientas de software, nombres de modelos, ni fuentes externas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{instruccion_pregunta}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FORMATO DE RESPUESTA OBLIGATORIO:

## Razonamiento
Máximo 5 viñetas. Describe qué sabes (o no sabes) sobre esta especie desde tu conocimiento previo. Si no puedes responder, explica qué información adicional se necesitaría y por qué.

## Respuesta
3 a 4 oraciones. Responde la pregunta con el conocimiento que tienes, indicando tu nivel de certeza. Si no puedes responder de forma responsable sin datos adicionales, declárate incapaz de responder con precisión y describe qué datos resolverían la pregunta.
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
# T2 — Mapa botánico interpretado (Manual + Unidades Fitogeográficas + DEM)
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_T2 = """Eres un evaluador ecológico imparcial. Tu tarea es responder una pregunta concreta sobre la especie {species_name} basándote ÚNICAMENTE en la imagen adjunta y su leyenda.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUENTE DISPONIBLE — MAPA DE HÁBITAT BOTÁNICO (Imagen adjunta)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
El mapa fue generado cruzando la descripción de hábitat del Manual de Plantas de Costa Rica con las Unidades Fitogeográficas (Hammel 2014) y un modelo digital de elevación. Leyenda:
- CYAN brillante = hábitat óptimo (región geográfica correcta + elevación dentro del rango del Manual)
- Color apagado/muted = región geográfica correcta pero fuera del rango altitudinal óptimo
- Gris oscuro = fuera del rango geográfico del Manual
- Puntos rojos = presencias GBIF confirmadas en Costa Rica
{fuente_manual}
{_regla}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{instruccion_pregunta}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FORMATO DE RESPUESTA OBLIGATORIO:

## Razonamiento
Máximo 5 viñetas. Interpreta solo lo que muestra el mapa: zonas cyan, muted, gris, y posición de los puntos GBIF. No describas la leyenda, interpreta lo que ves. Si la imagen no permite responder algún aspecto, indícalo explícitamente.

## Respuesta
3 a 4 oraciones. Responde la pregunta basándote únicamente en el mapa adjunto. Menciona zonas geográficas concretas de Costa Rica visibles en el mapa. Si algo no se puede inferir del mapa, dilo.
"""

# ─────────────────────────────────────────────────────────────────────────────
# T3 — Sistema completo: mapa botánico + mapa RF + métricas SHAP/AUC
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_T3 = """Eres un evaluador ecológico imparcial. Tu tarea es responder una pregunta concreta sobre la especie {species_name} cruzando ÚNICAMENTE las tres fuentes de evidencia proporcionadas a continuación.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUENTE 1 — MODELO CLIMÁTICO (datos cuantitativos)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Precisión del modelo (AUC): {rf_auc:.4f}
- Rango altitudinal observado en CR: {info_altitud}
- Variable climática más limitante: {var_humana} (impacto {direccion} sobre la idoneidad)
- Ecosistema de mayor idoneidad: {zona_humana}
- Variables secundarias: {secundaria_1}, {secundaria_2}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUENTE 2 — MAPA DE HÁBITAT BOTÁNICO (Imagen 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generado cruzando el Manual de Plantas de Costa Rica con las Unidades Fitogeográficas (Hammel 2014) y DEM.
- CYAN brillante = hábitat óptimo (región correcta + elevación dentro del rango del Manual)
- Color apagado/muted = región correcta pero fuera del rango altitudinal
- Gris oscuro = fuera del rango geográfico del Manual
- Puntos rojos = presencias GBIF confirmadas en CR
{fuente_manual}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUENTE 3 — MAPA PREDICTIVO CLIMÁTICO (Imagen 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Distribución modelada a partir de variables bioclimáticas WorldClim y presencias Mesoamericanas. Las zonas más oscuras indican mayor idoneidad climática predicha.

{_regla}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{instruccion_pregunta}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FORMATO DE RESPUESTA OBLIGATORIO:

## Razonamiento
Máximo 5 viñetas. Cruza las tres fuentes: ¿coinciden el mapa botánico y el modelo climático? ¿Los puntos GBIF caen dentro del hábitat óptimo? ¿La variable limitante explica la distribución observada? Si hay discrepancias entre fuentes, señálalas. Si algo no está en los datos, no lo infiereas.

## Respuesta
3 a 4 oraciones. Responde la pregunta usando únicamente la evidencia cruzada de las tres fuentes. Menciona zonas geográficas concretas de Costa Rica. Si los datos no permiten responder algún aspecto, indícalo.
"""

# Alias para compatibilidad con código existente (main.py usa BIMODAL_PROMPT para T3)
BIMODAL_PROMPT = PROMPT_T3

# Export all tier prompts
TIER_PROMPTS_ALL = {"T0": PROMPT_T0, "T1": PROMPT_T1, "T2": PROMPT_T2, "T3": PROMPT_T3}