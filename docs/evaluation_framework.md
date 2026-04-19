# CR-BioLM: Framework de Evaluación Experimental
**Versión 1.0 — Abril 2026**

---

## 1. Objetivo

Determinar si — y cuánto — cada capa de información del pipeline CR-BioLM mejora la calidad de las respuestas ecológicas generadas por LLMs, tomando como referencia experta el *Manual de Plantas de Costa Rica* (Hammel et al.).

La pregunta de investigación central es:

> **¿Agrega valor medible cada componente del pipeline (mapa botánico, modelo RF, métricas SHAP) sobre un baseline puramente visual de registros GBIF, al responder preguntas ecológicas sobre especies de plantas costarricenses?**

---

## 2. Diseño Experimental

### 2.1 Tiers de información (variable independiente principal)

Cada especie se procesa bajo tres condiciones distintas. Cada tier agrega exactamente **un componente** sobre el anterior para permitir atribución limpia.

| Tier | Nombre | Imagen 1 | Imagen 2 | Métricas RF/SHAP | Texto Manual |
|------|--------|----------|----------|------------------|--------------|
| **T1** | Baseline GBIF | Mapa Mesoamérica (puntos GBIF planos) | — | — | — |
| **T2** | Mapa Botánico | Mapa hábitat Manual (cyan/muted/gris + GBIF CR) | — | — | — |
| **T3** | Sistema Completo | Mapa hábitat Manual | Mapa predictivo RF | ✓ SHAP + AUC | — |

**Nota crítica**: El texto del Manual de Plantas **nunca se inyecta** al LLM en ningún tier. Sirve exclusivamente como ground truth de evaluación.

### 2.2 Modelos generadores (variable independiente secundaria)

| Modelo | Proveedor | Rol |
|--------|-----------|-----|
| `openai/gpt-4o` | OpenAI vía OpenRouter | Generador principal |
| `anthropic/claude-sonnet-4-5` | Anthropic vía OpenRouter | Generador comparativo |

### 2.3 Tipos de pregunta (variable moderadora)

| Perfil | Código | Foco de evaluación |
|--------|--------|--------------------|
| Turista | P1 | Precisión geográfica — ¿menciona zonas correctas de CR? |
| Botánico | P2 | Variables climáticas — ¿identifica el factor limitante correcto? |
| Municipalidad | P3 | Razonamiento de impacto — ¿coherencia ecológica aplicada? |

En cada corrida batch, cada especie recibe **una pregunta aleatoria por perfil**, la misma pregunta en los tres tiers (para comparabilidad directa).

### 2.4 Corpus de evaluación

- **100 especies** del catálogo `picked_species_enhanced_clean.csv`
- Criterio de inclusión mínimo: ≥ 50 registros GBIF en CR
- Ground truth: ficha `{Especie}_ficha_MdP.txt` generada por el pipeline (datos del Manual)
- **Total de respuestas**: 100 especies × 3 tiers × 2 modelos × 3 perfiles = **1,800 respuestas**

---

## 3. Métricas de Evaluación

### 3.1 Rúbrica principal — LLM-as-Judge (G-Eval)

Basada en Liu et al. (2023) *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment* (EMNLP 2023). El juez recibe: la respuesta generada + la ficha MdP (ground truth) + la pregunta formulada.

#### M1 — Precisión Geográfica (0–3)
> ¿Las zonas geográficas mencionadas (vertiente, cordillera, región, cantón) son consistentes con el Manual?

| Score | Criterio |
|-------|----------|
| 0 | Zonas incorrectas, ausentes, o contradicen el Manual |
| 1 | Una zona correcta mencionada, pero incompleta o con errores |
| 2 | Zonas correctas pero falta especificidad (ej. dice "vertiente Caribe" sin mencionar cordillera) |
| 3 | Zonas completas, específicas y consistentes con el Manual |

#### M2 — Precisión Altitudinal (0–2)
> ¿El rango de elevación mencionado es consistente con el Manual? (Solo aplica si la ficha MdP tiene datos de altitud)

| Score | Criterio |
|-------|----------|
| 0 | Rango incorrecto, ausente, o contradice el Manual (error > 400m) |
| 1 | Aproximado — dentro de ±400m del rango del Manual |
| 2 | Correcto — dentro de ±200m del rango del Manual |

*Si la ficha MdP no tiene datos de altitud, esta métrica se excluye del cálculo de score compuesto para esa especie.*

#### M3 — Relevancia de Respuesta (0–3)
> ¿La respuesta contesta directamente la pregunta formulada con información específica?

| Score | Criterio |
|-------|----------|
| 0 | No responde la pregunta o respuesta genérica sin datos de la especie |
| 1 | Responde parcialmente — toca el tema pero sin especificidad |
| 2 | Responde la pregunta pero incluye información irrelevante o divaga |
| 3 | Responde directamente, concisamente y con datos específicos de la especie |

#### M4 — Variable Climática Correcta (0–2)
> ¿La variable identificada como más limitante es ecológicamente plausible para el hábitat descrito en el Manual?
> **Solo aplica para perfil P2 (botánico)**

| Score | Criterio |
|-------|----------|
| 0 | Variable incorrecta o ausente |
| 1 | Variable plausible pero no la principal según el hábitat del Manual |
| 2 | Variable correcta y mecanismo ecológico explicado correctamente |

### 3.2 Score compuesto normalizado

```
Score_base(T, M, P) = (M1 + M2 + M3) / 8     [para P1, P3]
Score_base(T, M, P) = (M1 + M2 + M3 + M4) / 10  [para P2]
```

Donde T = tier, M = modelo, P = perfil. Resultado en rango [0, 1].

### 3.3 Métrica secundaria — BERTScore

BERTScore (Zhang et al., 2020) mide similitud semántica entre la respuesta generada y la ficha MdP usando embeddings contextuales (DeBERTa-v3). Se reporta como métrica de screening complementaria, no como métrica primaria, dado su conocido desacoplamiento con el juicio humano en textos especializados.

```python
# Implementación: bert-score library
# Modelo recomendado: microsoft/deberta-xlarge-mnli
# Se computa F1 entre respuesta y ficha MdP
```

### 3.4 Consistencia interna (estabilidad)

Para un subconjunto de 20 especies (20% del corpus), cada tier × modelo corre **dos veces** con la misma pregunta. Se calcula la varianza del score compuesto entre corridas. Alta varianza indica inestabilidad del sistema.

---

## 4. Implementación del LLM-Judge

### 4.1 Arquitectura del juez

Modelo juez: `google/gemini-2.0-flash-001` — familia de modelos completamente distinta a los generadores (OpenAI y Anthropic), lo que elimina self-preference bias por diseño sin necesidad de evaluación cruzada. Un único juez además garantiza consistencia en todos los scores.

| Rol | Modelo | Justificación |
|-----|--------|---------------|
| Generador A | `openai/gpt-4o` | Generador principal |
| Generador B | `anthropic/claude-sonnet-4-5` | Generador comparativo |
| **Juez único** | `google/gemini-2.0-flash-001` | Familia distinta → sin self-preference bias; consistente; bajo costo |

### 4.2 Prompt del juez (G-Eval con CoT)

```
Eres un evaluador experto en botánica costarricense. Se te proporciona:
1. Una PREGUNTA formulada al sistema
2. Una RESPUESTA generada automáticamente
3. La FICHA DE REFERENCIA del Manual de Plantas de Costa Rica (ground truth)

Tu tarea es evaluar la respuesta en las siguientes dimensiones.
Razona paso a paso antes de asignar cada puntaje.

PREGUNTA: {pregunta}
RESPUESTA A EVALUAR: {respuesta}
FICHA DE REFERENCIA (Manual de Plantas CR): {ficha_mdp}

Evalúa y responde en formato JSON:
{
  "razonamiento_M1": "...",
  "M1_precision_geografica": <0|1|2|3>,
  "razonamiento_M2": "...",
  "M2_precision_altitudinal": <0|1|2|"N/A">,
  "razonamiento_M3": "...",
  "M3_relevancia_respuesta": <0|1|2|3>,
  "razonamiento_M4": "...",  // solo si perfil es botanico
  "M4_variable_climatica": <0|1|2|"N/A">
}
```

### 4.3 Validación del juez

Antes de la evaluación completa, se ejecuta un **piloto de validación** con 15 especies seleccionadas a mano:

1. El juez LLM evalúa las 15 especies (45 respuestas por tier)
2. Un botánico experto evalúa las mismas 45 respuestas con la misma rúbrica
3. Se calcula Cohen's Kappa (κ) por métrica

| κ | Interpretación | Acción |
|---|----------------|--------|
| ≥ 0.75 | Excelente | Proceder con evaluación completa |
| 0.60–0.74 | Sustancial | Proceder con nota metodológica |
| 0.40–0.59 | Moderado | Refinar prompt del juez e iterar |
| < 0.40 | Débil | Rediseñar rúbrica |

*Umbral mínimo aceptable para publicación: κ ≥ 0.60 (Landis & Koch, 1977)*

---

## 5. Análisis Estadístico

### 5.1 Pregunta 1 — ¿Difieren los tiers en calidad de respuesta?

**Test**: Kruskal-Wallis H (no paramétrico, scores ordinales, 3 grupos independientes)
**Post-hoc**: Mann-Whitney U con corrección de Bonferroni para comparaciones pareadas (T1 vs T2, T2 vs T3, T1 vs T3)
**Efecto**: Epsilon-cuadrado (ε²) — thresholds: pequeño ≥0.01, mediano ≥0.06, grande ≥0.14

### 5.2 Pregunta 2 — ¿El tipo de pregunta modera la ganancia entre tiers?

**Test**: Interacción tier × perfil usando modelo ordinal de efectos mixtos (proportional odds logistic regression)
**Hipótesis**: La ganancia T1→T3 es mayor en P2 (botánico) que en P1 (turista) para la métrica M4

### 5.3 Pregunta 3 — ¿Difieren GPT-4o y Claude Sonnet en su aprovechamiento del contexto?

**Test**: Mann-Whitney U por tier y por métrica
**Hipótesis**: Diferencia mayor en T3 (sistema completo) que en T1 (baseline), donde ambos modelos solo ven puntos

### 5.4 Visualizaciones requeridas

| Figura | Tipo | Variables |
|--------|------|-----------|
| Fig. 1 | Heatmap 3×2 | Score compuesto medio por tier × modelo |
| Fig. 2 | Boxplot facetado | Score compuesto por tier, separado por perfil (P1/P2/P3) |
| Fig. 3 | Barplot apilado | M1/M2/M3 por tier — contribución de cada dimensión |
| Fig. 4 | Scatter | BERTScore vs Score-juez (correlación de métricas) |
| Fig. 5 | Heatmap κ | Acuerdo juez-experto por métrica (validación piloto) |

---

## 6. Hipótesis de Investigación

**H1** (principal): El score compuesto de T3 es significativamente mayor que T1 (p < 0.05, ε² ≥ 0.06).

**H2** (ganancia del mapa botánico): T2 > T1 en M1 (precisión geográfica) para perfiles P1 y P3.

**H3** (ganancia de SHAP): T3 > T2 en M4 (variable climática) para perfil P2, pero no en M1 o M2.

**H4** (interacción): La ganancia T1→T3 es mayor para P2 que para P1 (el sistema completo agrega más valor en preguntas técnicas).

**H5** (modelos): GPT-4o y Claude Sonnet muestran scores similares en T1 pero divergen en T3, donde la capacidad de integrar contexto multimodal diferencia los modelos.

---

## 7. Limitaciones y Mitigaciones

| Limitación | Mitigación |
|------------|------------|
| Self-preference bias del juez LLM | Juez de familia distinta (Gemini) — sin relación con los generadores |
| Verbosity bias (respuestas largas favorecidas) | Prompt del juez instruye explícitamente a no favorecer longitud |
| Ground truth incompleto (fichas MdP sin altitud) | M2 se excluye del score compuesto cuando N/A |
| Variabilidad estocástica del LLM | Temperatura=0 en generación; subconjunto de 20 especies corrido 2 veces |
| Muestra pequeña (100 especies) | Se reportan effect sizes (ε²) además de p-values |
| Juez sin conocimiento botánico real | Validación piloto con botánico experto (κ umbral 0.60) |

---

## 8. Cronograma de Implementación

| Fase | Tarea | Dependencia |
|------|-------|-------------|
| 1 | Implementar modo T1/T2/T3 en pipeline (`--tier`) | — |
| 2 | Implementar módulo `llm/judge_client.py` | Fase 1 |
| 3 | Correr piloto de validación (15 especies, 3 tiers) | Fase 2 |
| 4 | Calcular κ piloto, refinar prompt si κ < 0.60 | Fase 3 |
| 5 | Correr batch completo (100 especies × 3 tiers) | Fase 4 |
| 6 | Correr juez sobre todas las respuestas | Fase 5 |
| 7 | Análisis estadístico y visualizaciones | Fase 6 |
| 8 | Redacción de resultados | Fase 7 |

---

## 9. Referencias

- Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023. https://arxiv.org/abs/2306.05685
- Liu, Y. et al. (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.153/
- Zhang, T. et al. (2020). *BERTScore: Evaluating Text Generation with BERT*. ICLR 2020.
- Landis, J.R. & Koch, G.G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159–174.
- Liang, P. et al. (2024). *Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge*. https://arxiv.org/html/2410.02736v1
- Pörtner, R. et al. (2025). *Large language models possess some ecological knowledge, but how much?* bioRxiv. https://www.biorxiv.org/content/10.1101/2025.02.10.637097v3.full
- Hammel, B. et al. (Eds.). *Manual de Plantas de Costa Rica*. Vols. I–VIII. Missouri Botanical Garden Press.
