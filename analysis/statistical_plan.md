# CR-BioLM — Plan de Análisis Estadístico (Pre-registrado)

**Estado**: PRE-REGISTRADO — este documento fue comprometido a git ANTES del run completo de 100 especies.
Citar el hash de commit en el capítulo de Métodos de la tesis para demostrar que las decisiones
analíticas no fueron post-hoc.

**Fecha de pre-registro**: 2026-04-16
**Experimento objetivo**: Run completo de 100 especies × 4 tiers × 2 generadores = 800 llamadas generadoras
**Métrica primaria**: `score_compuesto` por especie × tier × generador

---

## 1. Hipótesis pre-especificadas

### Hipótesis principales (por estrato)

- **H1 (Estrato A — preguntas tier_min=T1, todas las condiciones):**
  El score medio aumenta monotónicamente: T0 < T1 < T2 < T3.
  Justificación: cada tier agrega contexto relevante para preguntas respondibles con GBIF puro.

- **H2 (Estrato B — preguntas tier_min=T2):**
  T0 y T1 tienen rendimiento sustancialmente inferior a T2 y T3.
  T2 y T3 no difieren significativamente entre sí en este estrato
  (el mapa botánico ya contiene la información necesaria; RF agrega poco en estas preguntas).

- **H3 (Estrato C — preguntas tier_min=T3):**
  T3 supera significativamente a todos los tiers inferiores.
  T0 muestra la tasa de rechazo más alta (esperado: M3 bajo, M5 alto si rechaza correctamente).

### Hipótesis secundarias

- **H4 (M5 en T0):** T0 tendrá M5 promedio ≥ T1 en preguntas de Estrato C (las negativas calibradas
  son epistemológicamente más honestas que las respuestas de T1 con datos insuficientes).
- **H5 (Generadores):** No se hipotetiza superioridad a priori de ningún generador — análisis descriptivo.
- **H6 (Bonus T3):** El incremento de score de T2→T3 es mayor en Estrato C que en Estrato A
  (las métricas RF/SHAP son más diagnósticas para preguntas de Estrato C).

---

## 2. Tests primarios

### 2.1 Test de Friedman (por estrato, por generador)

- **Propósito**: comparar los cuatro tiers (T0/T1/T2/T3) dentro de cada especie como condiciones apareadas.
- **Unidad de análisis**: especie (N = 100, o subconjunto si hay datos faltantes).
- **Variable dependiente**: `score_compuesto` por especie × tier × generador.
- **Estructura**: 4 condiciones apareadas (T0, T1, T2, T3) por especie.
- **Umbral de significancia**: α = 0.05.
- **Se aplica por**: (a) Estrato A, (b) Estrato B, (c) Estrato C, (d) todos los estratos combinados.
- **Implementación**: `scipy.stats.friedmanchisquare`

### 2.2 Comparaciones post-hoc de Wilcoxon (si Friedman p < 0.05)

- **Pares pre-especificados** (6 combinaciones): T0–T1, T0–T2, T0–T3, T1–T2, T1–T3, T2–T3.
- **Corrección de múltiple comparación**: Bonferroni (α/6 = 0.0083).
- **Implementación**: `scipy.stats.wilcoxon` (signed-rank, pareado por especie).
- **Dirección**: prueba bilateral (no asumir dirección en post-hoc).

---

## 3. Tamaño del efecto

- **Kendall's W** por estrato: medida de concordancia del Friedman (efecto global).
  `W = χ²_Friedman / (k*(n-1))` donde k=4 tiers, n=n_especies.
- **r = Z / √N** por par Wilcoxon: tamaño del efecto por comparación pareada.
  Interpretación convencional: r < 0.1 = trivial, 0.1–0.3 = pequeño, 0.3–0.5 = mediano, > 0.5 = grande.

---

## 4. Intervalos de confianza (bootstrap)

- **Método**: percentile bootstrap, 10 000 remuestras, semilla = 42.
- **Variable**: score_compuesto medio por tier × estrato × generador.
- **CI al 95%** para:
  - Cada media de tier dentro de cada estrato.
  - Cada diferencia tier–tier dentro de cada estrato.
- **Implementación**: `scipy.stats.bootstrap`

---

## 5. Análisis secundarios (pre-especificados, descriptivos salvo indicación)

### 5.1 Por métrica individual

- Medias de M1, M3, M5 para todos los tiers T0–T3 (todas las preguntas).
- Medias de M2, M4 solo dentro de T3 (N/A para otros tiers).
- Distribución de M5 en T0 para verificar H4.

### 5.2 Por generador

- Comparación Claude Sonnet 4.5 vs GPT-4o en cada tier via Wilcoxon signed-rank, pareado por especie.
- Bonferroni si se compara en múltiples tiers simultáneamente (α/4 = 0.0125).

### 5.3 Sub-estudio de calibración experta (después del run principal)

- **Muestra**: 10 respuestas estratificadas por dificultad de especie y tier (ver plan Task 6).
- **Correlación de Spearman** entre score experto total y score del juez total.
- **Error absoluto medio (MAE)** por métrica entre experto y juez.
- **Análisis cualitativo de desacuerdos**: para respuestas con diferencia ≥ 2 puntos en cualquier métrica.
- *No* se calculará kappa ni Krippendorff's α con un solo experto — limitación explícita.

---

## 6. Verificaciones de robustez (pre-especificadas)

Estas re-ejecuciones del análisis principal se realizarán sobre subconjuntos para verificar que
los resultados no son artefactos de casos límite:

1. **Excluir especies con < 50 registros GBIF**: puede haber subrepresentación geográfica.
2. **Excluir especies donde altitud GBIF y Manual difieren > 500 m**: datos de altitud ruidosos
   pueden distorsionar el scoring de M2 en T3.
3. **Excluir el 10% de especies con score de juez más bajo/más alto** (si el juez emite confianza):
   respuestas en los extremos pueden tener más ruido de juez.

---

## 7. Lo que NO se analizará (exclusiones pre-registradas)

- **Segundo juez durante el run principal**: diferido a sub-estudio post-run para proteger presupuesto.
- **Panel de varianza de juez** (3 repeticiones por respuesta): diferido o eliminado.
- **Ablación de regla de aislamiento** (T1 sin instrucción de aislamiento): diferido.
- **Análisis de red de co-ocurrencia de variables SHAP**: fuera de alcance de esta tesis.

---

## 8. Software y versiones

```
Python        >= 3.11
scipy         >= 1.13    # Friedman, Wilcoxon, Spearman, bootstrap
scikit-posthocs >= 0.9   # post-hoc tests (si scipy resulta insuficiente)
pandas        >= 2.0
matplotlib    >= 3.8
seaborn       >= 0.13
```

Semilla global: `random.seed(42)`, `numpy.random.seed(42)`.

---

## 9. Outputs esperados del script `run_analysis.py`

1. `analysis/results/stats_all.json` — todos los estadísticos, p-values, effect sizes, CIs.
2. `analysis/results/report_results.md` — reporte Markdown para el capítulo de Resultados.
3. Figuras en `analysis/results/figures/`:
   - `tier_means_by_stratum.png` — barras con CIs bootstrap por estrato (4 tiers × 3 estratos).
   - `species_tier_lines.png` — líneas por especie mostrando el gradiente T0→T3.
   - `metric_heatmap.png` — heatmap de medias de cada métrica por tier.
   - `m5_distribution.png` — distribución de M5 por tier (para verificar H4).
