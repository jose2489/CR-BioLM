# PLAN: Question Bank, Tier Restructure & Evaluation Framework Upgrade

**Status:** Decisions resolved — ready for staged implementation
**Date:** 2026-05-04
**Scope:** Three coordinated changes — (1) restructure tiers (drop T2, fix ficha leakage), (2) tighten question bank to two personas with strict geospatial focus, (3) upgrade evaluation framework per the multi-tiered LLM-as-Judge research.

**Demo target:** working pipeline at **25 species** for next-week expert demo. Pre-demo unit test at 3 species. Full thesis run at 100 species after demo feedback is incorporated.

---

## Part 0 — Tier Restructure (CRITICAL — affects everything else)

### 0.1 Goal
Reduce experiment to three tiers that represent **clean, non-overlapping ablations**:

| Tier | Context provided to LLM | Purpose |
|---|---|---|
| **T0** | Nothing (no images, no data) | Pure parametric-knowledge baseline |
| **T1** | GBIF Mesoamerica map only | Adds raw observational geospatial signal |
| **T3** | GBIF map + RF predictive map + SHAP variables + altitude range | Full pipeline — what the system uniquely contributes |

**Drop T2** (the Hammel/Manual habitat map as a standalone tier). The botanical map currently mixes Manual textual data with Hammel polygons + DEM — meaning T2 already contains ground-truth-derived information. Removing it eliminates a tier that wasn't a clean ablation, and simplifies the comparison.

### 0.2 Ficha leakage in T3 — must fix

**Problem:** [llm/openrouter_client.py:86-94](llm/openrouter_client.py#L86-L94) currently injects `texto_manual` (the Manual de Plantas ficha — i.e. the ground truth) into T2 and T3 prompts via the `fuente_manual` slot. This means:

- **T3 generation reads the same ficha that T3 evaluation uses as ground truth.**
- High M1/M3 scores at T3 may be partially explained by direct text overlap, not geospatial reasoning.
- This invalidates the core comparison of the thesis.

**Fix:**
1. Remove `fuente_manual` block from `PROMPT_T3` in [llm/prompt_templates.py:98](llm/prompt_templates.py#L98).
2. Remove the `texto_manual` parameter path in [llm/openrouter_client.py:86-94](llm/openrouter_client.py#L86-L94).
3. Keep the **derived map** (the predictive RF map) as the only Manual-derived artifact passed to T3 — and only because it's a transformation, not the text.
4. **Decision (resolved):** keep the Hammel-derived predicted-habitat map in T3. It is the most informative Manual-derived artifact we produce, and it's a transformation (phytoregion names → polygons → DEM filter), not text leakage. **Rename in code/UI** to "Mapa de hábitat predicho (Manual + Hammel + DEM)" so the derivation chain is explicit, and document it in [README.md](README.md). The Manual text itself never reaches the LLM.

### 0.3 Migration impact
- **Existing runs (EXP-* directories):** **archive** to `experiment/runs/_archive_pre_cleanup/`. Add a `README.md` in that folder noting these contain ficha-text leakage and are NOT thesis-grade results. Hide them from the report UI by default (toggle to show archived).
- **New runs:** generate only T0/T1/T3 artifacts.
- **Code changes:** [main.py](main.py), [experiment/run_experiment.py](experiment/run_experiment.py), `TIER_PROMPTS_ALL` dict, report templates.
- **Score formula:** simplifies — no T2 branch in `_score_compuesto`.
- **Re-run required:** all baseline numbers in the thesis come from clean runs **after the leakage fix**.

---

## Part 1 — Question Bank Refinement

### 1.1 Goals
- **Drop `municipalidad`.** Limit to two personas: `botanico` and `turista`.
- **Geospatial-only constraint.** Every question must require spatial reasoning to answer; the pipeline's value-add is geographic, not general botanical knowledge.
- **Botanico questions grounded in real botanist research patterns** (not invented from intuition).
- **Re-stratify** for new T0/T1/T3 tier set: drop `stratum B` (which mapped to T2).

### 1.2 New strata mapping

- **Stratum A** = `tier_min = T1` — answerable with GBIF points alone
- **Stratum C** = `tier_min = T3` — requires RF/SHAP/altitudinal context

(Old stratum B questions get re-evaluated: most will collapse into A — answerable from GBIF — or C — needing the full RF pipeline.)

### 1.3 Audit of current bank ([utils/question_bank.py](utils/question_bank.py))

**Botanico (11 questions) — keep / modify / drop:**

| # | Current question | Old stratum | New stratum | Decision |
|---|---|---|---|---|
| 1 | Variable climática más determinante | C | C | Keep |
| 2 | Rango altitudinal óptimo vs Manual | B | C | Move to T3; T1 has no altitude reference |
| 3 | Coherencia modelo climático vs Manual | C | C | Keep |
| 4 | Tipo de bosque define el nicho realizado | A | A | Modify: add "...en qué vertiente/cordillera" |
| 5 | Influencia de precipitación trimestre seco | C | C | Keep |
| 6 | Zonas alta idoneidad sin GBIF confirmado | C | C | Keep |
| 7 | Preferencia por vertiente (Caribe vs Pacífico) | A | A | Keep |
| 8 | Sensibilidad a temperatura media anual | C | C | Modify: "...y en qué zonas de CR" |
| 9 | Distribución predicha en Mesoamérica vs ecología | C | C | Keep |
| 10 | GBIF confirma o contradice hábitat Manual | A | A | Keep |
| 11 | Factores climáticos secundarios | C | C | Modify: add "...y en qué subregiones" |

**Turista (10 questions):** all currently A or B. Q3 ("época florece") → modify to add zone reference. Old-B "altura para encontrar" → reclassify as C (no altitude in T1) or drop (might be too narrow for T3 framing).

### 1.4 New botanico questions (research-backed)

Based on common research patterns of tropical plant ecologists (MdP review tasks, herbarium curation, EIA support, range modeling work):

1. **Range edge detection** — "¿Dónde se localizan los límites altitudinales y geográficos de distribución de esta especie en Costa Rica, y qué factores climáticos los explican?" *(stratum C)*

2. **Endemism / range gap** — "¿Existen disyunciones notables en la distribución conocida de esta especie y dónde se ubican geográficamente?" *(stratum A)*

3. **Climate change vulnerability (spatial)** — "¿Qué zonas actuales de presencia de esta especie son más vulnerables a desplazamiento bajo escenarios de aumento de temperatura?" *(stratum C)*

4. **Habitat overlap with disturbance** — "¿En qué zonas de Costa Rica el hábitat predicho de esta especie se solapa con áreas de bosque ya intervenido o fragmentado?" *(stratum C)*

5. **Sampling bias diagnosis** — "¿La distribución observada en GBIF refleja un sesgo de muestreo geográfico (sobre-representación cerca de carreteras o estaciones biológicas)?" *(stratum A)*

6. **Topographic specificity** — "¿La especie muestra preferencia por alguna posición topográfica específica (laderas, cumbres, fondos de valle) según la distribución de elevación de sus registros?" *(stratum C — needs altitudinal data)*

### 1.5 Final structure

- **`turista`**: ~10 questions, strata A only (or A + occasional C if altitude needed)
- **`botanico`**: ~17 questions, mix of A and C
- **Drop**: `municipalidad` block + `CANTONES_CR` + `PROYECTOS_TIPO` constants
- **Update**: `experiment/run_experiment.py`, `main.py` CLI flags (`--canton`, `--proyecto`), and references in [experiment/report/templates/](experiment/report/templates/)

### 1.6 Validation step before merging
Each new/modified question must be **manually answered** for at least one species using each tier's available context, to confirm it is genuinely answerable at its declared `tier_min` and not below.

---

## Part 2 — Evaluation Framework Upgrade

### 2.1 Current state ([llm/judge_client.py](llm/judge_client.py))
- **Single-judge:** Gemini 2.0 Flash only.
- **5 metrics:** M1 (geo precision), M2 (altitudinal use), M3 (relevance), M4 (climate var SHAP), M5 (analytical depth).
- **No taxonomy validation, no GBIF cross-check, no biogeographic alignment, no LevelEval calibration, no human review tier.**

### 2.2 Target architecture: 3-tier evaluation

#### **Tier 1 — Automated Linguistic Screening (LLM-as-Judge)**
*Scalable, runs on every response, fast.*

Existing M5 + M3 partially cover this. **Add:**
- **L1 — Spanish linguistic coherence** (0–3): naturalness of Spanish, technical register appropriate to persona.
- **L2 — Structural completeness** (0–3): expected components present (distribution, habitat, direct response).
- **L3 — Internal consistency** (0–3): no self-contradictions in the response.

Cheap — Gemini Flash handles in same call (extend `JUDGE_PROMPT`).

#### **Tier 2 — Domain-Specific Validation (programmatic + LLM)**
*Highest value-add tier — currently mostly missing.*

- **D1 — Taxonomic accuracy** (binary): species name validates against GBIF backbone taxonomy via `pygbif`. Programmatic, no LLM. Stored as `taxonomy_valid: true/false`.
- **D2 — Geospatial consistency** (0–2): zones mentioned by the LLM intersect with the GBIF occurrence convex hull / RF high-idoneidad mask. Programmatic: extract zone names → match against CR gazetteer → check spatial overlap with rasters.
- **D3 — Elevation plausibility** (0–2): elevations mentioned fall within `[min - 200m, max + 200m]` of GBIF/Manual range. Programmatic regex + numeric check (replaces current loose M2).
- **D4 — Biogeographic alignment** (0–2): zones match Manual's regional vocabulary (vertiente Pacífica, Cordillera de Talamanca, etc.) and align with Hammel 2014 unidades fitogeográficas. LLM judge with controlled vocabulary.

**Implementation:** new module `experiment/validators/domain_validators.py` for programmatic checks; extend `judge_client.py` for D4.

#### **Tier 3 — Expert Human Validation (sample-based)**
*Calibration only, not full coverage.*

- For demo / thesis: 15% random sample + all "uncertain" cases (ensemble disagrees, see §2.3) + all endemic species.
- Build a simple review UI extending the existing FastAPI viewer: pregunta + respuesta + ficha + automated scores → expert gives 1–5 overall + free-text comment.
- Store reviews in `experiment/runs/<EXP-ID>/human_reviews.json`.
- Use these to compute **judge-human agreement (Cohen's κ or Spearman ρ)** — required for thesis defense.

### 2.3 Multi-judge ensemble (free / low-cost)

Replace single-judge with a multi-judge ensemble using **free / low-cost APIs**. Judges are text-only (no images) → Groq's free tier is ideal.

- **Judge A:** Gemini 2.0 Flash via OpenRouter (current — cheap, near-free)
- **Judge B:** GPT-4o-mini via OpenRouter (cheap, different family from Gemini). Hold off on Claude Sonnet 4.6 for now to keep costs predictable.
- **Judge C (Groq free tier, when capacity allows):** Llama 3.3 70B Versatile or DeepSeek-R1-Distill via Groq. Used as **tie-breaker** for high-disagreement cases, and optionally as a third independent vote.

**Why Groq:** [config.py:23](config.py#L23) already pulls `GROQ_API_KEY`. Free tier covers Llama 3.3 70B and DeepSeek with rate limits suitable for our scale (≤25 species in the demo run = ~150 judge items). Both handle Spanish rubric scoring adequately.

**Rules:**
- Judges A and B score every response independently.
- **Self-enhancement guard:** if a judge's family appears in `modelo_generador`, swap it (e.g., generator = GPT-4o → don't use GPT-4o-mini as judge; use the Groq Llama judge instead for that call).
- **Final score per metric:** mean of Judges A and B.
- **Disagreement flag:** if |Judge A − Judge B| ≥ 2 on any metric → invoke Judge C (Groq) as tie-breaker AND flag for human review.
- **Cost impact:** Gemini Flash + GPT-4o-mini for 100 species × 3 tiers × 2 generators = 600 judge items × 2 judges = 1200 calls. Estimate: <$2 total. Groq tie-breaker is free.

**Implementation:**
- Parameterize `JudgeClient.__init__` to accept `(provider, model, api_key, base_url)` so the same class works for OpenRouter and Groq endpoints (both are OpenAI-compatible).
- Add `EnsembleJudge` wrapper that runs N judges, applies self-enhancement guard, aggregates scores, and triggers the Groq tie-breaker on disagreement.
- Persist all individual judge scores in `eval_*.json` (keys: `judge_A_scores`, `judge_B_scores`, `judge_C_scores`, `judge_aggregate`) so disagreements are auditable post-hoc.

### 2.4 Bias mitigation (per research)

- **Verbosity bias:** strengthen judge prompt: *"No premies respuestas más largas. Evalúa solo precisión técnica (metros exactos, nombres de zonas exactos)."*
- **Self-enhancement:** enforced via family-mismatch rule above.
- **Positional bias:** N/A (not pairwise). Note for future work.

### 2.5 LevelEval calibration pipeline

Validate the judge itself with synthetic responses at three quality levels.

**Three-stage rollout matching the experiment scale:**

| Stage | Species | Tiers | Levels | Total items | Purpose |
|---|---|---|---|---|---|
| Unit test | 3 | 3 (T0/T1/T3) | 3 | 27 | Sanity-check rubric and judge wiring after Part 0 lands |
| Demo calibration | 25 | 3 | 3 | 225 | Demo-grade calibration to show experts |
| Full thesis | 100 | 3 | 3 | 900 | Final calibration shipped with thesis |

**Levels:**
- **Level 1 (gold):** hand-written by you, correct facts, precise elevations, expert Spanish.
- **Level 2 (mid):** LLM-rewritten Level 1 with introduced minor errors (50% noise: wrong cordillera, ±300 m altitude).
- **Level 3 (poor):** LLM-rewritten Level 1 with major errors (wrong vertiente, irrelevant content).

Expected: judge ranks Level 1 > Level 2 > Level 3 with clear gap. If indistinguishable, refine rubric before trusting production scores. Store as `experiment/calibration/leveleval_<stage>_<date>.json`.

**Demo workflow:** run unit test (3) → review with you → run demo calibration (25) → present to experts next week → incorporate feedback → run full thesis (100).

### 2.6 Domain context in judge prompt

Per recommendation §6 of the research, **inject CR-specific context** into the judge prompt:

- Append fixed reference block to `JUDGE_PROMPT`:
  - **Biogeographic zones:** Península de Osa, Cordillera Volcánica Central, Cordillera de Talamanca, vertiente Caribe vs Pacífico, valles intermontanos, llanuras del Norte/Tortuguero, etc.
  - **Manual conventions:** parenthetical notation for marginal elevation records (`1500 (–1800) m` = rare records up to 1800 m).
  - **Required morphological vocabulary** when relevant: estípulas, pecíolos, inflorescencias.

~300 tokens per call. Negligible cost, large quality gain.

### 2.7 Updated score composition

Current: `(M1 + M3 + M5) / 9` baseline + bonuses.
Proposed:

```
Linguistic block (Tier 1):  L1 + L2 + L3 + M5  (max 12)
Domain block (Tier 2):       D2 + D3 + D4 + M1  (max 11; D1 is binary gate)
T3 bonuses:                  M2 + M4            (max 4, T3 only)
Relevance gate:              M3                  (max 3)

score_compuesto = (M3 + L_block + D_block + bonuses) / max_possible
```

**D1 (taxonomy) hard cap at 0.1**: if `taxonomy_valid == false` → `score_compuesto = min(computed, 0.1)`. Set to 0.1 rather than 0 so the rubric still distinguishes "wrong species + total nonsense" from "wrong species + structurally coherent text," but ensures no taxonomy failure can earn a passing score.

**"Empty answer" guard (addresses well-structured-but-vacuous responses):** If M3 (relevancia) ≤ 1 → cap `score_compuesto` at 0.2. Reasoning: a fluent refusal or a vague non-answer should score like a human reviewer would score it (~0–0.2), not get rewarded for tone, structure, or hedge-quality. This prevents L1/L2/L3 + M5 from inflating scores when the response doesn't actually answer the question.

The two caps stack: a wrong-species + non-answer response is bounded by `min(0.1, 0.2) = 0.1`.

### 2.8 Hallucination & RAG-style faithfulness (deferred)

Research recommends Arize Phoenix for hallucination, RAG+MLflow for retrieval. **For thesis:**
- Heavy framework integrations. Defer to post-thesis.
- For now, **D2 + D3 (geospatial + elevation consistency)** serve as proxy hallucination checks since they're grounded in concrete GBIF/Manual data.

---

## Part 3 — Implementation Order

**Phase A — Pre-demo (target: ready for 25-species demo run next week)**
- [x] 1. **Tier restructure + ficha leakage fix** — Part 0. Removed `{fuente_manual}` from PROMPT_T3; dropped T2; updated TIER_PROMPTS_ALL; renamed "Mapa de hábitat botánico" → "Mapa de hábitat predicho (Manual + Hammel + DEM)"; removed T2 branch from main.py and openrouter_client.py.
- [x] 2. **Archive existing 16 runs** to `experiment/runs/_archive_pre_cleanup/` with README explaining ficha leakage contamination.
- [x] 3. **Question bank refinement** — Part 1. Dropped `municipalidad` block + `CANTONES_CR` + `PROYECTOS_TIPO`; restratified B→A/C; modified Q3(turista)/Q2(botanico)/Q4/Q8/Q11; added 6 new botanico questions (research-backed). Updated CLI args in main.py and run_experiment.py.
- [x] 4. **Multi-judge ensemble** — Part 2.3. Refactored `JudgeClient` to be provider-agnostic (model/api_key/base_url); added `EnsembleJudge` wrapper with self-enhancement guard, disagreement flag, and Groq tie-breaker. Kept `JudgeClientLegacy` for backwards compat.
- [x] 5. **Domain context + verbosity guard** in judge prompt — Part 2.4 + 2.6. Added `_CR_CONTEXT` block (~300 tokens) with CR biogeographic zones + Manual conventions + anti-verbosity instruction.
- [x] 6. **D1 taxonomy + D3 elevation** programmatic checks — `experiment/validators/domain_validators.py`. D1 via pygbif backbone (v2 API); D3 via regex elevation extraction vs GBIF+Manual reference window (tolerance ±200 m).
- [x] 7. **Score composition with D1 cap (0.1) + M3-empty-answer cap (0.2)** — Part 2.7. Both caps implemented in `_score_compuesto`; taxonomy_valid parameter wired through EnsembleJudge.
- [x] 8. **LevelEval script** — `experiment/calibration/leveleval.py`. Generates L1/L2/L3 synthetic responses via GPT-4o-mini, evaluates with ensemble judge, reports rank_ok per species. Run: `python experiment/calibration/leveleval.py --stage unit` (3 species).
- [x] 9. **Unit LevelEval passed** (3/3 rank OK): Pilea L1=0.818>L2=0.769>L3=0.200 · Miconia L1=0.885>L2=0.692>L3=0.462 · Carapa L1=0.885>L2=0.615>L3=0.231. Fixed Groq model name (`llama-3.3-70b-versatile`) + added 429 retry-with-backoff to JudgeClient. **Next:** demo calibration + 25-species pipeline run.

**Phase B — Post-demo (incorporate expert feedback, then thesis run)**
- [ ] 10. L1/L2/L3 linguistic metrics — extend judge prompt + JSON schema.
- [ ] 11. D2 + D4 (geospatial overlap + biogeographic alignment).
- [ ] 12. **Full thesis LevelEval (100 species)** + 100-species pipeline run.
- [ ] 13. Tier 3 human review session with the experts who attended the demo (already willing per Part 4 #7).

**Phase C — Post-thesis**
- [ ] 14. Phoenix/MLflow hallucination integration.

---

## Part 4 — Decisions resolved

1. **Drop municipalidad entirely.** No dormant code path; remove cleanly.
2. **Judge ensemble = Gemini Flash + GPT-4o-mini (paid, cheap) + Groq Llama 3.3 70B (free, tie-breaker).** Hold off on Claude Sonnet 4.6. Groq covers tie-break and self-enhancement-guard substitutions at zero marginal cost.
3. **LevelEval staged:** 3-species unit test → 25-species demo calibration → 100-species full thesis run. Demo runs next week; full run after expert feedback is incorporated.
4. **D1 cap at 0.1** (not 0). Plus a separate **M3 ≤ 1 cap at 0.2** to prevent well-structured non-answers from inflating scores via L1/L2/L3/M5. Caps stack (`min` of all applicable).
5. **Hammel-derived predicted-habitat map stays in T3** — it's the most informative artifact the pipeline produces. Rename in code/UI to "Mapa de hábitat predicho (Manual + Hammel + DEM)" with the derivation chain documented in [README.md](README.md). The Manual *text* never reaches the LLM; only the map (a transformation) does.
6. **Archive existing 14 runs** to `experiment/runs/_archive_pre_cleanup/` with a README documenting ficha leakage. Hide from the report UI by default.
7. **Tier 3 human review** — experts attending the demo are likely available for a structured review session afterward. Build the simple review UI as part of Phase B.

---

*End of plan. Implementation begins with Phase A, item 1 (tier restructure + ficha leakage fix).*
