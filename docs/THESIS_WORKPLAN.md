# Thesis work plan — from pipeline to truthful answers

Written 2026-09-12 after three review sessions. Supersedes the evaluation design in
`docs/evaluation_framework.md`, `PLAN_questions_and_evaluation.md` and
`analysis/statistical_plan.md` (archive them, do not delete).

Related plans still in force: `docs/SHIP_PLAN.md` (bachelor handoff, time-critical),
`docs/MCP_PLAN.md`, `mpcr_rag/PLAN.md`.

---

## The thesis in one paragraph

For a regional tropical flora, how truthful are general-purpose LLMs on the everyday
questions people actually ask; which of those questions can the digitized scientific
record (flora + occurrences + environmental layers) answer; and does grounding answers in
that evidence make them more trustworthy, including knowing when not to answer?

| | Contribution | Kind |
|---|---|---|
| K1 | Truthfulness of LLMs along the long tail of the CR flora, verified claim by claim | Empirical |
| K2 | Which everyday questions the record can answer, and how well data-derived inferences (e.g. sun/shade from land cover) agree with experts | Ecological informatics |
| K3 | Evidence-tagged answers with abstention; precision vs coverage trade-off | Design |
| R  | Expert-validated question benchmark + trait table for 5,791 CR species | Resource |

BIP paper = chapter 1 (grounding in the flora text). This plan = the rest.

---

## Track 0 — Housekeeping (today, ~45 min)

- [x] **0.1** Commit the uncommitted work in logical commits (2026-09-12: 5adfa14,
      4757226, 4edb1bb, 8782afb)
- [x] **0.2** Tag `bip-2026-submission` → 4cff428 (last commit on submission day).
      Verified: `make_report` from the tag regenerates `manual_vs_gbif_results.md`
      identically except the date line
- [x] **0.3** Backup `C:\Users\Jose\Documents\Tesis\backups\regiones_botanicas_20260912.zip`
      + `.sha256`. Still TODO: an off-machine copy (Drive sync is not a backup)
- [ ] **0.4** The Supabase project behind `DATABASE_URL` no longer resolves ("tenant/user
      not found" on 2026-09-12). Check the dashboard: the expert-review tables
      (`experiment/db.py`) lived there

---

## Track A — Decisions (gate everything; take to the tutor)

- [ ] **A1** Thesis framing K1/K2/K3/R (above)
- [ ] **A2** Question types in scope and the answer key for each:

      | Type | Example | Answer key |
      |---|---|---|
      | Where | ¿La veo en Braulio Carrillo? | Manual regions + held-out GBIF |
      | Elevation | ¿A qué altura? | Manual |
      | Phenology | ¿Cuándo florece? | Manual + GBIF dates |
      | Recognition | ¿Cómo la reconozco? ¿Color de flor? ¿Altura? | Manual morphology / traits |
      | Light | ¿Sol o sombra? | Expert labels (+ land-cover proxy) |
      | Cultivation | ¿Crece en mi finca a 1200 m? | Climate envelope + experts |
      | Out of scope for the record | usos, toxicidad | Correct behaviour = abstain |

- [ ] **A3** Systems compared: closed-book frontier LLM · frontier LLM + web search ·
      open local model · grounded system · grounded without data-derived evidence.
      Agent arm later, time-boxed.
- [ ] **A4** Expert commitment: contribute questions; label traits for ~100 species
      (~3–4 h each, BEFORE seeing any system output); adjudicate claims; blind preference.
- [ ] **A5** Retire: M1–M5 rubric, T0/T1/T3 factorial, question bank v1, SHAP/LIME/CNN
      as paper content. Re-pre-register the analysis plan in a new commit before any
      full run.

---

## Track B — Full Manual content in the catalog

Finding: `fichas.sqlite.full_text` is header + synonyms + distribution only (median 356
chars). The segmenter already captures morphology (~72%), discussion (~97%) and genus
description (~66%) in `RawFicha` — they are dropped at persistence.

Segmenter defects found while doing B (2026-09-12), all fixed in the same change:
- Genus boundaries never detected (regex expected a newline `_clean` had removed) →
  every species inherited the first genus's description, and wrong inherited habits
- ~1,200 species headers missed: citations without "vol: page" (`L., Sp. pl. 342.
  1753.`) failed the header test; their text leaked into the previous species and the
  species never reached the catalog (5,791 → ~6,939)
- 77 figure credits ("… Cortesía Flora of …") parsed as headers; dedupe preferred the
  polluted duplicate, so some species carried another species' distribution
- Morphology/discussion spanning several blocks (page breaks) kept only the first block
- Remaining: 54 species with no detected distribution paragraph (mostly orchids)

- [x] **B1** Persist `morphology`, `discussion`, `genus_description` (in `ficha_json`;
      `full_text` now header + morphology + distribution + discussion)
- [x] **B2** Dehyphenation in all sections incl. distribution (+79 regions / 64 spp, 0 lost)
- [x] **B3** Re-ingest + `python -m mpcr_rag.eval.catalog_diff OLD NEW`. Result: 5,791 →
      **6,946** species (+1,159; 4 removed: 1 junk, 3 had a neighbour's distribution).
      22 distribution swaps, all verified corrections against the PDF blocks. Other geo
      changes additive: +140 spp regions, +101 flowering, +33 vertientes; only loss =
      a spurious region guessed from a truncated paragraph. Habits: coverage 91% → 95%,
      species with ≥4 habits 6.9% → 2.1% (family-level garbage removed)
- [x] **B4** 50-entry hand check (20 added, 20 kept, 10 longest discussions): all 50
      correctly attributed. Found and fixed: distribution tails split at block breaks
      (164 spp), glued figure credits/labels, genus species-count sentences. Not fixed:
      rare appendix species / next-genus bibliography at the end of a discussion (2/50);
      8 stray soft hyphens; 54 species with no detectable distribution (mostly orchids)
- Pre-change backup: `mpcr_rag/data/fichas.pre_trackB_20260912.sqlite`
- [ ] **B5** Expose new sections in MCP `get_species` and in the REST contract
      (do this BEFORE `SHIP_PLAN` step 2.6 freezes `openapi.json`)
- [ ] **B6** Structured traits via LLM with a fixed schema: max height, flower color,
      fruit type/color (habit exists). Each value tagged `species` vs `genus` level.
      Local model or gpt-4o-mini (a few dollars)
- [ ] **B7** Validate B6 against 100 hand-checked fichas → precision/recall per trait
- Not doing: identification keys, Vols VII–VIII, leaf dimensions (OCR drops the `×`)

---

## Track C — pgvector replaces Pinecone

Decision: **replace, not add.** SQLite stays the portable source of truth (ingest writes
it; bachelor bundle ships it). Postgres becomes the query index synced from SQLite — the
same role Pinecone has today. Not on the pilot's critical path: per-species questions use
keyed lookup, no vectors.

Why now: Track B forces a re-embed anyway; removes a vendor key from the paper's
reproducibility; filters + vectors in one SQL query; Supabase Postgres already hosts the
experiment tables and ships pgvector.

- [ ] **C1** Local embeddings: `intfloat/multilingual-e5-large` (same model as Pinecone's
      hosted one, for BIP comparability), `passage:` / `query:` prefixes, on the 4070 Ti
- [ ] **C2** Schema: `fichas` (filter columns: `elev_min_eff`, `elev_max_eff`, `family`,
      `endemic`, arrays for `regions`/`vertientes`/`forest_types`/`habits`/months with GIN
      indexes, full JSONB) + `ficha_chunks(vector_id, section, text, embedding vector(1024))`
      with HNSW index. Sections: `distribution` (BIP-comparable) and `description`
      (morphology + discussion)
- [ ] **C3** `mpcr_rag/store/pg_client.py`: `sync_from_sqlite()`, `search(query, filters, k, section)`
- [ ] **C4** Port `retriever.build_filter` → SQL `WHERE`; keep `pattern_b` signature so
      `answer.py` and the MCP server do not change. Switch via
      `MPCR_VECTOR_BACKEND=pinecone|pgvector` until parity is shown
- [ ] **C5** Parity check: BIP eval queries on both backends, overlap@k + the
      rag_vs_baseline numbers on a subsample
- [ ] **C6** Hosting: Supabase for development (free tier: 500 MB, pauses when idle —
      fine for research). `pgvector/pgvector` container in the mini-PC compose for the
      public artifact; same code, different `DATABASE_URL`
- [ ] **C7** Keep a single access module rule (SHIP_PLAN 2.3): nothing outside `store/`
      touches SQL

---

## Track D — Evidence sources (feed K2)

- [ ] **D1** Evidence cache per species: build GBIF points, maps, envelope, RF outputs
      ONCE; generation reads the cache (today `run_experiment.py` recomputes per tier)
- [ ] **D2** One GBIF source (`gbif_map.get_points`) for map and model; minimum-records
      rule (e.g. >=15 unique cells) for model-based evidence
- [ ] **D3** Land cover at occurrence points (ESA WorldCover 10 m), post-2000 records →
      open vs closed fraction = sun/shade proxy
- [ ] **D4** `climate_at_location(place)` + existing envelope → cultivation questions
- [ ] **D5** RF upgrades (if RF stays as evidence): 5-fold spatial CV with mean ± SD,
      Boyce + TSS, thinning to one record per cell, balanced RF, SHAP on all data
- [ ] **D6** If the RF image is ever shown to a model: fix `prompt_templates.py:92`
      (says darker = higher and "presencias Mesoamericanas"; surface is magma, lighter =
      higher, CR-only) and stop drawing points in cyan

---

## Track E — New evaluation layer

- [ ] **E1** Question bank v2: everyday questions, Spanish, types from A2, including
      expert-contributed ones
- [ ] **E2** Species sampler from `fichas.sqlite` (replaces the obsolete CSV in
      `run_experiment.py:30`): stratified by popularity (GBIF count, Wikipedia page),
      seeded, frozen list committed, replacement rule declared in advance
- [ ] **E3** Generation harness: systems × questions × species; fixed temperature; log
      model version, date, tokens, cost; web-search via APIs (OpenAI / Gemini grounding /
      Perplexity), not chat UIs
- [ ] **E4** Grounded answer format: every claim tagged
      `Manual (p.) | occurrence data (n, method) | related species | no evidence`; abstain
      when no evidence
- [ ] **E5** Claim extraction (LLM) + hand spot-check of a sample
- [ ] **E6** Automatic claim verification against Manual fields, B6 traits, GBIF + DEM,
      envelope, land cover → `supported | contradicted | unverifiable`
- [ ] **E7** Expert app v2 (reuse `app/app.py` auth + Postgres): trait labeling screen,
      claim adjudication, blind side-by-side preference for N systems (not fixed A/B)
- [ ] **E8** Metrics: factual precision, contradiction rate, unverifiable rate,
      abstention quality, informativeness, accuracy–coverage curves, accuracy vs
      popularity (mixed-effects logistic, species random effect), Krippendorff's α
- [ ] **E9** Re-pre-register the analysis plan (new commit) before the full run

---

## Track F — Pilot (go / no-go for the full design)

- [ ] **F1** 20 species (popularity-stratified) × 6 question types × 3 systems
      (closed-book, web search, grounded)
- [ ] **F2** Claim extraction + one expert adjudicates
- [ ] **F3** Decide: is the long-tail gap real? Which framing gets the emphasis (K1 vs K2)?
- Needs explicit approval for API spend. Full run only on explicit request.

---

## Track G — Parallel, date-driven

- **Bachelor handoff** — `docs/SHIP_PLAN.md`, ~5.5 focused days, Phases 1 → 2.6 are the
  critical path. Do B1–B5 first so the frozen contract includes the new ficha sections.
- **BIP** — notification 2026-09-25; camera-ready 2026-10-12 (TODOs in `mpcr_rag/paper/main.tex`).

---

## Suggested order

| When | Work |
|---|---|
| Today (few hours) | Track 0 · B1–B4 |
| Next 2 days | B5 · SHIP Phase 1–2 → send contract to the students |
| Tutor meeting | Track A |
| Days 3–5 | E1, E2, E4, E5 (minimal) · D1, D3 prototype · SHIP 3–4 |
| Week 2 | Pilot (F) · C1–C5 in parallel · B6–B7 |
| Week 3+ | D5 · E6–E9 · expert sessions · full run (on request) |
| Late Sept / early Oct | BIP notification + camera-ready |
