# Integration plan — MPCR-RAG maps into the full pipeline + multi-provider LLM

Session goal: (A) make the full pipeline use the expert-validated MPCR-RAG map path
instead of the obsolete CSV, and (B) allow local models alongside OpenRouter.

Context: W05 of the semester plan. Both items are prerequisites for the factorial run
in W09–W10. Task A is BLOCKING — every T3 result produced before it is invalid.

---

## Task A — MPCR-RAG map integration

### Why this is blocking

`main.py:94` reads `outputs/picked_species_enhanced_clean.csv` (**133 species**, marked
obsolete) while `mpcr_rag/data/fichas.sqlite` holds **5,791 species** built by the
validated segmenter. The full pipeline is therefore rendering maps from a stale catalog
and calling them the expert-validated maps. Two consequences:

1. Every T3 run so far used a map built from the wrong source.
2. The T3 prompt describes a renderer that no longer exists (`Hammel 2014`,
   `Color apagado/muted` — the current renderer draws matched regions outline-only and
   the DEM cyan IS the fill).

Both must be fixed together: the prompt is a description of the map.

### A1 — Source the ficha from the MPCR-RAG catalog

Replace the CSV block in `main.py` (roughly lines 92–175) with a catalog lookup:

```python
from mpcr_rag.store import local_store
from mpcr_rag import config as rag_config

conn = local_store.connect(rag_config.SQLITE_PATH)
f = local_store.get(conn, especie_nombre.replace(" ", "_"))
```

- [ ] **A1.1** Look up the species in `fichas.sqlite`
- [ ] **A1.2** Derive `alt_manual = (f.elev_min, f.elev_max)` — still needed by the T3 prompt
- [ ] **A1.3** Keep a fallback when the species is absent (should now be rare) — the
      existing Mesoamerica-map fallback stays, but log it loudly so misses are visible
- [ ] **A1.4** Delete the `texto_manual` construction — it fed the removed T2 tier and is
      dead code that risks re-introducing the leakage bug

### A2 — Render through the validated path

Call the same function the MPCR-RAG system uses, so the map is byte-identical:

```python
from mpcr_rag.query.gbif_map import single_species_map
map_path, n_pts = single_species_map(f, out_path=os.path.join(out_dir, "mapa_habitat_manual.png"))
```

**Decision required — which GBIF points go on the map?**

| Source | Path | Properties |
|---|---|---|
| `GBIFExtractor` (current) | Mesoamerica fetch, cleaned to CR | scientificName search, no taxon resolution |
| `gbif_map.get_points` (MPCR-RAG) | accepted taxonKey, EXACT match only, uncertainty <10 km, lat/lon swap fix, DEM-tagged, cached | more rigorous; what the expert validation used |

Recommendation: **use `gbif_map.get_points` for the map** (fidelity with the validated
maps), and **keep `GBIFExtractor` for RF training** — the model needs Mesoamerica-wide
presences, which the MPCR-RAG path does not provide. These are different needs; do not
collapse them.

- [ ] **A2.1** Render via `single_species_map`
- [ ] **A2.2** Leave `presencias_meso` / RF training untouched
- [ ] **A2.3** Confirm the RF path still receives Mesoamerica points

### A3 — Rewrite the T3 prompt to match the real renderer

`llm/prompt_templates.py`, FUENTE 2 block. Replace the stale legend with what
`renderer.py` actually draws:

| Element | Current render |
|---|---|
| Cyan fill | DEM elevation mask inside matched regions — the habitat band |
| Hatched cyan | outlier elevation band |
| White outline | botanical region named in the Manual (no fill) |
| Dashed amber outline | GBIF-inferred region (>=5 occurrences, not in Manual text) |
| Orange outline | named park, elevation-filtered |
| Amber thin outline | protected areas in the elevation range |
| Red points | GBIF occurrences |
| Dark gray | outside the Manual's geographic range |

- [ ] **A3.1** Rewrite FUENTE 2 to match the table above
- [ ] **A3.2** Fix the attribution: `Regiones Botánicas (José Araya, 2026)`, not Hammel 2014
- [ ] **A3.3** Rephrase the answer instruction toward POSITIONAL language (north/south,
      Pacific/Caribbean, inside/outside the cyan) rather than
      `"Menciona zonas geográficas concretas"`, which invites the model to OCR the legend
      instead of reading geometry (observed with qwen3-vl on 2026-08-23)

### A4 — Verify

- [ ] **A4.1** `Peltogyne purpurea` — map matches the MPCR-RAG output; ~110 GBIF points
- [ ] **A4.2** `Talamancaster minusculus` — zero-width elevation caveat still surfaces
- [ ] **A4.3** A species in the 5,791 catalog but NOT in the old 133-row CSV — proves the
      expansion works
- [ ] **A4.4** Diff a freshly rendered map against the MPCR-RAG one for the same species;
      they should be identical

### A5 — Consequences to record

- [ ] **A5.1** Note in the experiment log that pre-integration T3 runs are void
- [ ] **A5.2** **Test-set decision now open**: species selection was constrained to the
      133-row CSV. With 5,791 available, selection becomes a real design choice and is a
      W08 freeze item. Raise with the tutor.

---

## Task B — Multi-provider LLM (add local models)

### Current state

`main.py:383` hardcodes `["openai/gpt-4o", "anthropic/claude-sonnet-4-5"]`, and
`OpenRouterClient.url` is fixed to OpenRouter. There are 12 hardcoded `openrouter.ai`
endpoints repo-wide. The model is env-configurable (`MPCR_ENRICH_MODEL`); the base URL
is not.

### B1 — Provider resolution

Add `llm/providers.py` mapping a model spec to a transport:

```
"openrouter:openai/gpt-4o"        -> https://openrouter.ai/api/v1, OPENROUTER_API_KEY
"ollama:qwen3-vl:8b-instruct"     -> http://localhost:11434/v1,     no key
```

Both speak the OpenAI chat-completions shape, so the existing `content_parts` payload
with base64 `image_url` works unchanged for both. This is the whole reason the swap is cheap.

- [ ] **B1.1** `resolve_provider(spec) -> (base_url, api_key, model_name)`
- [ ] **B1.2** Default provider from env (`LLM_PROVIDER`), overridable per call

### B2 — Generalize the client

- [ ] **B2.1** `OpenRouterClient.__init__` takes `base_url`; keep the class name for now
      to avoid touching call sites (rename in the W14–16 restructure)
- [ ] **B2.2** Skip the `Authorization` header when the provider needs no key
- [ ] **B2.3** Model names with colons already survive
      `.replace(':','_')` in the output filename — verify with `qwen3-vl:8b-instruct`

### B3 — Ollama specifics

- [ ] **B3.1** Use the `/v1` OpenAI-compatible endpoint so the payload stays identical
- [ ] **B3.2** **Context length**: `/v1` silently IGNORES `options.num_ctx`. Set
      `OLLAMA_CONTEXT_LENGTH=16384` in the environment and restart the Ollama service, or
      the long T3 prompt plus two images truncates invisibly. Verify with `ollama ps`
      showing `CONTEXT 16384`
- [ ] **B3.3** Set `OLLAMA_KEEP_ALIVE=30m` so sweeps do not reload the model each call
- [ ] **B3.4** Expect ~18 s warm per call on the 4070 Ti; first call ~47 s (model load)

### B4 — Config-driven model list

- [ ] **B4.1** Move the model list out of `main.py` into `config.py`, e.g.
      `MULTIMODAL_MODELS = [...]`, env-overridable
- [ ] **B4.2** `experiment/run_experiment.py:229` writes `meta["models"]` — read it from
      the same config so run metadata cannot drift from what actually ran
- [ ] **B4.3** Record provider AND model in the profile metadata header

### B5 — Verify

- [ ] **B5.1** Run one T3 profile through `ollama:qwen3-vl:8b-instruct` end to end
- [ ] **B5.2** Same species through `openrouter:openai/gpt-4o` — compare outputs
- [ ] **B5.3** Confirm both write distinct `llm_profile_BIMODAL_*.txt` files
- [ ] **B5.4** Confirm the run metadata records which provider produced which file

---

## Sequencing

A before B. Task A is blocking and changes what the models see; running B first would
just validate the provider plumbing against maps that are about to change.

Suggested order: A1 -> A2 -> A4 (verify) -> A3 (prompt) -> A4 again -> commit;
then B1 -> B2 -> B3 -> B5 -> commit.

---

## Out of scope this session

- The remaining 11 hardcoded OpenRouter endpoints (only `llm/openrouter_client.py`
  matters for the experiment; the rest are utility scripts)
- `legend=False` renderer flag for the OCR-vs-geometry ablation (W05–W06, separate)
- Test-set selection from the 5,791 catalog (W08 freeze, needs tutor input)
- REST API / bachelor handoff (W06–W07)
