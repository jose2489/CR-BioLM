# CR-BioLM Architecture

## Layers

```
Manual PDFs / GBIF / DEM / WorldClim
          |
     [ Core Services ]
          |
   ┌──────┴──────────┬──────────┬───────┐
  REST              MCP        CLI     pkg
   └──────────────────────────────────────┘
          |
     [ Consumers ]
   bachelor UI team / Claude Code / scripts
```

Core services are **siblings to** every adapter — not stacked beneath one.
Swapping REST for MCP or adding a new consumer does not require touching core.

---

## Five Core Products

| # | Product | Module | Description |
|---|---|---|---|
| 1 | **Ficha Extractor** | `mpcr_rag/ingest/` | Segments Manual PDFs → per-species RawFicha; extracts structured fields (geo_parser + regex); LLM enrichment pass (cached). Outputs `DistributionFicha` JSON. |
| 2 | **Map Renderer** | `utils/distribution_map/renderer.py` | Takes a `DistributionFicha` and optional GBIF GeoDataFrame → publication-quality PNG. Layers: CR base, matched regions (outline), GBIF-inferred regions (dashed amber), elevation mask (DEM cyan fill), parks/cantons (outline), GBIF presence points. |
| 3 | **Occurrence Service** | `mpcr_rag/query/gbif_map.py` | Fetches and cleans GBIF points for a species (cached per species). Provides `get_points`, `count_only` (frozen DOI snapshot), `single_species_map`, `most_likely_map`. |
| 4 | **Query Service** | `mpcr_rag/query/` | `intent.py`: parses NL question → structured filter + intent type. `retriever.py`: `filter_all` (exhaustive structured) + `pattern_b` (semantic top-k via Pinecone). `answer.py`: routes to A / B / B→A recipe; composes grounded text via LLM. |
| 5 | **Answer Composer** | `mpcr_rag/query/answer.py::_compose` | LLM-composes a grounded answer from retrieved fichas only; no external facts. Isolated so the composition strategy is swappable (different model, prompt, or local model). |

---

## The `DistributionFicha` Contract

This is the boundary object between the Ficha Extractor (producer) and the Map
Renderer / Query Service (consumers). Everything downstream speaks this schema.

```json
{
  "species":              "Peltogyne purpurea",
  "habitat_raw":          "Bosque muy húmedo y pluvial, 0–800(–1200) m",
  "vertientes":           ["Caribe", "Pacífico"],
  "regions":              [{"canonical_name": "Llanura Atlántica", "hierarchy_level": 2, "source_span": "..."}],
  "parks":                [{"canonical_name": "Parque Nacional Corcovado", ...}],
  "cantons":              [],
  "districts":            [],
  "elevation":            {"min_m": 0, "max_m": 800, "outlier_lo": null, "outlier_hi": 1200},
  "forest_types":         ["muy húmedo", "pluvial"],
  "locality_occurrences": [{"feature_type": "cordillera", "feature_name": "Fila Costeña",
                            "qualifier": "S", "vertiente": "Pacífico",
                            "embedded_protected_areas": ["R.N.A. Cabo Blanco"]}],
  "confidence":           {"overall": 1.0, "geographic": 0.8, "elevation": 1.0},
  "unresolved_tokens":    []
}
```

Producers write this; consumers read it. No consumer should re-parse the Manual
text — all geospatial interpretation happens in the Ficha Extractor layer.

---

## Query Routing (answer.py)

Three fixed recipes, not an open-ended agent:

```
Question
  │
  ├─ named species detected?  ──yes──► A  (lookup)
  │                                        get Ficha from SQLite
  │                                        single_species_map (Manual-grounded)
  │
  └─ no ──► parse_intent()
              │
              ├─ intent_type = "superlative"  ──► B→A
              │     filter_all() (exhaustive, no top_k)
              │     _select_superlative() (deterministic: elev_max/min, n_regions, gbif_count)
              │     single_species_map on winner
              │
              └─ intent_type = "list"  ──► B  (geospatial)
                    pattern_b() (Pinecone semantic + structured filters, top_k=12)
                    most_likely_map (GBIF evidence scatter, multi-species)
```

`gbif_count` is the only proxy for "más común/abundante" — the Manual records no
abundance field. It is always reported as a proxy in the answer text.

---

## Adapter Layer

All three adapters call the same core functions. None of them contain business logic.

| Adapter | Entry point | Status |
|---|---|---|
| CLI | `python -m mpcr_rag.query.answer "..."` | done |
| MCP | `mpcr_rag/mcp/server.py` (FastMCP) | in progress |
| REST | FastAPI, `app/` | planned (W06–W07, bachelor team) |

### MCP tool tiers (cost ladder)

| Tier | Tools | Requirements |
|---|---|---|
| L0 — free, SQLite only | `get_vocabulary`, `search_species`, `get_species` | none |
| L1 — seconds, cached | `parse_distribution_text`, `get_occurrences`, `render_species_map` | none |
| L2 — LLM + vector DB | `semantic_search`, `answer_question` | `PINECONE_API_KEY`, `OPENROUTER_API_KEY` |

Every MCP tool returns a provenance envelope:
`{value, source, citation, confidence, caveat}` — `source` in `{MPCR, GBIF, DEM, SINAC}`,
`confidence` in `{exact, estimated, insufficient}`.

---

## Known Bugs / Pre-run Checklist

1. **T3 prompt mismatch** (`llm/prompt_templates.py`): describes "muted color =
   region outside elevation range" but the renderer now draws matched regions
   outline-only (DEM cyan IS the fill). Also credits "Hammel 2014" instead of
   "Regiones Botánicas (José Araya, 2026)" and never mentions GBIF-inferred dashed
   regions or hatched outlier bands. **Fix before re-running any T3 experiment.**

2. **`elev_min == elev_max` warning**: Manual "ca. 2950 m" produces a zero-width
   elevation range → empty DEM mask → blank map. The MCP `render_species_map` tool
   must detect and caveat this case (e.g. *Talamancaster minusculus*).

3. **12 hardcoded OpenRouter endpoints**: model is env-configurable
   (`MPCR_ENRICH_MODEL`) but the base URL is not. Centralize the client to swap
   OpenRouter ↔ Ollama ↔ vLLM by env var before local-model A/B testing.
