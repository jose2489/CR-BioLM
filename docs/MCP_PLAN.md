# MCP orientation + GitHub structure — execution plan

Session goal: expose `mpcr_rag` as an MCP server (map creation + RAG queries callable
directly), and get the project onto GitHub with a structure that makes clear how the
parts relate.

Principle for this session: **additive only.** No mass file moves before the W08 test-set
freeze. Document the target structure; migrate after the paper.

---

## Phase 0 — Safety first (10 min)

The `mpcr-rag` branch exists only locally. Push it before touching anything else.

- [ ] **0.1** Confirm nothing sensitive is staged:
      `git status --short` and `git ls-files | grep -iE "\.env|key|secret"`
- [ ] **0.2** Commit the current working-tree changes (10 modified, ~12 untracked)
      in logical groups — do not one-shot them into a single commit.
- [ ] **0.3** `git push -u origin mpcr-rag`

Acceptance: `git ls-remote --heads origin` lists `mpcr-rag`.

---

## Phase 1 — Repo structure (45 min, docs only)

### Target structure (document now, migrate in W14–16)

```
CR-BioLM/
├── README.md              project overview; how the parts relate
├── docs/
│   ├── ARCHITECTURE.md    layering, products, contracts
│   └── MCP_PLAN.md        this file
│
├── mpcr_rag/              SUBSYSTEM A — flora catalog + RAG        [BIP paper]
│   ├── ingest/ store/ query/ eval/
│   └── mcp/               <- added this session
│
├── utils/distribution_map/  SHARED CORE — ficha parser + map renderer
│                            (consumed by BOTH subsystems; future: core/)
│
├── main.py, models/, xai/, data/, experiment/, evaluator/
│                          SUBSYSTEM B — SDM + multimodal + experiments  [journal paper]
│
└── app/                   frontend / reporting
```

**Dependency rule (enforce from now on):** `mpcr_rag/` may import from
`utils/distribution_map` and root `config` only. Nothing in `mpcr_rag/` may import from
`models/`, `xai/`, or `experiment/`. This keeps the subsystem liftable to its own repo
later without untangling.

- [ ] **1.1** Write root `README.md`: what CR-BioLM is, the two subsystems, the two
      papers, what is shared, quickstart, data-not-in-repo note.
- [ ] **1.2** Write `docs/ARCHITECTURE.md`: the five core products (Ficha Extractor,
      Map Renderer, Occurrence Service, Query Service, Answer Composer), the
      `DistributionFicha` JSON contract, and the adapter layer (REST / MCP / CLI as
      siblings over the same core).
- [ ] **1.3** Document how to obtain the gitignored data (`data_raw/`, DEM, shapefiles)
      — a `make sync-data` target or a documented bundle. Needed for the bachelor team.

Acceptance: a newcomer can read README + ARCHITECTURE and know where things live and
which parts they may depend on.

---

## Phase 2 — MCP server (main work, 2–3 h)

### 2.0 Install
- [ ] `pip install "mcp[cli]"` and add to `requirements.txt`

### 2.1 Create `mpcr_rag/mcp/server.py`

Use `FastMCP`. Tools grouped by cost; **cost and preconditions go in each docstring** —
the description is what an agent reasons over.

**Level 0 — free, instant, SQLite only, no API keys**

| Tool | Backing | Returns |
|---|---|---|
| `get_vocabulary()` | `intent.load_vocab` | habits, forest types, regions, vertientes, families |
| `search_species(...)` | `retriever.filter_all` | matching species + count (exhaustive, no top-k) |
| `get_species(name)` | `local_store.get` | full ficha + Tomo/página |

**Level 1 — seconds, cached, no LLM**

| Tool | Backing | Returns |
|---|---|---|
| `parse_distribution_text(habitat_raw, geographic_notes, species)` | `parser.build_ficha` | DistributionFicha as JSON |
| `get_occurrences(species, elev_lo?, elev_hi?, vertiente?, region?)` | `gbif_map.get_points` + `filter_points` | n, bbox, elevation stats, region breakdown |
| `render_species_map(species)` | `gbif_map.single_species_map` | PNG absolute path + layer summary + n points |

**Level 2 — needs PINECONE_API_KEY / OPENROUTER_API_KEY**

| Tool | Backing | Returns |
|---|---|---|
| `semantic_search(query_text, ...filters)` | `retriever.pattern_b` | ranked species + relevance |
| `answer_question(question)` | `answer.answer` | grounded text + citations + map path |

**Resources**
- `mpcr://vocabulary` — the controlled vocabulary (the schema-guidance artifact)
- `mpcr://species/{vector_id}` — one ficha as addressable context

### 2.2 Provenance envelope

Every tool returns `{value, source, citation, confidence, caveat}` rather than a bare
value. `source` in `{MPCR, GBIF, DEM, SINAC}`; `confidence` in
`{exact, estimated, insufficient}`. This is what preserves the verifiability claim once
an agent is composing calls.

Known caveat to emit: when `elev_min == elev_max` (e.g. Manual "ca. 2950 m"), the DEM mask
is empty — flag it rather than returning a blank map. See *Talamancaster minusculus*.

### 2.3 Wire it up
- [ ] `mpcr_rag/mcp/__init__.py`
- [ ] `.mcp.json` at repo root registering the server for Claude Code
- [ ] Graceful degradation: L2 tools report "API key not configured" instead of raising

### 2.4 Test
- [ ] `mcp dev mpcr_rag/mcp/server.py` (MCP Inspector), or restart Claude Code and call
      the tools directly
- [ ] Smoke: `get_species("Talamancaster minusculus")` → ficha with Tomo IV p.320
- [ ] Smoke: `render_species_map("Peltogyne purpurea")` → PNG path, 110 GBIF points
- [ ] Smoke: `search_species(endemic=True, elev_lo=2000)` → non-empty

Acceptance: map creation and catalog queries are callable as MCP tools without touching
Python directly.

---

## Phase 3 — Commit and push (20 min)

- [ ] **3.1** Commit in logical units (`feat(mcp): ...`, `docs: ...`)
- [ ] **3.2** Push `mpcr-rag`
- [ ] **3.3** Decide: keep working on `mpcr-rag`, or open a PR into `main`?

---

## Explicitly NOT in this session

- Moving `utils/distribution_map` to `core/` — deferred to W14–16 (breaks many imports)
- The REST API for the bachelor team — Phase 2 of the semester plan (W06–W07)
- The agent loop — Paper 3
- Centralizing the 12 hardcoded OpenRouter endpoints — separate small task, needed
  before local-model A/B testing
- `legend=False` renderer flag — W05–W06, before the metric work

---

## Open questions

1. Is `jose2489/CR-BioLM` public or private? Determines the bachelor handoff mechanism.
2. Does `mpcr-rag` eventually merge into `main`, or stay a long-lived branch?
3. `data_raw/regiones_botanicas/*.shp` is gitignored and irreplaceable — git-lfs, or a
   released data bundle? Needed before anyone else can run the map pipeline.
