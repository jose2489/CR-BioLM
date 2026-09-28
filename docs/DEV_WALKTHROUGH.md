# MPCR-RAG — developer walkthrough

For the developers building the backend/API that the bachelor team will consume.
Read this first, then `docs/SHIP_PLAN.md` (what to build, in order) and
`docs/BACHELOR_QUICKSTART.md` (what the students will see).

You own the **service layer**. The domain logic underneath already works and is
validated against experts — wrap it, do not reimplement it.

---

## 1. What this system is, in one page

`mpcr_rag/` turns the *Manual de Plantas de Costa Rica* (a printed flora, Tomos II–VI)
into a queryable catalog, and answers questions about Costa Rican plants with
**evidence that can be checked**: every fact carries where it came from.

Three kinds of things live here:

| | What it holds | Where it comes from |
|---|---|---|
| **Catalog** | 6,946 species: elevation, regions, slopes, forest types, habits, phenology, morphology, discussion | OCR'd Manual PDFs, parsed |
| **Occurrence evidence** | 999,625 cleaned Costa Rican records, 143 protected areas, per-species aggregates | frozen GBIF snapshot (DOI 10.15468/dl.8yhee8) |
| **Names** | 514 common-name links over 183 species | the Manual, iNaturalist (CR), GBIF, Tropicos |

Two rules explain most design decisions you will meet:

1. **Provenance over convenience.** An answer that cannot be traced to a source is a
   bug, not a feature. Everything returns `{value, source, citation, confidence, caveat}`.
2. **Never assert absence.** GBIF records reflect collection effort. "No records in
   that park" never becomes "it does not grow there".

---

## 2. Data: what exists, and how to get it

| Store | What | Size | Status |
|---|---|---|---|
| `mpcr_rag/data/fichas.sqlite` | the catalog — **source of truth** | 59 MB | ships in the data bundle |
| Postgres schema `mpcr` | derived: embeddings, occurrences, places, taxa, vernacular | ~445 MB | rebuilt locally, never shipped |
| `data_raw/` | botanical regions, DEM, SINAC protected areas, IGN boundaries | ~50 MB | ships in the bundle, **not in git** |
| `mpcr_rag/data/gbif_cache/` | per-species occurrence points already downloaded | 528 species | ships (optional) |
| `mpcr_rag/data/maps/` | rendered PNG maps | — | ships (optional) |

**How to get the data:** `python utils/make_handoff_bundle.py` produces a zip with
everything gitignored that the code needs. Unzip at the repo root.

**Postgres** is embedded (the `pgserver` package: a real PostgreSQL 16 with pgvector,
pip-installed, no service to configure):

```
python -m mpcr_rag.store.pg_store start     # 127.0.0.1:5433, user postgres, no password
python -m mpcr_rag.store.pg_store status    # row counts, sync date
python -m mpcr_rag.store.pg_store sync      # rebuild embeddings from SQLite (GPU: ~3 min)
```

Everything except semantic search and answering works **without** Postgres, straight
from SQLite. Keep it that way: it is what makes the student handoff portable.

---

## 3. Setup (about 20 minutes)

```
git clone <repo> && cd CR-BioLM
git checkout mpcr-rag
pip install -r mpcr_rag/requirements.txt
#  unzip the data bundle at the repo root  ->  data_raw/ and mpcr_rag/data/
python -m mpcr_rag.store.pg_store start
python -c "from mpcr_rag import config; from mpcr_rag.store import local_store as s; print(len(s.filter_fichas(s.connect(config.SQLITE_PATH))), 'species')"
```

Keys (`.env` at the repo root, never committed):

| Variable | Needed for |
|---|---|
| `OPENROUTER_API_KEY` | natural-language answering and intent parsing only |
| `TROPICOS_API_KEY` | expanding the name table (optional) |
| `MPCR_VECTOR_BACKEND` | `pgvector` (default) or `pinecone` (paper reproduction) |
| `MPCR_PG_URL` / `MPCR_PG_PORT` | point at another Postgres instead of the embedded one |

---

## 4. Live demo

Easiest for a session, prints each step with its cost level:

```
python -m mpcr_rag.scripts.demo_walkthrough              # steps 1-7, free
python -m mpcr_rag.scripts.demo_walkthrough --with-llm   # adds the paid answer
python -m mpcr_rag.scripts.demo_walkthrough --species "Dalbergia retusa"
```

Run it once before the meeting: step 7 takes ~18 s the first time while the embedding
model loads, and a few seconds afterwards.

The same thing typed by hand, if you prefer a REPL:

```python
from mpcr_rag import config
from mpcr_rag.store import local_store
conn = local_store.connect(config.SQLITE_PATH)

# 1. one species, everything the Manual says
f = local_store.get(conn, "Peltogyne_purpurea")
f.species, f.elev_min, f.elev_max, f.regions, f.flowering_months, f.morphology[:80]

# 2. structured search, exhaustive (no ranking, no cap)
from mpcr_rag.query.retriever import filter_all
len(filter_all(conn, habit="árbol", endemic=True, elev_lo=2000))

# 3. occurrence points for a species (cached, cleaned)
from mpcr_rag.query import gbif_map
pts = gbif_map.get_points("Peltogyne purpurea");  len(pts)

# 4. the map (PNG on disk)
path, n = gbif_map.single_species_map(f);  path, n

# 5. where can I see it — parks with records, Manual vs record regions
from mpcr_rag.evidence.places import where_to_see
where_to_see("Peltogyne purpurea")["confirmed_places"][:3]

# 6. common name -> candidate species, with a decision
from mpcr_rag.evidence import vernacular
vernacular.resolve_decision("cocobolo")["decision"]        # 'preferred' -> Dalbergia retusa
vernacular.resolve_decision("cortez amarillo")["decision"] # 'ambiguous', 3 Handroanthus

# 7. semantic search (needs Postgres)
from mpcr_rag.query.retriever import pattern_b
[(x.species, round(s,3)) for x, s in pattern_b("bosque nuboso en Talamanca", top_k=5)][:3]

# 8. full natural-language answer (needs OPENROUTER_API_KEY)
from mpcr_rag.query.answer import answer
a = answer("¿Dónde puedo observar Peltogyne purpurea?");  a["mode"], a["text"][:200]
```

Then the same surface as MCP tools: `python -m mpcr_rag.mcp.server` (11 tools).

---

## 5. The functions you will wrap

Cost levels: **L0** free and instant (SQLite only) · **L1** seconds, cached, local
embeddings or disk I/O · **L2** needs a paid key.

| Service | Call | Cost | Returns |
|---|---|---|---|
| Species lookup | `local_store.get(conn, "Genus_species")` | L0 | `Ficha` dataclass |
| Structured search | `retriever.filter_all(conn, **filters)` | L0 | `list[Ficha]`, exhaustive |
| Controlled vocabulary | `intent.load_vocab(conn)` | L0 | valid filter values — **expose this first** |
| Common name | `vernacular.resolve_decision(name, region=, elev=)` | L0 | decision + ranked candidates |
| Where to see it | `places.where_to_see(species)` | L0 | parks with records, regions, elevation |
| Occurrences | `gbif_map.get_points(species)` | L1 | GeoDataFrame, cleaned, DEM elevation |
| Occurrence count | `gbif_map.count_only(species)` | L0 | count from the frozen snapshot |
| Map (one species) | `gbif_map.single_species_map(ficha)` | L1 | `(png_path, n_points)` |
| Map (a query) | `gbif_map.most_likely_map(results, query_text=...)` | L1 | `(png_path, n_points)` |
| Parse a distribution text | `parser.build_ficha(habitat_raw=, geographic_notes=)` | L0 | `DistributionFicha` |
| Semantic search | `retriever.pattern_b(text, **filters)` | L1 | `[(Ficha, score)]` |
| Description search | `pg_store.search(text, section="description")` | L1 | hits |
| Intent parsing | `intent.parse_intent(question)` | L2 | structured filters + `common_name` |
| Full answer | `answer.answer(question)` | L2 | text + map + mode + resolution |

**The contract object is `Ficha`** (`mpcr_rag/schema.py`): dataclass,
`to_json()` / `from_json()`, `vector_id` = species name with underscores. Your API
serializes this; do not invent a second shape.

---

## 6. What to expose, and in what order

Follow `SHIP_PLAN.md` Phase 2, which the students are waiting on:

1. `GET /v1/vocabulary` — valid filter values. Everything else depends on it.
2. `GET /v1/species/{name}` — one ficha.
3. `GET /v1/species?habit=&elev_lo=&region=…` — structured search.
4. `GET /v1/species/{name}/map` — the PNG, **served as a URL**, never a filesystem path.
5. `GET /v1/species/{name}/occurrences` — points, counts, elevation stats.
6. `GET /v1/species/{name}/where-to-see` — parks and regions.
7. `GET /v1/names/resolve?q=poró` — the decision plus candidates.

L2 (answering, semantic search) stays **out** of the student-facing API: it costs money
per call and is not what they consume.

**Every response uses the envelope:**

```json
{"value": …, "source": "MPCR", "citation": "Manual …, Tomo V, p. 669",
 "confidence": "exact", "caveat": ""}
```

`confidence` ∈ exact | estimated | insufficient. A non-empty `caveat` means the client
should show it, not hide it.

---

## 7. Rules that must not be broken

These are not style preferences; breaking them invalidates the science.

1. **SQLite never leaks past `local_store`.** No raw SQL in the API layer. That rule is
   why swapping to Postgres or handing the service to another institution is a
   one-module change.
2. **Never turn missing evidence into absence.** No records in a park means unknown.
3. **Occurrence counts are collection effort, not abundance.** Say so wherever you rank
   by them.
4. **Common names are never resolved silently.** `ambiguous` must reach the UI as a
   choice, not a guess. See §8.
5. **Genus-level text is genus-level.** `Ficha.genus_description` describes the genus;
   never render it as a statement about the species.
6. **The catalog is Tomos II–VI.** *Cecropia* ("guarumo") is genuinely absent, not a bug.
   Say "not in the catalog", never "does not exist".

Engineering traps (from `SHIP_PLAN.md`, still open):

- `sqlite3.connect()` has no `check_same_thread=False` and the connection is a module
  global → a threaded server will raise. Use a per-request connection or a pool.
- **matplotlib is not thread-safe** → serialize map rendering behind a lock, or
  precompute the map corpus (Phase 3) and serve static files.
- Map tools currently return **server-side paths**; the API must map them to URLs.
- `MANUAL_ROOT` and `DATA_DIR` are absolute paths → env-drive them before containerizing.
- Root `requirements.txt` is a UTF-16 dev freeze; use `mpcr_rag/requirements.txt`.

---

## 8. Common names — the piece that needs UI thinking

**How it works:** a name is looked up in `mpcr.vernacular` (accent-sensitive first,
unaccented as fallback), candidates are filtered by anything else the question gives
(region, elevation), then ranked by **source trust** and then by how well recorded the
species is in Costa Rica.

Source trust, highest first: `MPCR` (the Manual, citable by Tomo and page) ·
`INAT_CR` (the name iNaturalist marks as preferred *for Costa Rica*) · `INAT_ES` ·
`GBIF` / `TROPICOS` (global indexes, no country field).

**`resolve_decision()` returns one of four decisions, and the UI must handle each:**

| Decision | Meaning | UI |
|---|---|---|
| `unique` | one candidate | answer, but state the interpretation ("por *cocobolo* entiendo *Dalbergia retusa*") |
| `preferred` | one Costa Rica-specific source names it, and it is well recorded | answer, say why it was chosen |
| `ambiguous` | several plausible | **show the candidates and ask** — never pick |
| `none` | not in the table | say the name is unknown here; do not guess |

Real examples: "cocobolo" → preferred (*Dalbergia retusa*, over three rivals a global
index adds). "nazareno" → preferred (*Peltogyne purpurea*, over a *Tibouchina* used by
that name elsewhere in Latin America). "cortez amarillo" → ambiguous over three
*Handroanthus*, each with different parks. "ceiba" → ambiguous. "guarumo" → none
(*Cecropia* is in Tomos VII–VIII, outside this catalog).

Note that adding Tropicos moved "cocobolo" from unique to preferred: more sources means
more candidates, and the trust ranking — not a longer list — is what keeps the answer
right. Expect that to keep happening as sources are added.

**Current limits — tell the students, and plan for them:**
- The table covers **183 species**, not the whole catalog. A full fetch
  (iNaturalist at ~1 request/second ≈ 2 h, then GBIF, then Tropicos expansion) is
  pending and costs nothing but wall-clock.
- The data lives **only in Postgres**. `Ficha.common_names` in SQLite is still empty, so
  the portable bundle has no names yet.
- The Manual extractor is a **prototype** (`vernacular.from_mpcr`): ~36% of species yield
  a name and author surnames still leak. The bachelor team's extractor, validated on a
  gold set (target precision ≥ 0.95), replaces that one function — see
  `docs/EXTENSIONS.md` for the contract and the known traps.

---

## 9. First tasks, in order

1. Read `docs/ARCHITECTURE.md` (layering) and `docs/SHIP_PLAN.md` Phase 1–2.
2. Fix the three blockers that stop a threaded server: SQLite threading, matplotlib lock,
   env-driven paths.
3. Ship `/v1/vocabulary`, `/v1/species/{name}`, `/v1/species` and export
   `docs/openapi.json` — then hand that spec to the students, before anything else exists.
4. Add maps (precomputed corpus + URL serving), occurrences, where-to-see.
5. Add `/v1/names/resolve` and agree the ambiguity UX with the students.

**Acceptance for step 3:** a mock server generated from `docs/openapi.json` answers every
endpoint with realistic fixtures, and the students can start building against it.

---

## 10. Reading order

| File | Why |
|---|---|
| `docs/ARCHITECTURE.md` | the five products and the layering |
| `mpcr_rag/schema.py` | `Ficha`, the contract object |
| `mpcr_rag/store/local_store.py` | the only module that touches SQLite |
| `mpcr_rag/query/retriever.py` | structured + semantic retrieval |
| `mpcr_rag/query/gbif_map.py` | occurrences and maps |
| `mpcr_rag/evidence/` | occurrences, places, taxa, vernacular |
| `mpcr_rag/mcp/server.py` | the same surface already wrapped once, with docstrings stating cost and caveats — **your best reference for API shapes** |
| `docs/SHIP_PLAN.md` | the build order and the open blockers |
| `docs/EXTENSIONS.md` | how to add a new evidence provider |
