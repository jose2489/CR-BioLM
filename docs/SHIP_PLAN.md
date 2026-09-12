# Shipping the MPCR Map/Catalog API to the bachelor team

Goal: the bachelor team can consume the map + catalog services without touching
Python internals, without our API keys, and without depending on our uptime.

Delivery order: **contract first, container second, hosted instance third.**
They should be unblocked by the contract within days; the container follows; the
shared instance is a convenience, never a dependency.

Scope: **L0 + L1 only** (SQLite catalog, GBIF occurrences, map rendering).
L2 (`semantic_search`, `answer_question`) is out of scope for the handoff — it needs
paid keys and is not what they consume.

Two design constraints run through every phase, because they are what keeps this
alive past the semester:

- **Portability** — it must be able to move to a RedBioMA / GBIF node later.
- **Extensibility** — adding volumes or species must be routine, not a rebuild.

---

## Blockers to clear before anything ships

Real defects for a networked or containerized deployment, found in the current code.
Ordered by how much they hurt.

| # | Blocker | Where | Fix |
|---|---|---|---|
| 1 | Map tools return a **server-side filesystem path** — meaningless to a remote client | [mcp/server.py:349](../mpcr_rag/mcp/server.py#L349) `map_path` | Return a URL served by StaticFiles; keep the path internal |
| 2 | `sqlite3.connect()` has no `check_same_thread=False`, and `_conn` is a **process global** | [local_store.py:37](../mpcr_rag/store/local_store.py#L37), [mcp/server.py:47](../mpcr_rag/mcp/server.py#L47) | Per-request connection or a pool. Under a threaded ASGI server the current code raises `ProgrammingError` |
| 3 | **matplotlib is not thread-safe**; concurrent renders corrupt figures | [renderer.py](../utils/distribution_map/renderer.py) | Serialize renders behind a lock, or single worker + queue. Mostly moot once maps are precomputed |
| 4 | `MANUAL_ROOT` is a **hardcoded Windows path** | [mpcr_rag/config.py:22](../mpcr_rag/config.py#L22) | `os.environ.get("MPCR_MANUAL_ROOT", ...)`. Fails soft today (empty `CORPUS`) but must not be a surprise |
| 5 | Root `DATA_DIR` is pinned to `<repo>/data_raw` | [config.py:13](../config.py#L13) | `os.environ.get("CRBIOLM_DATA_DIR", ...)` so the container can mount data elsewhere |
| 6 | Root `requirements.txt` is **UTF-16 encoded** (PowerShell `pip freeze >`) and is a full dev freeze incl. jupyter/debugpy/groq | `requirements.txt` | Do not use it in the image. Write a minimal `requirements_api.txt` |
| 7 | Directories created **at import time** (`gbif_cache`, `maps`, `gbif_snapshot`) | [gbif_map.py:37-47](../mpcr_rag/query/gbif_map.py#L37) | Fine, but they must be writable volumes in the container |
| 8 | **No auth, no rate limit** | — | Bearer token on render endpoints; read endpoints open |
| 9 | Geodata is **gitignored** (`*.shp`, `*.tif`, `data_raw/`) | [.gitignore](../.gitignore) | Out-of-band bundle — see Phase 5 |

---

## Phase 1 — Portability (half day)

- [ ] **1.1** Env-drive the paths: `CRBIOLM_DATA_DIR`, `MPCR_DATA_DIR`, `MPCR_MANUAL_ROOT`.
      Defaults stay exactly as today so nothing breaks on your machine.
- [ ] **1.2** `local_store.connect(path, *, check_same_thread=True)` — pass `False`
      from the API layer only, so the CLI keeps its safety.
- [ ] **1.3** Write `requirements_api.txt` (UTF-8, no BOM). Minimal set:
      `fastapi uvicorn[standard] pydantic python-dotenv geopandas rasterio shapely
      matplotlib pandas numpy pygbif`. Explicitly NOT: torch, shap, jupyter, pinecone, groq.
- [ ] **1.4** Add a `PUBLIC_BASE_URL` setting (default `""` = relative URLs). Every map
      URL is built from it. **Needed for external hosting**: a node that mounts the app
      at `/mpcr/` instead of `/` breaks any hardcoded `/maps/...` path.
- [ ] **1.5** Verify a clean `pip install -r requirements_api.txt` on Linux/aarch64.
      GDAL/GEOS ship inside the rasterio and fiona wheels — no apt packages needed.

Acceptance: `CRBIOLM_DATA_DIR=/data python -c "import mpcr_rag.query.gbif_map"` works
on a machine that is not yours.

---

## Phase 2 — The REST layer (1.5 days)

New package `api/` — a thin adapter, **no business logic**, sibling to MCP and CLI
per [ARCHITECTURE.md](ARCHITECTURE.md#L92).

### Endpoints (L0/L1 only)

| Method | Path | Backing | Tier |
|---|---|---|---|
| GET | `/health` | — | — |
| GET | `/v1/vocabulary` | `intent.load_vocab` | L0 |
| GET | `/v1/species` | `retriever.filter_all` | L0 |
| GET | `/v1/species/{name}` | `local_store.get` | L0 |
| GET | `/v1/species/{name}/occurrences` | `gbif_map.get_points` + `filter_points` | L1 |
| GET | `/v1/species/{name}/map` | precomputed lookup, fallback render | L1 |
| POST | `/v1/parse-distribution` | `parser.build_ficha` | L1 |
| GET | `/maps/{file}.png` | StaticFiles | — |

`GET /v1/species` query params mirror `search_species`: `habit, elev_lo, elev_hi,
vertiente, region, forest_type, family, flowering_month, endemic`, plus `limit`/`offset`
for pagination (the MCP tool is exhaustive; an HTTP client needs paging).

### Response envelope

Keep the provenance envelope — it is the verifiability claim, the students should
surface it in their UI, and it is what makes the data institutionally acceptable later:

```json
{
  "value": { },
  "source": "MPCR",
  "citation": "Manual de Plantas de Costa Rica, Tomo IV, p. 320",
  "confidence": "exact",
  "caveat": ""
}
```

`source` in `{MPCR, GBIF, DEM, SINAC}` (and `+`-joined combinations),
`confidence` in `{exact, estimated, insufficient}`.

- [ ] **2.1** Pydantic models for the envelope + each `value` shape.
- [ ] **2.2** Implement endpoints as thin calls into existing core functions.
- [ ] **2.3** **Rule, written down: SQLite must not leak past `local_store`.** The API
      calls `local_store.get` / `filter_all` and never raw SQL. It is clean today; keeping
      it clean is what makes both the pgvector migration and a node handoff a one-module
      change instead of an audit.
- [ ] **2.4** Errors: 404 unknown species, 422 bad filter value (include the valid set
      from the vocabulary in the error body), 429 rate limited, 503 data volume missing.
- [ ] **2.5** CORS enabled for `localhost` origins so their frontend can call it.
- [ ] **2.6** Export the OpenAPI spec to `docs/openapi.json` and commit it — the frozen
      contract they build against.
- [ ] **2.7** Lightweight request logging (endpoint, params, status, duration). This is
      the adoption evidence for paper 2 — decide it now, because retrofitting loses the data.

Acceptance: `docs/openapi.json` exists and a mock server generated from it answers every
endpoint with realistic fixtures. **Send this to the students at this point** — before
Phase 3 or 4 exist.

---

## Phase 3 — Precompute the map corpus (1 day + batch time)

The input space is bounded at **5,791 species**. Render once, serve static forever.

- [ ] **3.1** `utils/precompute_maps.py` — iterate the catalog, render, write to
      `maps/{vector_id}.v{N}.png`, skip existing.
- [ ] **3.2** Manifest table/JSON: `vector_id, filename, render_version, git_sha,
      generated_at, n_gbif_pts, sha256, status`. `render_version` tied to the
      parser/renderer commit — **without this the corpus rots invisibly** and you cannot
      tell which maps predate a parser fix. It is also what makes adding species safe:
      new entries render, existing ones are skipped unless the version moved.
- [ ] **3.3** Record failures rather than crashing the batch; expose `status` so the API
      can 404 with a reason instead of hanging on a render.
- [ ] **3.4** Run on the Pi or your desktop. Estimate ~3 GB total.
- [ ] **3.5** `/v1/species/{name}/map` serves from the corpus; on-demand render only as
      a rate-limited fallback for species not yet precomputed.

Acceptance: 5,791 PNGs + manifest; the endpoint never invokes matplotlib in the common path.

---

## Phase 4 — Container bundle (1 day)

The existing [Dockerfile](../Dockerfile) is for the **experiment viewer** (`app.app:app`,
Subsystem B) — do not reuse it. New `api/Dockerfile`.

- [ ] **4.1** `api/Dockerfile` on `python:3.12-slim`, installing `requirements_api.txt`.
- [ ] **4.2** `docker-compose.yml`: the API service + a `./data` volume mount.
      No Postgres yet — SQLite is fine for the handoff; pgvector is a separate track.
- [ ] **4.3** Bake nothing heavy into the image; data arrives via volume so the image
      stays small and the data bundle updates independently.
- [ ] **4.4** Healthcheck verifying the data volume is present and readable, so a
      misconfigured mount fails loudly at startup rather than at first request.
- [ ] **4.5** Test on a clean machine with no CR-BioLM checkout.

Acceptance: `docker compose up` on a laptop that has never seen this project serves
`/v1/species/Peltogyne%20purpurea` correctly.

---

## Phase 5 — Data bundle distribution (half day)

~115 MB core, or ~3.1 GB with the precomputed map corpus.

| Component | Size | Needed for |
|---|---|---|
| `data_raw/topography/altitud_cr.tif` (DEM) | 313 KB | elevation mask |
| `data_raw/regiones_botanicas/` | 3.3 MB | region matching |
| `data_raw/vectors/areas_protegidas_v2.*` | 4.3 MB | protected areas layer |
| `data_raw/Cartografia/` (prov/cantonal/distrital/Holdridge) | ~70 MB | base + zoom scopes |
| `data_raw/gazetteer/entities.csv` | 12 KB | place-name resolution |
| `mpcr_rag/data/fichas.sqlite` | 14 MB | catalog |
| `mpcr_rag/data/gbif_cache/` | 6.7 MB | occurrences offline |
| `species_counts.json` (derived from snapshot) | small | count-based ranking |
| **Core subtotal** | **~115 MB** | |
| Precomputed maps (optional) | ~3 GB | instant map serving |

**Do not ship `gbif_snapshot/` (149 MB)** — derive `species_counts.json` from it once
and ship that instead.

- [ ] **5.1** `utils/make_data_bundle.py` producing a versioned tarball + SHA256.
- [ ] **5.2** Include a **machine-readable attribution manifest** (`ATTRIBUTION.json`):
      per layer, its source, license, version, and DOI where one exists. A GBIF node
      will ask for exactly this; having it prepared shortens that conversation.
- [ ] **5.3** Host for now on Google Drive (private link) for the students.
- [ ] **5.4** **Licensing check before any public deposit**: confirm redistribution terms
      for the IGN Cartografía and SINAC protected-areas shapefiles. Rendered maps with
      attribution are very likely fine; redistributing source shapefiles may not be.
      Gates Zenodo and RedBioMA, not the private handoff.
- [ ] **5.5** Later: Zenodo deposit with a DOI — free, permanent, citable, and doubles
      as the durable artifact.

---

## Phase 6 — Handoff (half day)

- [ ] **6.1** [BACHELOR_QUICKSTART.md](BACHELOR_QUICKSTART.md) — written, keep it current.
- [ ] **6.2** Kickoff call: walk the contract, the envelope, and the caveat semantics.
      Make sure they understand `confidence: insufficient` and a non-empty `caveat` are
      things to render, not swallow.
- [ ] **6.3** Agree a support channel and a "contract is frozen, ask before assuming" rule.
- [ ] **6.4** Tell them the known caveat cases so they design for them, especially
      `elev_min == elev_max` producing an empty DEM mask (see *Talamancaster minusculus*).

---

## Extensibility — adding volumes and species

The storage layer is already incremental; the cost is upstream in ingest.

**Already solved:**
- [local_store.upsert](../mpcr_rag/store/local_store.py#L45) uses `INSERT OR REPLACE`
  keyed on `vector_id` — idempotent, incremental.
- [pinecone_client.upsert_fichas](../mpcr_rag/store/pinecone_client.py#L81) — same,
  batched by `_id`.
- [config.CORPUS](../mpcr_rag/config.py#L57) auto-discovers PDFs from `_VOL_DIRS`, so a
  new volume **in the same format** is one dictionary entry.

**Three different asks hide under "add more species".** Be explicit about which one is
being requested:

| Ask | Cost |
|---|---|
| A volume of the same OCR era (II–VI) | One `_VOL_DIRS` entry + re-ingest |
| Vols VII / VIII | Segmenter tuning — 2014 MBOT/InDesign layout, see [config.py:33](../mpcr_rag/config.py#L33) |
| A non-Manual catalog | New extractor; the parser and renderer are reusable, the segmenter is not |

**To do:**
- [ ] **E.1** Add a `--volume` / `--family` filter to
      [build_from_corpus](../mpcr_rag/store/local_store.py#L112) and
      [build_catalog.main](../mpcr_rag/ingest/build_catalog.py#L16). Today both iterate
      all of `CORPUS`. Safe to re-run (upsert is idempotent) but it re-extracts
      everything. ~30 min, and it turns "add a volume" into a routine operation.
- [ ] **E.2** Report the **top unresolved gazetteer tokens** at the end of each ingest
      run. New families from new regions will reference place names absent from
      `entities.csv`; the parser already tracks `unresolved_tokens`, so growth is
      measurable rather than silently degrading. Surfacing it makes extending the
      gazetteer a checklist item instead of a discovery.
- [ ] **E.3** After any ingest, re-run `precompute_maps.py` — new species render, existing
      ones skip via the manifest.

**Naming hazard for the docs:** `utils/add_new_species_to_catalog.py` writes
`outputs/picked_species_enhanced*.csv`, which is the **SDM species list (Subsystem B)** —
a different catalog from `fichas.sqlite`. State this explicitly so nobody extends the
wrong one.

---

## Hosting

The shared instance is **not the Pi** — others depend on it.

| Workload | Uptime matters? | Host |
|---|---|---|
| Batch map rendering | No | Pi / desktop |
| Shared API instance | Yes — others depend on it | Cheap VPS (~$5–7/mo) |
| Their development | No | Their own laptop, via compose |
| Static site + map gallery | Yes, but static | Cloudflare/GitHub Pages, free |

Check for free institutional hosting (university research computing, GitHub Student Pack,
cloud education credits, RedBioMA contacts) before paying for the VPS.

Once maps are precomputed the serving path is database reads plus static files — no
geopandas, no rasterio — so the cheapest tier is genuinely sufficient.

### Path to external hosting (RedBioMA / GBIF node)

What already works in favour of it: the container is the portability unit; the provenance
envelope puts attribution in the data rather than a README; the precomputed map corpus can
be hosted with no Python at all (the lowest-commitment version of the ask); the
sibling-adapter design lets them take REST and ignore MCP.

**The blocker is not technical.** It is the Manual copyright and the IGN/SINAC
redistribution terms (5.4). That is the only item with no engineering workaround, and
permissions run on institutional time. **Start that conversation now, in parallel** — not
when the code is ready.

---

## Effort

| Phase | Estimate |
|---|---|
| 1 Portability | 0.5 d |
| 2 REST layer | 1.5 d |
| 3 Precompute | 1 d + batch |
| 4 Container | 1 d |
| 5 Data bundle | 0.5 d |
| 6 Handoff | 0.5 d |
| E Extensibility fixes | 0.5 d |
| **Total** | **~5.5 focused days** |

Phase 2 through step 2.6 is the critical path for unblocking the students. Everything
after that lands incrementally behind the frozen contract.

---

## Explicitly out of scope

- L2 tools (`semantic_search`, `answer_question`) — paid keys, not their use case
- The Postgres + pgvector migration — separate track, do it before paper 2's eval
- PostGIS — attractive, but scope creep against the paper deadline
- The personal site / public gallery — falls out of Phases 3 and 5 later
