# CR-BioLM

Research pipeline for plant distribution modeling in Costa Rica, built on the
*Manual de Plantas de Costa Rica* (Hammel et al.). Two subsystems, two papers,
one shared geospatial core.

---

## Subsystems

| Subsystem | Directory | Paper | Status |
|---|---|---|---|
| **MPCR-RAG** — catalog + geospatial RAG | `mpcr_rag/` | BIP conference (Paper 1) | drafted |
| **SDM Multimodal** — Random Forest + LLM | `main.py`, `models/`, `llm/`, `xai/`, `experiment/` | Biodiversity informatics journal (Paper 2) | in progress |

**Shared core:** `utils/distribution_map/` — the Manual text parser, geographic
entity resolver, and map renderer. Both subsystems consume it; neither owns it.

**Dependency rule:** `mpcr_rag/` may import from `utils/distribution_map/` and
root `config` only. It must not import from `models/`, `llm/`, `xai/`, or
`experiment/`. This keeps the subsystem liftable to its own repo later.

---

## MPCR-RAG (Subsystem A)

Segments Manual PDFs into per-species *fichas*, extracts structured geospatial
fields, indexes in Pinecone + SQLite, and answers natural-language questions with
a grounded text answer and a distribution map.

```
ingest/  →  store/  →  query/
  segmenter → extractor → index      # ingest pipeline
  retriever → intent → answer        # query pipeline
  mcp/                               # MCP tool layer (in progress)
```

**Setup:**
```bash
pip install -r mpcr_rag/requirements.txt
# .env: PINECONE_API_KEY=... OPENROUTER_API_KEY=...
```

**Query examples:**
```bash
python -m mpcr_rag.query.answer "arbustos endémicos sobre 2000 m en Talamanca"
python -m mpcr_rag.query.intent  "¿cuál arbusto crece a mayor elevación en la Península de Nicoya?"
```

See `mpcr_rag/DEMO.md` for the full demo script.

---

## SDM Multimodal (Subsystem B)

Trains a Random Forest per species on GBIF occurrences + WorldClim bioclimatic
variables, explains it with SHAP/LIME, generates two maps, and queries a
multimodal LLM with the images and metrics to produce a structured ecological
profile.

```bash
# Single species
python main.py -s "Quercus costaricensis"

# With a free-text question
python main.py -s "Quercus costaricensis" -q "¿Cómo le afecta el cambio climático?"

# Persona-based question bank
python main.py -s "Peltogyne purpurea" --persona botanico

# Batch
python main.py -f lista_especies.txt --persona botanico
```

Outputs land in `outputs/{Especie}/run_{timestamp}/`: habitat map, GBIF map,
SHAP summary, LIME local explanation, confusion matrix, LLM profile.

---

## Data (not in this repo)

Geospatial assets are gitignored (`data_raw/`) due to size and licensing.
Contact jose2489@gmail.com to obtain the bundle.

| Asset | Size | Notes |
|---|---|---|
| `data_raw/topography/altitud_cr.tif` | 313 KB | DEM, EPSG:4326, Int16 |
| `data_raw/regiones_botanicas/Jose_regiones_botanicas_con_vertiente.shp` | — | Author's own work; credited on every map |
| `data_raw/vectors/areas_protegidas_v2.shp` | — | SINAC protected areas |
| `data_raw/Cartografia/` (cantones, distritos, provincias) | — | IGN cartography |
| `wc2.1_30s_bio_*.tif` (19 layers) | ~51 MB | WorldClim, SDM side only |
| `data_raw/ecoregions/` | ~251 MB | SDM side only |
| Manual de Plantas PDFs | — | Source corpus; not redistributable |

Only `data_raw/gazetteer/entities.csv` and `data_raw/gazetteer/ocr_corrections.csv`
are force-tracked in git — they are the hand-curated lookup tables that the map
pipeline depends on.

---

## Requirements

```bash
pip install -r requirements.txt          # Subsystem B (SDM)
pip install -r mpcr_rag/requirements.txt # Subsystem A (RAG)
```

`.env` at repo root:
```
OPENROUTER_API_KEY=...
PINECONE_API_KEY=...        # MPCR-RAG only
GBIF_USER=...               # optional, for GBIF download
GBIF_PWD=...
GBIF_EMAIL=...
```

Python 3.10+. No R required.

---

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the five core products,
the `DistributionFicha` JSON contract, and the adapter layer design.

---

## Citation

> Araya, J. (2026). *CR-BioLM: Pipeline multimodal de modelado de distribución
> de especies de plantas de Costa Rica usando Machine Learning e Inteligencia
> Artificial Generativa*. Tesis de Maestría.
