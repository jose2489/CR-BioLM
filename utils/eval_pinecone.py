"""
Pinecone index evaluation.

Checks:
  1. Coverage   — total indexed vectors vs SQLite species count
  2. Self-retrieval — for a random sample, does the species rank #1 when queried
                      with its own distribution_paragraph?
  3. Semantic quality — ecological free-text queries; inspect top-5 results
  4. Metadata filter — elevation + endemic filter sanity check

Usage:
    python utils/eval_pinecone.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from mpcr_rag.store import local_store, pinecone_client
from mpcr_rag import config

# ── helpers ──────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)


def show_hits(hits: list[dict], n: int = 5) -> None:
    for h in hits[:n]:
        elev = f"{h.get('elev_min', '?')}–{h.get('elev_max', '?')} m"
        endemic = "★ endémica" if h.get("endemic") else ""
        print(f"  {h['score']:.3f}  {h.get('species', h['id']):<35} [{elev}] {endemic}")


# ── 1. Coverage ──────────────────────────────────────────────────────────────

section("1. COBERTURA — vectores en Pinecone vs. fichas en SQLite")

conn = local_store.connect(config.SQLITE_PATH)
sqlite_ids = {row[0] for row in conn.execute("SELECT vector_id FROM fichas")}
sqlite_n = len(sqlite_ids)
conn.close()
print(f"  SQLite fichas : {sqlite_n:,}")

idx = pinecone_client.ensure_index()
stats = idx.describe_index_stats()
ns = stats.get("namespaces", {}) or {}
pinecone_n = ns.get("mpcr", {}).get("vector_count", 0)
print(f"  Pinecone (ns=mpcr) : {pinecone_n:,}")
print(f"  Diferencia          : {sqlite_n - pinecone_n:+,}")


# ── 2. Self-retrieval precision ───────────────────────────────────────────────

section("2. SELF-RETRIEVAL — ¿el texto propio recupera la ficha correcta? (muestra n=30)")

conn = local_store.connect(config.SQLITE_PATH)
rows = conn.execute(
    "SELECT vector_id, species, ficha_json FROM fichas "
    "WHERE json_extract(ficha_json, '$.distribution_paragraph') IS NOT NULL "
    "  AND json_extract(ficha_json, '$.distribution_paragraph') != '' "
    "ORDER BY RANDOM() LIMIT 30"
).fetchall()
conn.close()

import json
hit_at_1 = 0
hit_at_5 = 0
misses = []

for vid, species, fjson in rows:
    ficha = json.loads(fjson)
    dist_text = ficha.get("distribution_paragraph", "")
    if not dist_text:
        continue
    results = pinecone_client.search(dist_text, top_k=5, index=idx)
    ids_returned = [r["id"] for r in results]
    if ids_returned and ids_returned[0] == vid:
        hit_at_1 += 1
        hit_at_5 += 1
    elif vid in ids_returned:
        hit_at_5 += 1
        misses.append((species, ids_returned[0], results[0]["score"]))
    else:
        misses.append((species, ids_returned[0] if ids_returned else "—", 0.0))

n = len(rows)
print(f"  Hit@1 : {hit_at_1}/{n} ({100*hit_at_1/n:.0f}%)")
print(f"  Hit@5 : {hit_at_5}/{n} ({100*hit_at_5/n:.0f}%)")
if misses:
    print(f"\n  Casos que NO quedaron en top-1:")
    for sp, top_id, sc in misses:
        print(f"    {sp:<35}  → top-1 fue '{top_id}' (score {sc:.3f})")


# ── 3. Semantic quality ───────────────────────────────────────────────────────

section("3. CALIDAD SEMÁNTICA — consultas ecológicas libres")

QUERIES = [
    "bosque nuboso lluvioso de la Cordillera de Talamanca, alturas sobre 2000 m",
    "manglar y bosque de galería en la llanura del Caribe",
    "bosque seco del Pacífico Norte, Guanacaste",
    "páramo y subpáramo, zonas altas sobre 3000 metros",
    "árboles maderables de bosque muy húmedo tropical, vertiente Pacífico",
]

for q in QUERIES:
    print(f"\n  Consulta: «{q}»")
    hits = pinecone_client.search(q, top_k=5, index=idx)
    show_hits(hits)


# ── 4. Metadata filter ────────────────────────────────────────────────────────

section("4. FILTROS DE METADATA — endémicas sobre 2500 m en bosque nuboso")

hits = pinecone_client.search(
    "bosque nuboso de altura",
    top_k=8,
    flt={"endemic": True, "elev_max": {"$gte": 2500}},
    index=idx,
)
print(f"  Resultados (endémicas, elev_max ≥ 2500 m):")
show_hits(hits, n=8)

print("\n✓ Evaluación completada.\n")
