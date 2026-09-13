"""Pinecone vs pgvector retrieval parity — the gate for switching backends.

Same model on both sides (multilingual-e5-large), hosted vs local, so results should
nearly coincide. Differences can come from truncation, numeric precision or text
changes between catalog builds; this report quantifies them before the default
backend flips.

Comparison is restricted to species present in BOTH indexes with unchanged
distribution text and filterable metadata: the pgvector store also holds species the Pinecone index
never had, which would otherwise count as spurious disagreement.

  1. Self-retrieval: query = a species' own distribution paragraph; rank of itself.
  2. Filtered NL queries: overlap@k between backends (Jaccard) and top-1 agreement.

Run:  python -m mpcr_rag.eval.vector_parity [--n 200] [--k 10]
Writes eval/results/vector_parity.{csv,md}
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import sqlite3
import statistics as st
from pathlib import Path

from .. import config
from ..schema import Ficha
from ..store import pg_store, pinecone_client as pc

_RESULTS = Path(__file__).resolve().parent / "results"

# Filtered natural-language queries in the shape intent.parse_intent produces.
QUERIES = [
    ("bosque nuboso de altura en la Cordillera de Talamanca", {}),
    ("arbustos de tierras bajas", {"habit": "arbusto", "elev_lo": 150, "elev_hi": 300}),
    ("bosque de altura, robledales", {"elev_lo": 2000, "vertiente": "Pacífico", "endemic": True}),
    ("palmas del Pacífico", {"habit": "palma", "vertiente": "Pacífico"}),
    ("epífitas de bosque pluvial", {"habit": "epífita", "forest_type": "pluvial"}),
    ("árboles de bosque seco en Guanacaste", {"habit": "árbol", "forest_type": "seco"}),
    ("hierbas de páramo", {"habit": "hierba", "elev_lo": 3000}),
    ("especies de manglar y zonas costeras", {"elev_hi": 50}),
    ("Lauraceae de la vertiente Caribe", {"family": "Lauraceae", "vertiente": "Caribe"}),
    ("plantas que florecen en enero en la Península de Osa", {"flowering_month": 1}),
    ("bejucos de bosque muy húmedo", {"habit": "bejuco", "forest_type": "muy húmedo"}),
    ("endémicas de la Cordillera Central", {"region": "Cordillera Central", "endemic": True}),
    ("orillas de caminos y potreros", {}),
    ("sotobosque de bosque primario", {"elev_hi": 800}),
]


_FILTER_FIELDS = ("habits", "vertientes", "regions", "forest_types", "family", "endemic_cr",
                  "flowering_months", "elev_min", "elev_max", "elev_outlier_min",
                  "elev_outlier_max")


def _signature(f: Ficha) -> tuple:
    """Distribution text + every filterable field. Species whose signature differs
    between builds match different filters on each backend, which is a catalog change,
    not a retrieval difference."""
    text = f.distribution_paragraph or f.species
    return (text,) + tuple(tuple(sorted(v)) if isinstance(v, list) else v
                           for v in (getattr(f, k) for k in _FILTER_FIELDS))


def _load(sqlite_path: Path) -> dict[str, tuple]:
    conn = sqlite3.connect(sqlite_path)
    out = {vid: _signature(Ficha.from_json(js))
           for vid, js in conn.execute("SELECT vector_id, ficha_json FROM fichas")}
    conn.close()
    return out


def _restrict(hits: list[dict], allowed: set[str], k: int) -> list[str]:
    return [h["id"] for h in hits if h["id"] in allowed][:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-sqlite", type=Path,
                    default=config.DATA_DIR / "fichas.pre_trackB_20260912.sqlite",
                    help="catalog build that the Pinecone index was upserted from")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    old = _load(a.old_sqlite)
    cur_build = _load(config.SQLITE_PATH)
    pgc = pg_store.connect()
    with pgc.cursor() as cur:
        cur.execute("SELECT vector_id, text FROM mpcr.ficha_chunks WHERE section='distribution'")
        new = dict(cur.fetchall())
    common = {vid for vid in old.keys() & new.keys() & cur_build.keys()
              if old[vid] == cur_build[vid] and new[vid] == old[vid][0]}
    print(f"pinecone={len(old)} pgvector={len(new)} common-unchanged={len(common)}")

    idx = pc.ensure_index()
    pool = 60   # over-fetch so restricting to common ids still leaves k
    rows = []

    rng = random.Random(a.seed)
    sample = rng.sample(sorted(common), min(a.n, len(common)))
    for i, vid in enumerate(sample, 1):
        text = new[vid]
        p_ids = _restrict(pc.search(text, top_k=pool, index=idx), common, a.k)
        g_ids = _restrict(pg_store.search(text, top_k=pool, conn=pgc), common, a.k)
        rows.append({
            "kind": "self", "query": vid, "filters": "",
            "pinecone_self_rank": p_ids.index(vid) + 1 if vid in p_ids else "",
            "pgvector_self_rank": g_ids.index(vid) + 1 if vid in g_ids else "",
            "jaccard_at_k": len(set(p_ids) & set(g_ids)) / max(1, len(set(p_ids) | set(g_ids))),
            "top1_agree": bool(p_ids and g_ids and p_ids[0] == g_ids[0]),
        })
        if i % 50 == 0:
            print(f"  self-retrieval {i}/{len(sample)}", flush=True)

    for text, flt in QUERIES:
        p_ids = _restrict(pc.search(text, top_k=pool, flt=_pinecone_filter(flt), index=idx),
                          common, a.k)
        g_ids = _restrict(pg_store.search(text, top_k=pool, conn=pgc, **flt), common, a.k)
        rows.append({
            "kind": "nl", "query": text, "filters": str(flt),
            "pinecone_self_rank": "", "pgvector_self_rank": "",
            "jaccard_at_k": len(set(p_ids) & set(g_ids)) / max(1, len(set(p_ids) | set(g_ids))),
            "top1_agree": bool(p_ids and g_ids and p_ids[0] == g_ids[0]),
            "n_pinecone": len(p_ids), "n_pgvector": len(g_ids),
        })
    pgc.close()

    _RESULTS.mkdir(exist_ok=True)
    fields = ["kind", "query", "filters", "pinecone_self_rank", "pgvector_self_rank",
              "jaccard_at_k", "top1_agree", "n_pinecone", "n_pgvector"]
    with open(_RESULTS / "vector_parity.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    selfs = [r for r in rows if r["kind"] == "self"]
    nls = [r for r in rows if r["kind"] == "nl"]

    def at1(key):
        return sum(1 for r in selfs if r[key] == 1) / len(selfs)

    def found(key):
        return sum(1 for r in selfs if r[key] != "") / len(selfs)

    md = [
        "# Vector backend parity: Pinecone vs pgvector",
        "",
        f"*Generated {dt.date.today()} — model `{config.LOCAL_EMBED_MODEL}` (local) vs "
        f"Pinecone hosted `{config.EMBED_MODEL}`; k={a.k}; restricted to "
        f"{len(common)} species with identical distribution text AND filterable metadata in both indexes.*",
        "",
        "| Check | Pinecone | pgvector |",
        "|---|---|---|",
        f"| Self-retrieval @1 (n={len(selfs)}) | {at1('pinecone_self_rank'):.1%} | {at1('pgvector_self_rank'):.1%} |",
        f"| Self-retrieval @{a.k} | {found('pinecone_self_rank'):.1%} | {found('pgvector_self_rank'):.1%} |",
        "",
        "| Agreement between backends | Self queries | Filtered NL queries |",
        "|---|---|---|",
        f"| Mean Jaccard@{a.k} | {st.mean(r['jaccard_at_k'] for r in selfs):.3f} | "
        f"{st.mean(r['jaccard_at_k'] for r in nls):.3f} |",
        f"| Top-1 agreement | {sum(r['top1_agree'] for r in selfs)/len(selfs):.1%} | "
        f"{sum(r['top1_agree'] for r in nls)/len(nls):.1%} |",
        "",
        "## Filtered NL queries",
        "",
        f"| Query | Filters | Jaccard@{a.k} | Top-1 agree | n (P / G) |",
        "|---|---|---|---|---|",
    ]
    for r in nls:
        md.append(f"| {r['query']} | `{r['filters']}` | {r['jaccard_at_k']:.2f} | "
                  f"{'yes' if r['top1_agree'] else 'no'} | {r['n_pinecone']} / {r['n_pgvector']} |")
    (_RESULTS / "vector_parity.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def _pinecone_filter(flt: dict) -> dict:
    from ..query.retriever import build_filter
    return build_filter(**flt)


if __name__ == "__main__":
    main()
