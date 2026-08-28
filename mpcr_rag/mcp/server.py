"""
MPCR-RAG MCP server.

Exposes the Manual de Plantas de Costa Rica catalog as MCP tools callable by
Claude Code or any MCP client.  Tools are grouped by cost:

  L0 — free, instant, SQLite only, no API keys required
  L1 — seconds, I/O cached to disk, no LLM
  L2 — requires PINECONE_API_KEY and OPENROUTER_API_KEY

Every tool returns a provenance envelope:
  {value, source, citation, confidence, caveat}
  source     : one of MPCR | GBIF | DEM | SINAC
  confidence : one of exact | estimated | insufficient
  caveat     : non-empty string when the caller should distrust the value

Run locally:
  python -m mpcr_rag.mcp.server          (stdio, for Claude Code)
  mcp dev mpcr_rag/mcp/server.py         (MCP Inspector)
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Optional

from mcp.server.mcpserver import MCPServer

from mpcr_rag import config
from mpcr_rag.query import intent as intent_mod
from mpcr_rag.query import gbif_map
from mpcr_rag.query.retriever import filter_all
from mpcr_rag.store import local_store

server = MCPServer("mpcr-rag", version="0.1.0")

# ---------------------------------------------------------------------------
# Shared lazy state — connections opened once per process lifetime
# ---------------------------------------------------------------------------

_conn = None
_index = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = local_store.connect(config.SQLITE_PATH)
    return _conn


def _get_index():
    global _index
    if _index is None:
        if not os.environ.get("PINECONE_API_KEY"):
            return None
        from mpcr_rag.store import pinecone_client as pc
        _index = pc.ensure_index()
    return _index


# ---------------------------------------------------------------------------
# Provenance envelope
# ---------------------------------------------------------------------------

def _envelope(
    value,
    *,
    source: str,
    citation: str = "",
    confidence: str = "exact",
    caveat: str = "",
) -> dict:
    return {
        "value": value,
        "source": source,
        "citation": citation,
        "confidence": confidence,
        "caveat": caveat,
    }


def _ficha_summary(f) -> dict:
    return {
        "species": f.species,
        "family": f.family,
        "volume": f.volume,
        "pages": f.pages,
        "elev_min": f.elev_min,
        "elev_max": f.elev_max,
        "vertientes": f.vertientes,
        "regions": f.regions,
        "habits": f.habits,
        "endemic_cr": f.endemic_cr,
        "forest_types": f.forest_types,
        "flowering_months": f.flowering_months,
    }


# ---------------------------------------------------------------------------
# L0 — free, instant, SQLite only
# ---------------------------------------------------------------------------

@server.tool()
def get_vocabulary() -> dict:
    """Return the controlled vocabulary for all filter fields.

    Cost: L0 — free, instant, SQLite only.  Call this first to know which
    values are valid for the filter parameters of search_species and
    semantic_search.

    Returns the full set of distinct values present in the catalog for:
    habits, forest_types, regions, vertientes, families.
    """
    vocab = intent_mod.load_vocab(_get_conn())
    return _envelope(
        vocab,
        source="MPCR",
        citation="Manual de Plantas de Costa Rica (Hammel et al.) — 5,791 species",
        confidence="exact",
    )


@server.tool()
def search_species(
    habit: Optional[str] = None,
    elev_lo: Optional[int] = None,
    elev_hi: Optional[int] = None,
    vertiente: Optional[str] = None,
    region: Optional[str] = None,
    forest_type: Optional[str] = None,
    family: Optional[str] = None,
    flowering_month: Optional[int] = None,
    endemic: Optional[bool] = None,
) -> dict:
    """Exhaustive structured search over the Manual catalog (no Pinecone, no top-k cap).

    Cost: L0 — free, instant, SQLite only.  All filters are AND-combined;
    omit a parameter to skip that filter.  Use get_vocabulary() to see valid
    values for habit, vertiente, region, forest_type, family.

    Args:
        habit: growth form e.g. "árbol", "arbusto", "epifita", "hierba"
        elev_lo: minimum elevation overlap (metres)
        elev_hi: maximum elevation overlap (metres)
        vertiente: "Caribe" or "Pacífico"
        region: canonical botanical region name e.g. "Cordillera de Talamanca"
        forest_type: e.g. "muy húmedo", "pluvial", "nuboso", "páramo"
        family: plant family e.g. "Lauraceae", "Orchidaceae"
        flowering_month: 1–12
        endemic: true = Costa Rica endemics only

    Returns a provenance envelope whose value is
      {count: int, species: [{species, family, elev_min, elev_max, vertientes, volume, pages}]}
    """
    candidates = filter_all(
        _get_conn(),
        habit=habit,
        elev_lo=elev_lo,
        elev_hi=elev_hi,
        vertiente=vertiente,
        region=region,
        forest_type=forest_type,
        family=family,
        flowering_month=flowering_month,
        endemic=endemic,
    )
    summaries = [_ficha_summary(f) for f in candidates]
    return _envelope(
        {"count": len(candidates), "species": summaries},
        source="MPCR",
        citation="Manual de Plantas de Costa Rica — structured fields only",
        confidence="exact",
        caveat="" if candidates else "No species matched the given filters.",
    )


@server.tool()
def get_species(name: str) -> dict:
    """Fetch the full ficha for a single species by scientific name.

    Cost: L0 — free, instant, SQLite only.

    Args:
        name: scientific name e.g. "Peltogyne purpurea" or "Talamancaster minusculus"

    Returns a provenance envelope whose value is the full Ficha JSON including
    distribution_paragraph, elev_min/max, vertientes, regions, parks, habits,
    Tomo and página.
    """
    vector_id = name.strip().replace(" ", "_")
    f = local_store.get(_get_conn(), vector_id)
    if f is None:
        return _envelope(
            None,
            source="MPCR",
            confidence="insufficient",
            caveat=f"Species '{name}' not found in catalog. Check spelling or use search_species().",
        )
    value = {**_ficha_summary(f), "distribution_paragraph": f.distribution_paragraph}
    return _envelope(
        value,
        source="MPCR",
        citation=f"Manual de Plantas de Costa Rica, Tomo {f.volume}, p. {f.pages}",
        confidence="exact",
    )


# ---------------------------------------------------------------------------
# L1 — seconds, I/O cached, no LLM
# ---------------------------------------------------------------------------

@server.tool()
def parse_distribution_text(
    habitat_raw: str,
    geographic_notes: str,
    species: str,
) -> dict:
    """Parse raw Manual distribution text into a structured DistributionFicha.

    Cost: L1 — pure computation, no network, no LLM.  The parser is heuristic
    (regex + gazetteer), not ML — confidence is "estimated".

    Args:
        habitat_raw: text before the elevation's semicolon boundary
          e.g. "Bosque muy húmedo y pluvial, 0–800(–1200) m"
        geographic_notes: text after the semicolon boundary
          e.g. "vert. Caribe, Cord. de Talamanca"
        species: scientific name (used for GBIF taxon resolution in the parser)

    Returns a provenance envelope whose value is the DistributionFicha as JSON.
    """
    from utils.distribution_map.parser import build_ficha
    df = build_ficha(
        habitat_raw=habitat_raw,
        geographic_notes=geographic_notes,
        species=species,
    )
    value = dataclasses.asdict(df)
    return _envelope(
        value,
        source="MPCR",
        citation="geo_parser + gazetteer (entities.csv, José Araya 2026)",
        confidence="estimated",
        caveat="Parser is rule-based. Check unresolved_tokens for place names it could not match.",
    )


@server.tool()
def get_occurrences(
    species: str,
    elev_lo: Optional[int] = None,
    elev_hi: Optional[int] = None,
    vertiente: Optional[str] = None,
    region: Optional[str] = None,
) -> dict:
    """Fetch and filter Costa Rica GBIF occurrences for a species.

    Cost: L1 — first call fetches from GBIF API and caches to disk; subsequent
    calls are instant.  No LLM, no Pinecone.

    Args:
        species: scientific name
        elev_lo: minimum elevation filter (metres)
        elev_hi: maximum elevation filter (metres)
        vertiente: "Caribe" or "Pacífico"
        region: botanical region name

    Returns a provenance envelope whose value is:
      {n, bbox, elev_mean, elev_std, elev_min_obs, elev_max_obs, region_breakdown}
    """
    pts = gbif_map.get_points(species)
    if pts.empty:
        return _envelope(
            {"n": 0},
            source="GBIF",
            citation="GBIF occurrence API — Costa Rica (hasCoordinate=true, hasGeospatialIssue=false)",
            confidence="insufficient",
            caveat=f"No GBIF occurrences found for '{species}' in Costa Rica.",
        )

    filtered = gbif_map.filter_points(pts, elev_lo=elev_lo, elev_hi=elev_hi,
                                      vertiente=vertiente, region=region)
    n = len(filtered)
    caveat = ""
    if n == 0:
        caveat = "Filters excluded all occurrences. Try relaxing elev_lo/elev_hi or vertiente."

    elev_col = filtered["Altitud"] if "Altitud" in filtered.columns else None
    value: dict = {"n": n}
    if elev_col is not None and not elev_col.dropna().empty:
        value["elev_mean"] = round(float(elev_col.mean()), 1)
        value["elev_std"] = round(float(elev_col.std()), 1)
        value["elev_min_obs"] = int(elev_col.min())
        value["elev_max_obs"] = int(elev_col.max())

    if "Nombre" in filtered.columns:
        value["region_breakdown"] = filtered["Nombre"].value_counts().to_dict()

    if "geometry" in filtered.columns and n > 0:
        b = filtered.total_bounds  # minx, miny, maxx, maxy
        value["bbox"] = {"lon_min": round(b[0], 4), "lat_min": round(b[1], 4),
                         "lon_max": round(b[2], 4), "lat_max": round(b[3], 4)}

    return _envelope(
        value,
        source="GBIF",
        citation="GBIF occurrence API — Costa Rica (hasCoordinate=true, hasGeospatialIssue=false)",
        confidence="estimated",
        caveat=caveat,
    )


@server.tool()
def render_species_map(species: str) -> dict:
    """Render the Manual-grounded distribution map for a single named species.

    Cost: L1 — re-derives DistributionFicha from the stored distribution paragraph
    (no LLM), then renders via the validated map pipeline with GBIF presence points.
    Map is cached by species vector_id.

    Args:
        species: scientific name e.g. "Peltogyne purpurea"

    Returns a provenance envelope whose value is:
      {map_path: str, n_pts: int, layer_summary: str}
    """
    f = local_store.get(_get_conn(), species.strip().replace(" ", "_"))
    if f is None:
        return _envelope(
            None,
            source="MPCR",
            confidence="insufficient",
            caveat=f"Species '{species}' not found in catalog.",
        )

    caveat = ""
    if f.elev_min is not None and f.elev_min == f.elev_max:
        caveat = (
            f"WARNING: Manual gives a single elevation point ({f.elev_min} m) — "
            "the DEM elevation mask will be empty (zero-width range). "
            "The map is rendered without a cyan elevation layer. "
            "Use get_occurrences() to cross-check with GBIF evidence."
        )

    map_path, n_pts = gbif_map.single_species_map(f)
    layers = [
        "CR base (provinces)",
        "Unmatched botanical regions (dark gray)",
        "Matched regions — Manual text (white outline)",
    ]
    if n_pts >= 5:
        layers.append(f"GBIF presence points (n={n_pts}, red dots)")
    layers += [
        "DEM elevation mask (cyan fill, within matched regions)",
        "Parks mentioned (orange outline)",
        "Protected areas filtered by elevation (amber outline)",
    ]
    return _envelope(
        {
            "map_path": str(map_path),
            "n_pts": n_pts,
            "layer_summary": " | ".join(layers),
        },
        source="MPCR+GBIF+DEM",
        citation=f"Manual de Plantas de Costa Rica, Tomo {f.volume}, p. {f.pages}; "
                 "GBIF occurrences CR; DEM altitud_cr.tif; "
                 "Regiones Botánicas (José Araya, 2026)",
        confidence="estimated",
        caveat=caveat,
    )


# ---------------------------------------------------------------------------
# L2 — requires PINECONE_API_KEY and OPENROUTER_API_KEY
# ---------------------------------------------------------------------------

def _check_l2() -> str:
    missing = []
    if not os.environ.get("PINECONE_API_KEY"):
        missing.append("PINECONE_API_KEY")
    if not os.environ.get("OPENROUTER_API_KEY"):
        missing.append("OPENROUTER_API_KEY")
    return ", ".join(missing)


@server.tool()
def semantic_search(
    query_text: str,
    habit: Optional[str] = None,
    elev_lo: Optional[int] = None,
    elev_hi: Optional[int] = None,
    vertiente: Optional[str] = None,
    region: Optional[str] = None,
    family: Optional[str] = None,
    endemic: Optional[bool] = None,
    top_k: int = 12,
) -> dict:
    """Semantic search over the Manual catalog using Pinecone vector index.

    Cost: L2 — requires PINECONE_API_KEY.  Combines dense vector retrieval
    (hosted e5 embeddings) with structured metadata filters.  Returns species
    ranked by relevance to the query text.

    Args:
        query_text: natural-language description e.g. "árbol de madera dura en bosque nuboso"
        habit, elev_lo, elev_hi, vertiente, region, family, endemic: same as search_species
        top_k: maximum results (default 12)

    Returns a provenance envelope whose value is a ranked list of species summaries.
    """
    missing = _check_l2()
    if "PINECONE_API_KEY" in missing:
        return _envelope(
            None,
            source="MPCR",
            confidence="insufficient",
            caveat=f"API key not configured: {missing}. semantic_search requires Pinecone.",
        )

    from mpcr_rag.query.retriever import pattern_b
    constraints = {k: v for k, v in dict(
        habit=habit, elev_lo=elev_lo, elev_hi=elev_hi,
        vertiente=vertiente, region=region, family=family, endemic=endemic,
    ).items() if v is not None}

    results = pattern_b(
        query_text, top_k=top_k, conn=_get_conn(), index=_get_index(), **constraints
    )
    value = [{"score": round(score, 4), **_ficha_summary(f)} for f, score in results]
    return _envelope(
        value,
        source="MPCR",
        citation="Manual de Plantas de Costa Rica — Pinecone e5 semantic index",
        confidence="estimated",
        caveat="" if results else "No results matched. Try broader filters or different query_text.",
    )


@server.tool()
def answer_question(question: str) -> dict:
    """Answer a natural-language question about Costa Rica flora.

    Cost: L2 — requires PINECONE_API_KEY and OPENROUTER_API_KEY.  Routes to one
    of three fixed, citable recipes:
      A  (lookup)     — named species → Manual-grounded map + grounded text
      B  (list)       — geospatial filter → GBIF evidence map + grounded text
      B→A (superlative) — filter + deterministic extremal selection → single-species map

    Args:
        question: question in Spanish or English about CR flora
          e.g. "¿cuál arbusto endémico crece a mayor altitud en Talamanca?"

    Returns a provenance envelope whose value is:
      {text, mode, constraints, selector, map_path, n_pts, citations}
    where text is a grounded answer citing Manual (Tomo, página) for every claim.
    """
    missing = _check_l2()
    if missing:
        return _envelope(
            None,
            source="MPCR",
            confidence="insufficient",
            caveat=f"API keys not configured: {missing}. answer_question requires Pinecone + OpenRouter.",
        )

    from mpcr_rag.query import answer as ans_mod
    a = ans_mod.answer(question, conn=_get_conn(), index=_get_index())

    citations = [
        f"Manual de Plantas de Costa Rica, Tomo {f.volume}, p. {f.pages} ({f.species})"
        for f, _ in a["results"]
    ]
    value = {
        "text": a["text"],
        "mode": a["mode"],
        "constraints": a["constraints"],
        "selector": a.get("selector"),
        "map_path": str(a["map_path"]),
        "n_pts": a["n_pts"],
        "citations": citations,
    }
    return _envelope(
        value,
        source="MPCR+GBIF",
        citation="; ".join(citations) if citations else "Manual de Plantas de Costa Rica",
        confidence="estimated",
        caveat=(
            "Answer is grounded in Manual text only. GBIF occurrence data provides "
            "the map layer but does not alter the text. "
            "gbif_count selector is a collection-effort proxy, not real abundance."
            if a.get("selector") and "gbif_count" in (a.get("selector") or "") else ""
        ),
    )


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@server.resource("mpcr://vocabulary")
def vocabulary_resource() -> str:
    """The controlled vocabulary — inject this into an agent's system prompt
    to guide filter value selection before calling search_species or semantic_search."""
    vocab = intent_mod.load_vocab(_get_conn())
    return json.dumps(vocab, ensure_ascii=False, indent=2)


@server.resource("mpcr://species/{vector_id}")
def species_resource(vector_id: str) -> str:
    """One ficha as addressable context.  vector_id is the species name with
    spaces replaced by underscores e.g. 'Peltogyne_purpurea'."""
    f = local_store.get(_get_conn(), vector_id)
    if f is None:
        return json.dumps({"error": f"Species '{vector_id}' not found."})
    return json.dumps({**_ficha_summary(f), "distribution_paragraph": f.distribution_paragraph},
                      ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server.run()
