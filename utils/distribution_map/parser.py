"""
parser.py — Build a DistributionFicha from raw Manual de Plantas habitat text.

Primary path: gazetteer longest-match → rapidfuzz fuzzy fallback.
Optional LLM fallback for remaining unresolved tokens (pass enable_llm=True).

Public API:
    build_ficha(habitat_raw, geographic_notes, species, enable_llm) -> DistributionFicha
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

from .ficha import DistributionFicha, ElevationRange, EntityRef
from .gazetteer import lookup, lookup_parks, fuzzy_lookup, normalize_text

# ---------------------------------------------------------------------------
# Elevation regex (ported from extract_habitat_from_pdf.py)
# ---------------------------------------------------------------------------
_RE_ELEV_RANGE = re.compile(
    r"""
    (?:\(([—–\-]?\s*\d{1,4})\s*[—–\-]\s*\))?   # leading outlier: (X–)
    (\d{3,4})\s*[—–\-]\s*(\d{3,4})               # main range: X–Y
    (?:\s*\([—–\-]?\s*(\d{3,4})\s*\))?           # trailing outlier: (–Z)
    \s*m\b
    """,
    re.VERBOSE | re.IGNORECASE,
)
_RE_ELEV_SINGLE = re.compile(r"\b(\d{3,4})\s*m\b")
_RE_ELEV_APPROX = re.compile(r"ca\.?\s*(\d{3,4})\s*m\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Vertiente detection (simple, robust enough for normalized text)
# ---------------------------------------------------------------------------
_RE_CARIBE   = re.compile(r"vert(?:iente)?\.?\s*carib|caribena|caribeña", re.IGNORECASE)
_RE_PACIFICO = re.compile(r"vert(?:iente)?\.?\s*pac(?:if)?|pacifica", re.IGNORECASE)
_RE_AMBAS    = re.compile(r"ambas\s*vert", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Forest-type keywords in habitat_raw (before the semicolon separator)
# ---------------------------------------------------------------------------
_FOREST_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"bosque\s+muy\s+h[úu]medo", re.I), "muy húmedo"),
    (re.compile(r"bosque\s+h[úu]medo",       re.I), "húmedo"),
    (re.compile(r"bosque\s+pluvial",          re.I), "pluvial"),
    (re.compile(r"bosque\s+nuboso",           re.I), "nuboso"),
    (re.compile(r"bosque\s+seco",             re.I), "seco"),
    (re.compile(r"bosque\s+de\s+roble",       re.I), "roble"),
    (re.compile(r"p[áa]ramo",                 re.I), "páramo"),
    (re.compile(r"manglar",                   re.I), "manglar"),
    (re.compile(r"matorral",                  re.I), "matorral"),
]


def _parse_elevation(text: str) -> ElevationRange:
    """Extract elevation range from raw habitat text."""
    m = _RE_ELEV_RANGE.search(text)
    if m:
        lo_out, lo, hi, hi_out = m.groups()
        def _clean(v: Optional[str]) -> Optional[float]:
            if v is None:
                return None
            v = re.sub(r"[—–\-\s]", "", v)
            return float(v) if v.isdigit() or (v.lstrip("-").isdigit()) else None
        return ElevationRange(
            min_m        = _clean(lo),
            max_m        = _clean(hi),
            outlier_min_m= _clean(lo_out),
            outlier_max_m= _clean(hi_out),
        )

    # Approximate single value (ca. 1150 m) — use exact, let CSV supplement range
    m2 = _RE_ELEV_APPROX.search(text)
    if m2:
        v = float(m2.group(1))
        return ElevationRange(min_m=v, max_m=v)

    # Single value like "0–700 m" already handled above; bare "1300 m" means
    # we only know the ceiling, so leave it for the CSV supplement.
    # (Returning empty lets maps_only.py apply the catalog-extracted range.)
    return ElevationRange()

    return ElevationRange()


def _parse_forest_types(habitat_raw: str) -> list[str]:
    """Extract forest-type descriptors from the pre-semicolon habitat sentence."""
    # Only look in the first part (before the geographic notes start)
    pre_semi = habitat_raw.split(";")[0] if ";" in habitat_raw else habitat_raw
    found = []
    seen: set[str] = set()
    for pat, label in _FOREST_TYPE_PATTERNS:
        if pat.search(pre_semi) and label not in seen:
            found.append(label)
            seen.add(label)
    return found


def _detect_vertientes(text: str) -> list[str]:
    """Return list of matched vertientes from geographic text."""
    raw = text
    ambas    = bool(_RE_AMBAS.search(raw))
    has_car  = bool(_RE_CARIBE.search(raw))
    has_pac  = bool(_RE_PACIFICO.search(raw))

    if ambas or (has_car and has_pac):
        return ["Caribe", "Pacífico"]
    if has_car:
        return ["Caribe"]
    if has_pac:
        return ["Pacífico"]
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ficha(
    habitat_raw: str = "",
    geographic_notes: str = "",
    species: str = "",
    enable_llm: bool = False,
) -> DistributionFicha:
    """
    Parse habitat text into a structured DistributionFicha.

    Args:
        habitat_raw:       Raw habitat sentence from Manual de Plantas.
        geographic_notes:  Geographic context string (post-semicolon part).
        species:           Species name (for labelling only).
        enable_llm:        Call LLM for unresolved tokens (default off).
    """
    combined_text = f"{habitat_raw} {geographic_notes}".strip()

    # ── Elevation ────────────────────────────────────────────────────────────
    elevation = _parse_elevation(combined_text)

    # ── Forest types ─────────────────────────────────────────────────────────
    forest_types = _parse_forest_types(habitat_raw)

    # ── Vertientes ───────────────────────────────────────────────────────────
    vertientes = _detect_vertientes(combined_text)

    # ── Gazetteer lookup ─────────────────────────────────────────────────────
    geo_matches = lookup(combined_text)

    def _to_entity_refs(hits: list[dict]) -> list[EntityRef]:
        seen: set[tuple] = set()
        refs = []
        for h in hits:
            key = (h["canonical_name"], h["hierarchy_level"])
            if key not in seen:
                seen.add(key)
                refs.append(EntityRef(
                    canonical_name  = h["canonical_name"],
                    hierarchy_level = h["hierarchy_level"],
                    source_span     = h.get("source_span", ""),
                ))
        return refs

    regions   = _to_entity_refs(
        geo_matches.get("cordillera", []) +
        geo_matches.get("llanura", []) +
        geo_matches.get("valle", []) +
        geo_matches.get("peninsula", []) +
        geo_matches.get("fila", []) +
        geo_matches.get("region_other", [])
    )
    parks     = _to_entity_refs(geo_matches.get("park", []))
    cantons   = _to_entity_refs(geo_matches.get("canton", []))
    districts = _to_entity_refs(geo_matches.get("district", []))

    # ── Vertientes from gazetteer (supplement regex detection) ───────────────
    gazetteer_verts = [
        e["canonical_name"] for e in geo_matches.get("vertiente", [])
        if e["canonical_name"] not in vertientes
    ]
    vertientes = list(dict.fromkeys(vertientes + gazetteer_verts))

    # ── Unresolved-token detection ────────────────────────────────────────────
    # Anything after the ';' that is not consumed by the gazetteer and is
    # a multi-word geographic phrase (heuristic: contains a capital after ';')
    unresolved: list[str] = []
    norm_combined = normalize_text(combined_text)

    # Simple heuristic: look for proper nouns in geographic_notes that didn't match
    if geographic_notes:
        # Split on common delimiters used in the Manual
        tokens = re.split(r"[,;]", geographic_notes)
        for tok in tokens:
            tok = tok.strip()
            if len(tok) < 4:
                continue
            norm_tok = normalize_text(tok)
            # Check if any gazetteer alias fully covers this token
            found = any(norm_tok in alias or alias in norm_tok
                        for alias, *_ in [])  # placeholder — full check below
            # Simple heuristic: if token starts with uppercase letter and
            # isn't already matched, add to unresolved
            if tok and tok[0].isupper() and norm_tok not in norm_combined[:10]:
                # Check whether gazetteer matched anything from this token
                token_matches = lookup(tok)
                if not any(token_matches.values()):
                    unresolved.append(tok)

    # ── Fuzzy fallback ────────────────────────────────────────────────────────
    fuzzy_conf: dict[str, float] = {}
    if unresolved:
        fuzzy_hits = fuzzy_lookup(unresolved, cutoff=88)
        for level, hits in fuzzy_hits.items():
            for h in hits:
                ref = EntityRef(h["canonical_name"], h["hierarchy_level"],
                                h.get("source_span", ""))
                score = h.get("fuzzy_score", 0) / 100
                fuzzy_conf[h["canonical_name"]] = score
                if level in ("cordillera", "llanura", "valle", "peninsula",
                             "fila", "region_other"):
                    if ref not in regions:
                        regions.append(ref)
                        unresolved = [u for u in unresolved
                                      if u != h.get("source_span")]
                elif level == "park":
                    if ref not in parks:
                        parks.append(ref)
                        unresolved = [u for u in unresolved
                                      if u != h.get("source_span")]
                elif level == "canton":
                    if ref not in cantons:
                        cantons.append(ref)
                        unresolved = [u for u in unresolved
                                      if u != h.get("source_span")]

    # ── LLM fallback (off by default) ────────────────────────────────────────
    if enable_llm and unresolved:
        try:
            llm_result = _llm_resolve(unresolved, combined_text)
            # llm_result is expected to be a dict like {"regions": [...], ...}
            for name in llm_result.get("regions", []):
                ref = EntityRef(name, "region_other", "llm")
                if ref not in regions:
                    regions.append(ref)
            unresolved = llm_result.get("still_unresolved", unresolved)
        except Exception as e:
            print(f"  [WARN] LLM fallback failed: {e}")

    # ── Confidence summary ────────────────────────────────────────────────────
    n_matched = len(regions) + len(parks) + len(cantons) + len(districts)
    confidence = {
        "overall": 1.0 if n_matched > 0 else 0.0,
        "geographic": min(1.0, n_matched / 3) if n_matched else 0.0,
        "elevation": 1.0 if elevation.has_data() else 0.0,
        **fuzzy_conf,
    }

    return DistributionFicha(
        species          = species,
        habitat_raw      = habitat_raw,
        vertientes       = vertientes,
        regions          = regions,
        parks            = parks,
        cantons          = cantons,
        districts        = districts,
        elevation        = elevation,
        forest_types     = forest_types,
        confidence       = confidence,
        unresolved_tokens= unresolved,
    )


def _llm_resolve(unresolved_tokens: list[str], context: str) -> dict:
    """
    LLM fallback for unresolved geographic tokens.
    Uses the project's existing OpenRouter client.
    Returns {"regions": [...], "still_unresolved": [...]}.
    """
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    try:
        import config as _cfg
        import requests, time, json

        prompt = (
            "You are a Costa Rica botanist. The following geographic tokens from a "
            "plant distribution description could not be identified automatically. "
            "For each token, identify which botanical region of Costa Rica it belongs to "
            "(e.g. Cordillera de Talamanca, Llanuras de Tortuguero, Península de Osa, etc.) "
            "or return null if unknown.\n\n"
            f"Context: {context[:400]}\n\n"
            f"Tokens: {unresolved_tokens}\n\n"
            'Respond ONLY with JSON: {"resolved": {"token": "region"}, "unresolved": [...]}'
        )

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {_cfg.OPENROUTER_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "meta-llama/llama-3.3-70b-instruct",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0},
            timeout=20,
        )
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(text)
        regions = list(parsed.get("resolved", {}).values())
        still   = parsed.get("unresolved", [])
        return {"regions": [r for r in regions if r], "still_unresolved": still}
    except Exception as e:
        print(f"  [WARN] _llm_resolve error: {e}")
        return {"regions": [], "still_unresolved": unresolved_tokens}
