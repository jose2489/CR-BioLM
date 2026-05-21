"""
gazetteer.py — Entity gazetteer: text aliases → canonical geographic entities.

Replaces the 137-regex translation table with a deterministic longest-match
lookup against a CSV of known entities and their aliases. Designed to be
ordering-insensitive: no "more specific patterns must precede broader ones".

Lookup pipeline:
  1. normalize_text()  — NFKD → ASCII → lowercase → apply OCR corrections
  2. Longest-match scan over the alias index
  3. rapidfuzz fuzzy fallback for remaining unmatched fragments
  4. LLM fallback hook (optional, via parser._llm_resolve)
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rapidfuzz import process as fz_process

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_GAZETTEER_DIR  = Path(__file__).parent.parent.parent / "data_raw" / "gazetteer"
ENTITIES_CSV    = _GAZETTEER_DIR / "entities.csv"
CORRECTIONS_CSV = _GAZETTEER_DIR / "ocr_corrections.csv"


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_corrections() -> list[tuple[str, str]]:
    """Return (raw, corrected) pairs sorted longest-raw-first."""
    if not CORRECTIONS_CSV.exists():
        return []
    df = pd.read_csv(CORRECTIONS_CSV, comment="#")
    pairs = [(str(r["raw"]).strip(), str(r["corrected"]).strip())
             for _, r in df.iterrows()
             if str(r.get("raw", "")).strip()]
    # Apply longest substitution first to avoid partial clobbers
    pairs.sort(key=lambda x: -len(x[0]))
    return pairs


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.encode("ascii", "ignore").decode("ascii")


def normalize_text(text: str) -> str:
    """
    Full normalisation pipeline:
      NFKD → strip accents → lowercase → collapse whitespace →
      line-break hyphens → OCR corrections.

    The same function is used on both the input text and the alias strings
    in the CSV so comparisons are always apples-to-apples.
    """
    t = _strip_accents(str(text))
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    # Re-join words split by OCR line-break hyphens: "tortu- guero" → "tortuguero"
    t = re.sub(r"([a-z])- ([a-z])", r"\1\2", t)
    for raw, corrected in _load_corrections():
        t = t.replace(raw, corrected)
    return t


# ---------------------------------------------------------------------------
# Gazetteer index
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_entities() -> pd.DataFrame:
    if not ENTITIES_CSV.exists():
        raise FileNotFoundError(f"Gazetteer CSV not found: {ENTITIES_CSV}")
    df = pd.read_csv(ENTITIES_CSV, dtype=str).fillna("")
    return df


def _build_alias_index() -> list[tuple[str, str, str, str, str, str]]:
    """
    Build a flat list of (normalized_alias, canonical_name, hierarchy_level,
    target_shapefile, target_attribute, target_value) sorted longest-alias-first.

    Longest-first ensures "toda la vert. carib" beats "vert. carib" when both
    appear as substrings of the same text.
    """
    df = _load_entities()
    entries: list[tuple[str, str, str, str, str, str]] = []
    for _, row in df.iterrows():
        canonical    = row["canonical_name"].strip()
        level        = row["hierarchy_level"].strip()
        shapefile    = row["target_shapefile"].strip()
        attribute    = row["target_attribute"].strip()
        value        = row["target_value"].strip()
        raw_aliases  = row["aliases"].strip()
        if not raw_aliases:
            continue
        for alias in raw_aliases.split("|"):
            norm = normalize_text(alias.strip())
            if norm:
                entries.append((norm, canonical, level, shapefile, attribute, value))
    # Sort by alias length descending — longest match wins
    entries.sort(key=lambda x: -len(x[0]))
    return entries


# Cache the built index (called once per process)
@lru_cache(maxsize=None)
def _alias_index() -> list[tuple[str, str, str, str, str, str]]:
    return _build_alias_index()


# ---------------------------------------------------------------------------
# Public lookup
# ---------------------------------------------------------------------------

def lookup(text: str) -> dict[str, list[dict]]:
    """
    Scan normalized text for all known entity aliases.

    Returns a dict keyed by hierarchy_level, each containing a list of
    matched entity dicts:
      {canonical_name, hierarchy_level, target_shapefile,
       target_attribute, target_value, source_span}

    Longest-match rule is applied *per hierarchy level*: once a span is consumed
    within a level, a shorter alias at the same level cannot claim it. But a
    different level (e.g. canton vs region) CAN match the same span — this lets
    "Puriscal" match both region "Puriscal - Los Santos" and canton "Puriscal"
    from the same text fragment.
    """
    norm = normalize_text(text)
    index = _alias_index()

    # consumed positions tracked per (hierarchy_level) separately
    consumed_by_level: dict[str, set[int]] = {}
    # also track (canonical_name, hierarchy_level) to avoid duplicate entries
    seen_entities: set[tuple[str, str]] = set()

    matched: list[dict] = []

    for alias_norm, canonical, level, shapefile, attr, value in index:
        consumed = consumed_by_level.setdefault(level, set())
        start = 0
        while True:
            pos = norm.find(alias_norm, start)
            if pos == -1:
                break
            end = pos + len(alias_norm)
            span_positions = set(range(pos, end))
            entity_key = (canonical, level)
            if not (span_positions & consumed) and entity_key not in seen_entities:
                matched.append({
                    "canonical_name":   canonical,
                    "hierarchy_level":  level,
                    "target_shapefile": shapefile,
                    "target_attribute": attr,
                    "target_value":     value,
                    "source_span":      text[pos:end] if pos < len(text) else alias_norm,
                })
                consumed |= span_positions
                seen_entities.add(entity_key)
            start = end

    # Group by hierarchy_level
    result: dict[str, list[dict]] = {}
    for m in matched:
        result.setdefault(m["hierarchy_level"], []).append(m)

    return result


def lookup_parks(text: str, pa_names: list[str]) -> list[str]:
    """
    Return subset of `pa_names` whose normalized name appears in text.
    Handles the "P.N. Carara" → "Parque Nacional Carara" expansion via
    OCR corrections (which expand "parque nacional" from "p.n.").
    Short names (<5 chars) require word boundaries.
    """
    norm = normalize_text(text)
    matched = []
    for name in pa_names:
        norm_name = normalize_text(name)
        if len(norm_name) < 5:
            pat = r"\b" + re.escape(norm_name) + r"\b"
            if re.search(pat, norm):
                matched.append(name)
        else:
            if norm_name in norm:
                matched.append(name)
    return matched


def fuzzy_lookup(tokens: list[str], cutoff: int = 88) -> dict[str, list[dict]]:
    """
    Try to fuzzy-match a list of unresolved text tokens against all known
    canonical names. Returns the same structure as lookup().

    Only tokens with score >= cutoff are accepted.
    """
    df = _load_entities()
    candidates = df["canonical_name"].dropna().unique().tolist()
    result: dict[str, list[dict]] = {}

    for token in tokens:
        norm_token = normalize_text(token)
        if len(norm_token) < 4:
            continue
        match = fz_process.extractOne(
            norm_token,
            [normalize_text(c) for c in candidates],
            score_cutoff=cutoff,
        )
        if match is None:
            continue
        matched_norm, score, idx = match
        row = df[df["canonical_name"] == candidates[idx]].iloc[0]
        entry = {
            "canonical_name":   row["canonical_name"],
            "hierarchy_level":  row["hierarchy_level"],
            "target_shapefile": row["target_shapefile"],
            "target_attribute": row["target_attribute"],
            "target_value":     row["target_value"],
            "source_span":      token,
            "fuzzy_score":      score,
        }
        result.setdefault(row["hierarchy_level"], []).append(entry)

    return result
