# experiment/validators/domain_validators.py
#
# Programmatic domain validators for CR-BioLM evaluation.
#
# D1 — Taxonomic accuracy (binary): species name against GBIF backbone via pygbif.
# D3 — Elevation plausibility (0–2): elevations mentioned fall within
#       [min - 200 m, max + 200 m] of GBIF/Manual range.
#
# These run cheaply before calling the LLM judge ensemble.
# Results are merged into the eval_*.json output.

import re


# ── D1 — Taxonomic accuracy ───────────────────────────────────────────────────

def validate_taxonomy(species_name: str) -> dict:
    """
    Checks species_name against the GBIF backbone taxonomy.

    Returns:
        {
            "taxonomy_valid": bool,
            "taxonomy_status": str,   # "ACCEPTED" | "SYNONYM" | "DOUBTFUL" | "NOT_FOUND" | "ERROR"
            "taxonomy_match": str,    # matched name from GBIF (may differ from input)
            "taxonomy_confidence": int | None,
        }
    """
    try:
        from pygbif import species as gbif_species
        result = gbif_species.name_backbone(species_name, strict=False, verbose=False)

        # pygbif v2 structure: result["usage"] and result["diagnostics"]
        usage = result.get("usage") or {}
        diagnostics = result.get("diagnostics") or {}
        status     = usage.get("status", "NOT_FOUND")
        confidence = diagnostics.get("confidence")
        matched_name = usage.get("canonicalName") or usage.get("name") or species_name

        valid = status in ("ACCEPTED", "SYNONYM") and (confidence is None or confidence >= 80)

        return {
            "taxonomy_valid":      valid,
            "taxonomy_status":     status,
            "taxonomy_match":      matched_name,
            "taxonomy_confidence": confidence,
        }

    except ImportError:
        print("[D1] pygbif not installed — taxonomy validation skipped (assuming valid).")
        return {
            "taxonomy_valid":      True,
            "taxonomy_status":     "SKIPPED_NO_PYGBIF",
            "taxonomy_match":      species_name,
            "taxonomy_confidence": None,
        }
    except Exception as e:
        print(f"[D1] GBIF lookup failed for '{species_name}': {e}")
        return {
            "taxonomy_valid":      True,  # fail-open: don't penalize on API errors
            "taxonomy_status":     "ERROR",
            "taxonomy_match":      species_name,
            "taxonomy_confidence": None,
        }


# ── D3 — Elevation plausibility ───────────────────────────────────────────────

_ALT_PATTERNS = [
    r"(\d[\d\s]*(?:,\d+)?)\s*(?:–|-|a)\s*(\d[\d\s]*(?:,\d+)?)\s*m(?:\s*\.?\s*n\s*\.?\s*m\s*\.?)?",
    r"(\d{3,4})\s*(?:–|-)\s*(\d{3,4})\s*m",
    r"entre\s+(\d{3,4})\s+y\s+(\d{3,4})\s*m",
    r"(?:desde|de)\s+(\d{3,4})\s+(?:hasta|a)\s+(\d{3,4})\s*m",
    r"(\d{3,4})\s*m\s+(?:y|a|hasta)\s+(\d{3,4})\s*m",
]

_SINGLE_ALT_PATTERN = re.compile(
    r"(?:a(?:proximadamente)?|hasta|sobre|bajo|mínimo|máximo|cerca de)?\s*(\d{3,4})\s*m"
    r"(?:\s*\.?\s*n\s*\.?\s*m)?",
    re.IGNORECASE,
)


def _extract_elevations(text: str) -> list[int]:
    """Extract all elevation values (metres) mentioned in text."""
    found = []
    for pat in _ALT_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            try:
                lo = int(m.group(1).replace(" ", "").replace(",", ""))
                hi = int(m.group(2).replace(" ", "").replace(",", ""))
                found.extend([lo, hi])
            except (ValueError, IndexError):
                pass

    for m in _SINGLE_ALT_PATTERN.finditer(text):
        try:
            val = int(m.group(1).replace(" ", ""))
            if 0 <= val <= 4000:
                found.append(val)
        except ValueError:
            pass

    return list(set(found))


def validate_elevation(
    respuesta: str,
    alt_min_gbif: float | None,
    alt_max_gbif: float | None,
    alt_min_manual: float | None,
    alt_max_manual: float | None,
    tolerance_m: float = 200.0,
) -> dict:
    """
    D3 — Elevation plausibility check.

    Uses the wider of GBIF and Manual ranges as the accepted window,
    then extends by tolerance_m on each side.

    Returns:
        {
            "D3_elevation_plausibility": 0 | 1 | 2 | "N/A",
            "D3_elevations_found":  [list of ints],
            "D3_range_used":        [lo, hi] or None,
            "D3_out_of_range":      [list of outlier ints],
            "D3_rationale":         str,
        }
    """
    # Build reference range from available data
    lows  = [v for v in [alt_min_gbif, alt_min_manual] if v is not None]
    highs = [v for v in [alt_max_gbif, alt_max_manual] if v is not None]

    if not lows or not highs:
        return {
            "D3_elevation_plausibility": "N/A",
            "D3_elevations_found":       [],
            "D3_range_used":             None,
            "D3_out_of_range":           [],
            "D3_rationale":              "No reference elevation data available.",
        }

    ref_lo = min(lows) - tolerance_m
    ref_hi = max(highs) + tolerance_m

    elevations = _extract_elevations(respuesta)

    if not elevations:
        return {
            "D3_elevation_plausibility": 0,
            "D3_elevations_found":       [],
            "D3_range_used":             [ref_lo, ref_hi],
            "D3_out_of_range":           [],
            "D3_rationale":              "No elevation values found in response.",
        }

    out_of_range = [e for e in elevations if not (ref_lo <= e <= ref_hi)]

    if not out_of_range:
        score = 2
        rationale = f"All elevations {elevations} within reference window [{ref_lo:.0f}, {ref_hi:.0f}] m."
    elif len(out_of_range) < len(elevations):
        score = 1
        rationale = f"Some elevations out of range: {out_of_range} (window [{ref_lo:.0f}, {ref_hi:.0f}] m)."
    else:
        score = 0
        rationale = f"All elevations out of range: {out_of_range} (window [{ref_lo:.0f}, {ref_hi:.0f}] m)."

    return {
        "D3_elevation_plausibility": score,
        "D3_elevations_found":       elevations,
        "D3_range_used":             [ref_lo, ref_hi],
        "D3_out_of_range":           out_of_range,
        "D3_rationale":              rationale,
    }


# ── Combined runner ───────────────────────────────────────────────────────────

def run_domain_validators(
    species_name: str,
    respuesta: str,
    alt_min_gbif: float | None   = None,
    alt_max_gbif: float | None   = None,
    alt_min_manual: float | None = None,
    alt_max_manual: float | None = None,
) -> dict:
    """
    Runs D1 and D3 and returns a merged result dict.
    taxonomy_valid from D1 is used by EnsembleJudge for the 0.1 cap.
    """
    d1 = validate_taxonomy(species_name)
    d3 = validate_elevation(respuesta, alt_min_gbif, alt_max_gbif, alt_min_manual, alt_max_manual)
    return {**d1, **d3}
