"""
Extract habitat/location descriptions from Manual de Plantas de Costa Rica PDFs
for each species listed in outputs/picked_species.csv.

Strategy:
  1. PyMuPDF extracts embedded OCR text from BHL PDFs (cached per volume).
  2. Regex locates the species entry and isolates the habitat sentence.
  3. OpenRouter LLM is used as fallback when regex fails.

Output: outputs/picked_species_enhanced.csv

Usage:
    python utils/extract_habitat_from_pdf.py [--no-llm] [--input PATH] [--output PATH]
"""

import argparse
import os
import re
import sys
import time
from difflib import SequenceMatcher

import fitz  # PyMuPDF
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUT_BASE_DIR, OPENROUTER_API_KEY

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PDF_DIR   = r"C:\Users\Jose\Documents\Tesis\raw_data\Manual de Especies"
INPUT_CSV = os.path.join(OUTPUT_BASE_DIR, "picked_species.csv")
OUTPUT_CSV = os.path.join(OUTPUT_BASE_DIR, "picked_species_enhanced.csv")

VOLUME_TO_PDF = {
    "Vol. II":   "ManualPlantasCostaRica_BHL_v2.pdf",
    "Vol. III":  "ManualPlantasCostaRica_BHL_v3.pdf",
    "Vol. VI":   "ManualPlantasCostaRica_BHL_v6.pdf",
    "Vol. VIII": "ManualPlantasCostaRica_BHL_v8.pdf",
}

# Primary habitat keywords that reliably open the habitat sentence (NOT morphological)
HABITAT_KEYWORDS_PRIMARY = (
    "Bosque", "Matorral", "Páramo", "Paramo", "Sabana", "Manglar",
    "Vegetación", "Vegetacion", "Pastizal", "Terrenos", "Orilla",
    "Pantano", "Borde", "Selva", "Humedal", "Ribera",
    "Acuática", "Acuatica", "Ruderal", "Potreros", "Páramos",
)
# Secondary keywords — only valid if line also has geo/elevation context
HABITAT_KEYWORDS_SECONDARY = ("Epifita", "Epífita", "Terrestre", "Rupicola",)

# Geographic terms that appear in habitat/distribution sentences
RE_GEO = re.compile(
    r'\b(?:vert\.|Pac\.|Carib\.|Cord\.|Cords\.|cuenca|CR\b|Nic\.|Pan\.|Guat\.|Méx\.|Mex\.|'
    r'Herr\.|Guan\.|Puntarenas|Limon|Alajuela|Cartago|Heredia|vertiente|talud)\b'
)

# Regex: elevation range — requires values likely to be metres of altitude
# (avoid matching e.g. "tallos 1—5 m" by also requiring geographic context)
# Supports optional outlier parentheticals before and/or after the main range:
#   "0–700 (—1000) m"          → main 0–700, upper outlier 1000
#   "(0–)500–1850(—2500) m"    → main 500–1850, lower outlier 0, upper outlier 2500
#   "200–800 (100–900) m"      → main 200–800, full outlier range 100–900
RE_ELEV_RANGE  = re.compile(
    r'(?:\(([—–\-]?\d+[—–\-]?)\)\s*)?'  # group 1: optional leading outlier e.g. (0–) or (100–200)
    r'(\d+)\s*[—–\-]+\s*(\d+)'           # groups 2,3: main range lo–hi
    r'(?:\s*\(([—–\-]?\d+(?:[—–\-]\d+)?)\))?'  # group 4: optional trailing outlier (—2500)
    r'\s*m\b'
)
RE_ELEV_SINGLE = re.compile(r'\b(\d{3,4})\s*m\b')
# Regex: country code block like "CR", "CR—Nic.", "CR—Pan—Col."
RE_COUNTRY     = re.compile(r'\bCR\b')

# ---------------------------------------------------------------------------
# PDF loading (cached)
# ---------------------------------------------------------------------------
_pdf_cache: dict[str, str] = {}

def load_pdf_text(volume: str) -> str:
    if volume not in _pdf_cache:
        pdf_path = os.path.join(PDF_DIR, VOLUME_TO_PDF[volume])
        print(f"  Loading PDF: {os.path.basename(pdf_path)} ...", end=" ", flush=True)
        doc = fitz.open(pdf_path)
        _pdf_cache[volume] = "".join(page.get_text() for page in doc)
        doc.close()
        print("done")
    return _pdf_cache[volume]


# ---------------------------------------------------------------------------
# Species entry extraction
# ---------------------------------------------------------------------------
def find_species_block(text: str, species: str, block_chars: int = 3000) -> tuple[str, str]:
    """
    Return (block, method) where block is ~block_chars chars starting from the
    actual species entry header (line-start match). Falls back to fuzzy search.
    method is 'exact' or 'fuzzy' or 'not_found'.
    """
    parts = species.split()
    genus   = parts[0]
    epithet = parts[1] if len(parts) > 1 else ""

    # Priority 1: species name at the start of a line (actual entry header)
    entry_pattern = re.compile(
        r'(?:^|\n)(' + re.escape(genus) + r'\s+' + re.escape(epithet) + r'\b)',
        re.IGNORECASE
    )
    for m in entry_pattern.finditer(text):
        start = m.start(1)
        block = text[start: start + block_chars]
        # Validate: block should contain an elevation or habitat keyword
        if RE_ELEV_RANGE.search(block) or any(kw in block for kw in HABITAT_KEYWORDS_PRIMARY):
            return block, "exact"

    # Priority 2: any occurrence (may be in a key/synonym list — still try)
    idx = text.find(species)
    if idx != -1:
        block = text[idx: idx + block_chars]
        if RE_ELEV_RANGE.search(block) or any(kw in block for kw in HABITAT_KEYWORDS_PRIMARY):
            return block, "exact"

    # Priority 3: fuzzy — OCR may have mangled genus or epithet
    fuzzy_pattern = re.compile(
        r'(?:^|\n)' + re.escape(genus[:5]) + r'\S*\s+' + re.escape(epithet[:5]) + r'\S*',
        re.IGNORECASE
    )
    m = fuzzy_pattern.search(text)
    if m:
        start = m.start()
        return text[start: start + block_chars], "fuzzy"

    return "", "not_found"


def _collect_habitat_lines(lines: list[str], start_idx: int) -> str:
    """Collect continuation lines from start_idx until entry/page boundary."""
    habitat_lines = [lines[start_idx].strip()]
    for j in range(start_idx + 1, min(start_idx + 6, len(lines))):
        next_line = lines[j].strip()
        if re.match(r'^\d{2,4}$', next_line):      # standalone page number
            break
        if "Manual de Plantas" in next_line:
            break
        if not next_line:
            break
        # Stop when we hit what looks like a new species/genus entry
        if re.match(r'^[A-Z][a-z]+ [a-z]+\s+[A-Z]', next_line):
            break
        habitat_lines.append(next_line)
    return " ".join(habitat_lines)


def extract_habitat_regex(block: str) -> str | None:
    """
    Find the habitat sentence within a species entry block.

    Scoring priority:
      1. Line starts with a primary habitat keyword (Bosque, Matorral…) AND has geo/elev context
      2. Line starts with a primary habitat keyword (no extra context needed — reliable)
      3. Line has elevation range AND geographic term (catches non-keyword habitat lines)
      4. Secondary keyword (Epifita…) only if line also has elevation AND geo context
    Returns the best candidate or None.
    """
    lines = block.split("\n")
    candidates: list[tuple[int, int, str]] = []  # (priority, line_idx, text)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < 10:
            continue

        has_primary_kw  = any(stripped.startswith(kw) for kw in HABITAT_KEYWORDS_PRIMARY)
        has_secondary_kw = any(stripped.startswith(kw) for kw in HABITAT_KEYWORDS_SECONDARY)
        has_elevation   = bool(RE_ELEV_RANGE.search(stripped) or RE_ELEV_SINGLE.search(stripped))
        has_geo         = bool(RE_GEO.search(stripped))

        if has_primary_kw and (has_elevation or has_geo):
            candidates.append((1, i, _collect_habitat_lines(lines, i)))
        elif has_primary_kw:
            candidates.append((2, i, _collect_habitat_lines(lines, i)))
        elif has_elevation and has_geo:
            candidates.append((3, i, _collect_habitat_lines(lines, i)))
        elif has_secondary_kw and has_elevation and has_geo:
            candidates.append((4, i, _collect_habitat_lines(lines, i)))

    if not candidates:
        return None

    # Return highest-priority (lowest number), earliest occurrence
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


# ---------------------------------------------------------------------------
# OpenRouter LLM fallback
# ---------------------------------------------------------------------------
_llm_client = None

def get_llm_client():
    global _llm_client
    if _llm_client is None:
        try:
            from openai import OpenAI
            _llm_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
        except Exception as e:
            print(f"  [!] Could not initialise OpenRouter client: {e}")
    return _llm_client


def extract_habitat_llm(species: str, block: str) -> str | None:
    client = get_llm_client()
    if client is None:
        return None

    prompt = (
        f"Below is a botanical entry for '{species}' from the Manual de Plantas de Costa Rica.\n"
        "Extract ONLY the habitat/location sentence — the one that describes the vegetation type, "
        "elevation (in metres), and geographic distribution within Costa Rica. "
        "Return just that sentence, nothing else.\n\n"
        f"TEXT:\n{block[:1200]}"
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-3.2-3b-instruct:free",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            result = response.choices[0].message.content.strip()
            time.sleep(8)  # respect 8 RPM free-tier limit
            return result if result else None
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 20 * (attempt + 1)
                print(f"  [rate-limit] waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"  [!] LLM error for {species}: {e}")
                return None
    return None


# ---------------------------------------------------------------------------
# Structured field parsing
# ---------------------------------------------------------------------------
def _parse_outlier_token(token: str, lo: float, hi: float) -> tuple[float | None, float | None]:
    """
    Parse a single outlier parenthetical string (already stripped of parens).
    Returns (outlier_min, outlier_max).

    Examples:
      "0–"    → (0, None)     leading lower floor with trailing dash
      "—2500" → (None, 2500)  upper cap with leading dash
      "100"   → determined by comparing to lo/hi
      "100–900" → (100, 900)  full range
    """
    token = token.strip()
    # Has both a lower and upper component: "100–900"
    parts = re.split(r'[—–\-]', token.strip("—–-"))
    parts = [p for p in parts if p]  # remove empty strings from leading/trailing dashes

    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    elif len(parts) == 1:
        val = float(parts[0])
        if val > hi:
            return None, val   # upper cap
        elif val < lo:
            return val, None   # lower floor
        else:
            return None, val   # ambiguous → upper
    # Token was only dashes (e.g. "0–" with trailing dash stripped to "0")
    # — handled by single-digit extraction above
    return None, None


def parse_elevation(habitat_raw: str) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Returns (elev_min, elev_max, outlier_min, outlier_max).

    Handles patterns like:
      "0–700 m"                  → (0, 700, None, None)
      "0–700 (—1000) m"          → (0, 700, None, 1000)    upper cap
      "(0–)500–1850(—2500) m"    → (500, 1850, 0, 2500)    lower floor + upper cap
      "200–800 (100–900) m"      → (200, 800, 100, 900)    full outlier range
    """
    m = RE_ELEV_RANGE.search(habitat_raw)
    if m:
        # Groups: 1=leading outlier, 2=lo, 3=hi, 4=trailing outlier
        raw_lead  = m.group(1)  # e.g. "0–" or None
        lo        = float(m.group(2))
        hi        = float(m.group(3))
        raw_trail = m.group(4)  # e.g. "—2500" or "100–900" or None

        outlier_min, outlier_max = None, None

        # Parse leading outlier e.g. "(0–)" — strip trailing dash
        if raw_lead:
            token = raw_lead.rstrip("—–-").strip()
            if token:
                omin, omax = _parse_outlier_token(token, lo, hi)
                if omin is not None: outlier_min = omin
                if omax is not None: outlier_max = omax

        # Parse trailing outlier e.g. "(—2500)"
        if raw_trail:
            omin, omax = _parse_outlier_token(raw_trail, lo, hi)
            if omin is not None: outlier_min = omin
            if omax is not None: outlier_max = omax

        return lo, hi, outlier_min, outlier_max
    m = RE_ELEV_SINGLE.search(habitat_raw)
    if m:
        val = float(m.group(1))
        return val, val, None, None
    return None, None, None, None


def parse_habitat_type(habitat_raw: str) -> str:
    # Everything before the first comma or elevation number
    cut = re.split(r',|\d{3,4}\s*[—–\-]', habitat_raw)[0]
    return cut.strip()


def parse_geographic_notes(habitat_raw: str) -> str:
    # Text between the elevation block and the country code / specimen ref
    m_elev = RE_ELEV_RANGE.search(habitat_raw) or RE_ELEV_SINGLE.search(habitat_raw)
    if not m_elev:
        return ""
    after_elev = habitat_raw[m_elev.end():].strip().lstrip(";").strip()
    # Cut off at specimen reference (opening parenthesis with collector names)
    cut = re.split(r'\s*\(', after_elev)[0]
    # Cut off at FR. / Fr. (phenology)
    cut = re.split(r'\bFr\.\b|\bFl\.\b', cut)[0]
    return cut.strip().rstrip(";,").strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract habitat descriptions from Manual de Plantas PDFs.")
    parser.add_argument("--input",   default=INPUT_CSV,  help="picked_species.csv path")
    parser.add_argument("--output",  default=OUTPUT_CSV, help="Output enhanced CSV path")
    parser.add_argument("--no-llm",  action="store_true", help="Disable LLM fallback")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    results = []

    stats = {"exact": 0, "fuzzy": 0, "llm": 0, "failed": 0}

    for _, row in df.iterrows():
        species = row["species"]
        volume  = row["volume"]

        text  = load_pdf_text(volume)
        block, find_method = find_species_block(text, species)

        habitat_raw       = None
        extraction_method = "failed"

        if block:
            habitat_raw = extract_habitat_regex(block)
            if habitat_raw:
                extraction_method = find_method  # exact or fuzzy
            elif not args.no_llm:
                print(f"  [LLM] {species}")
                habitat_raw = extract_habitat_llm(species, block)
                extraction_method = "llm" if habitat_raw else "failed"
        elif not args.no_llm:
            print(f"  [LLM-notfound] {species}")
            extraction_method = "failed"

        stats[extraction_method if extraction_method in stats else "failed"] += 1
        elev_min, elev_max, outlier_min, outlier_max = parse_elevation(habitat_raw or "")

        results.append({
            **row.to_dict(),
            "habitat_raw":          habitat_raw or "",
            "habitat_type":         parse_habitat_type(habitat_raw) if habitat_raw else "",
            "elevation_min_m":      elev_min,
            "elevation_max_m":      elev_max,
            "elev_outlier_min_m":   outlier_min,
            "elev_outlier_max_m":   outlier_max,
            "geographic_notes":     parse_geographic_notes(habitat_raw) if habitat_raw else "",
            "extraction_method":    extraction_method,
        })

        status = "✓" if habitat_raw else "✗"
        print(f"  {status} [{volume}] {species:<45s} → {extraction_method}")

    out = pd.DataFrame(results)
    out.to_csv(args.output, index=False, encoding="utf-8")

    total = len(df)
    succeeded = total - stats["failed"]
    print(f"\n{'─'*60}")
    print(f"  Total species   : {total}")
    print(f"  Extracted       : {succeeded}  (exact={stats['exact']}, fuzzy={stats['fuzzy']}, llm={stats['llm']})")
    print(f"  Failed          : {stats['failed']}")
    print(f"  Output saved to : {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
