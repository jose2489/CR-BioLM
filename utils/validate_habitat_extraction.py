"""
Validate habitat extraction logic on Vol. VII (high-quality OCR).
Extracts 200 species entries to inspect extraction quality.

Usage:
    python utils/validate_habitat_extraction.py [--sample N]
"""

import argparse
import os
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUT_BASE_DIR

PDF_PATH = r"C:\Users\Jose\Documents\Tesis\raw_data\Manual de Especies\ManualPlantasCostaRica_BHL_v7_ocr.pdf"
OUTPUT_CSV = os.path.join(OUTPUT_BASE_DIR, "validation_vol7_extraction.csv")

# Same regex patterns as extract_habitat_from_pdf.py
HABITAT_KEYWORDS_PRIMARY = (
    "Bosque", "Matorral", "Páramo", "Paramo", "Sabana", "Manglar",
    "Vegetación", "Vegetacion", "Pastizal", "Terrenos", "Orilla",
    "Pantano", "Borde", "Selva", "Humedal", "Ribera",
    "Acuática", "Acuatica", "Ruderal", "Potreros", "Páramos",
)
HABITAT_KEYWORDS_SECONDARY = ("Epifita", "Epífita", "Terrestre", "Rupicola",)

RE_GEO = re.compile(
    r'\b(?:vert\.|Pac\.|Carib\.|Cord\.|Cords\.|cuenca|CR\b|Nic\.|Pan\.|Guat\.|Méx\.|Mex\.|'
    r'Herr\.|Guan\.|Puntarenas|Limon|Alajuela|Cartago|Heredia|vertiente|talud)\b'
)

RE_ELEV_RANGE  = re.compile(
    r'(?:\(([—–\-]?\d+[—–\-]?)\)\s*)?'
    r'(\d+)\s*[—–\-]+\s*(\d+)'
    r'(?:\s*\(([—–\-]?\d+(?:[—–\-]\d+)?)\))?'
    r'\s*m\b'
)
RE_ELEV_SINGLE = re.compile(r'\b(\d{3,4})\s*m\b')

# ---------------------------------------------------------------------------
# Extract species entries from PDF
# ---------------------------------------------------------------------------
def load_pdf_text(pdf_path: str) -> str:
    """Load all text from PDF."""
    print(f"Loading PDF: {Path(pdf_path).name} ...", end=" ", flush=True)
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    text = "".join(page.get_text() for page in doc)
    doc.close()
    print(f"OK ({page_count} pages)")
    return text

def extract_all_species_entries(text: str, max_species: int = 200) -> list[dict]:
    """
    Extract first N species entries from text.
    Returns list of dicts with: species_name, block, page_context
    """
    # Pattern: capitalized word + lowercase word at line start (species header)
    species_pattern = re.compile(r'(?:^|\n)([A-Z][a-z]+\s+[a-z]+)\b')

    entries = []
    for match in species_pattern.finditer(text):
        if len(entries) >= max_species:
            break

        species_name = match.group(1)
        start = match.start(1)
        block = text[start: start + 3000]  # ~3000 chars

        # Only include if block has elevation or habitat keywords
        if RE_ELEV_RANGE.search(block) or any(kw in block for kw in HABITAT_KEYWORDS_PRIMARY):
            entries.append({
                "species_name": species_name,
                "block": block,
                "position": start,
            })

    return entries

def extract_habitat_regex(block: str) -> str | None:
    """Same logic as extract_habitat_from_pdf.py"""
    lines = block.split("\n")
    candidates = []

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

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]

def _collect_habitat_lines(lines: list[str], start_idx: int) -> str:
    """Collect continuation lines."""
    habitat_lines = [lines[start_idx].strip()]
    for j in range(start_idx + 1, min(start_idx + 6, len(lines))):
        next_line = lines[j].strip()
        if re.match(r'^\d{2,4}$', next_line):
            break
        if "Manual de Plantas" in next_line:
            break
        if not next_line:
            break
        if re.match(r'^[A-Z][a-z]+ [a-z]+\s+[A-Z]', next_line):
            break
        habitat_lines.append(next_line)
    return " ".join(habitat_lines)

def parse_elevation(habitat_raw: str) -> tuple[float | None, float | None]:
    """Extract elevation min and max."""
    m = RE_ELEV_RANGE.search(habitat_raw)
    if m:
        lo = float(m.group(2))
        hi = float(m.group(3))
        return lo, hi
    m = RE_ELEV_SINGLE.search(habitat_raw)
    if m:
        val = float(m.group(1))
        return val, val
    return None, None

def parse_habitat_type(habitat_raw: str) -> str:
    """Extract habitat type (first part before elevation)."""
    cut = re.split(r',|\d{3,4}\s*[—–\-]', habitat_raw)[0]
    return cut.strip()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Validate habitat extraction on Vol. VII")
    parser.add_argument("--sample", type=int, default=200, help="Number of species to extract")
    args = parser.parse_args()

    if not os.path.exists(PDF_PATH):
        print(f"[!] PDF not found: {PDF_PATH}")
        return

    # Load and extract
    text = load_pdf_text(PDF_PATH)
    entries = extract_all_species_entries(text, max_species=args.sample)

    print(f"\nProcessing {len(entries)} species entries...\n")

    results = []
    stats = {"success": 0, "failed": 0}

    for entry in entries:
        species = entry["species_name"]
        block = entry["block"]

        habitat_raw = extract_habitat_regex(block)

        if habitat_raw:
            stats["success"] += 1
            elev_min, elev_max = parse_elevation(habitat_raw)
            habitat_type = parse_habitat_type(habitat_raw)
            status = "✓"
        else:
            stats["failed"] += 1
            elev_min, elev_max = None, None
            habitat_type = ""
            status = "✗"

        results.append({
            "species": species,
            "habitat_raw": habitat_raw or "",
            "habitat_type": habitat_type,
            "elevation_min_m": elev_min,
            "elevation_max_m": elev_max,
            "extraction_success": habitat_raw is not None,
        })

        print(f"  {status} {species:<45s} | elev: {elev_min or '—':>6} – {elev_max or '—':<6} m")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\n{'─'*80}")
    print(f"  Total processed : {len(entries)}")
    print(f"  Successfully extracted : {stats['success']}")
    print(f"  Failed : {stats['failed']}")
    print(f"  Success rate : {100*stats['success']/len(entries):.1f}%")
    print(f"  Output saved to : {os.path.abspath(OUTPUT_CSV)}")

if __name__ == "__main__":
    main()
