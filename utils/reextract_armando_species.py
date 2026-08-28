"""
Re-extract habitat data for Armando's species list directly from Manual de Plantas PDFs.
Targets only species with failed/llm/suspicious extractions.
Outputs a corrected CSV merging into picked_species_enhanced.csv.

Usage:
    python utils/reextract_armando_species.py [--dump-blocks] [--output PATH]
"""

import argparse
import os
import re
import sys

import fitz
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUT_BASE_DIR
from utils.extract_habitat_from_pdf import (
    HABITAT_KEYWORDS_PRIMARY,
    RE_ELEV_RANGE,
    RE_ELEV_SINGLE,
    RE_GEO,
    extract_habitat_regex,
    parse_elevation,
    parse_geographic_notes,
    parse_habitat_type,
)

PDF_DIR = r"C:\Users\Jose\Documents\Tesis\raw_data\Manual de Especies"

VOLUME_TO_PDF = {
    "Vol. II":   "ManualPlantasCostaRica_BHL_v2_Gimnospermas y Monocotiledoneas.pdf",
    "Vol. III":  "ManualPlantasCostaRica_BHL_v3_Monocotileidoneas.pdf",
    "Vol. VI":   "ManualPlantasCostaRica_BHL_v6_Dicotiledoneas.pdf",
    "Vol. VIII": "ManualPlantasCostaRica_BHL_v8.pdf",
}

# Species to re-extract (failed or LLM-hallucinated)
TARGETS = [
    "Albizia niopoides", "Anacardium excelsum", "Anthodiscus chocoensis",
    "Astronium graveolens", "Balizia elegans", "Brosimum utile",
    "Campnosperma panamensis", "Caryocar costaricense", "Cedrela odorata",
    "Ceiba pentandra", "Chaunochiton kappleri", "Chiangiodendron mexicanum",
    "Copaifera aromatica", "Cordia gerascanthus", "Cipura cubensis",
    "Crescentia cujete", "Dipteryx oleifera", "Hymenolobium mesoamericanum",
    "Lecythis ampla", "Macrohasseltia macroterantha", "Nelsonia canescens",
    "Oreamunnea pterocarpa", "Peltogyne purpurea", "Spirotheca rosea",
    "Tachigali versicolor",
]

_pdf_cache: dict[str, str] = {}


def load_vol(volume: str) -> str:
    if volume not in _pdf_cache:
        path = os.path.join(PDF_DIR, VOLUME_TO_PDF[volume])
        print(f"  Loading {os.path.basename(path)} ...", end=" ", flush=True)
        doc = fitz.open(path)
        _pdf_cache[volume] = "".join(page.get_text() for page in doc)
        doc.close()
        print("done")
    return _pdf_cache[volume]


def find_in_volume(text: str, genus: str, epithet: str, block_chars: int = 3000) -> tuple[str, str]:
    """Return (block, method). Tries exact line-start match first, then fuzzy."""
    pat = re.compile(
        r"(?:^|\n)(" + re.escape(genus) + r"\s+" + re.escape(epithet) + r"\b)",
        re.IGNORECASE,
    )
    for m in pat.finditer(text):
        start = m.start(1)
        block = text[start: start + block_chars]
        if RE_ELEV_RANGE.search(block) or any(kw in block for kw in HABITAT_KEYWORDS_PRIMARY):
            return block, "exact"

    # any occurrence fallback
    idx = text.find(f"{genus} {epithet}")
    if idx != -1:
        block = text[idx: idx + block_chars]
        if RE_ELEV_RANGE.search(block) or any(kw in block for kw in HABITAT_KEYWORDS_PRIMARY):
            return block, "exact"

    # fuzzy — truncated genus/epithet
    fuzzy = re.compile(
        r"(?:^|\n)" + re.escape(genus[:5]) + r"\S*\s+" + re.escape(epithet[:5]) + r"\S*",
        re.IGNORECASE,
    )
    m = fuzzy.search(text)
    if m:
        return text[m.start(): m.start() + block_chars], "fuzzy"

    return "", "not_found"


def search_all_volumes(species: str) -> tuple[str, str, str]:
    """Search all volumes, return (block, method, volume)."""
    genus, epithet = species.split()[0], species.split()[1]
    # Search clean volumes first (better OCR)
    order = ["Vol. VI", "Vol. VIII", "Vol. II", "Vol. III"]
    for vol in order:
        text = load_vol(vol)
        block, method = find_in_volume(text, genus, epithet)
        if method != "not_found":
            return block, method, vol
    return "", "not_found", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-blocks", action="store_true", help="Print raw text blocks for manual review")
    parser.add_argument("--output", default=os.path.join(OUTPUT_BASE_DIR, "picked_species_enhanced.csv"))
    args = parser.parse_args()

    enhanced_path = os.path.join(OUTPUT_BASE_DIR, "picked_species_enhanced.csv")
    df = pd.read_csv(enhanced_path)

    updates = []
    stats = {"exact": 0, "fuzzy": 0, "failed": 0}

    for species in TARGETS:
        block, method, vol = search_all_volumes(species)

        habitat_raw = None
        extraction_method = "failed"

        if block:
            if args.dump_blocks:
                print(f"\n{'='*70}")
                print(f"BLOCK: {species} [{vol}] method={method}")
                print(block[:600])

            habitat_raw = extract_habitat_regex(block)
            if habitat_raw:
                extraction_method = method
            else:
                # Try broader search: any line with elevation in the block
                for line in block.split("\n"):
                    if RE_ELEV_RANGE.search(line) or RE_ELEV_SINGLE.search(line):
                        if len(line.strip()) > 15:
                            habitat_raw = line.strip()
                            extraction_method = method + "_elev"
                            break

        stats[extraction_method.split("_")[0] if extraction_method != "failed" else "failed"] += 1

        elev_min, elev_max, out_min, out_max = parse_elevation(habitat_raw or "")
        hab_type  = parse_habitat_type(habitat_raw) if habitat_raw else ""
        geo_notes = parse_geographic_notes(habitat_raw) if habitat_raw else ""

        status = "✓" if habitat_raw else "✗"
        elev_str = f"{elev_min:.0f}–{elev_max:.0f} m" if elev_min is not None else "no elev"
        print(f"  {status} {species:<35s} [{vol:<10s}] {extraction_method:<12s} {elev_str}")
        if habitat_raw:
            print(f"    {habitat_raw[:100]}")

        updates.append({
            "species":              species,
            "volume":               vol,
            "habitat_raw":          habitat_raw or "",
            "habitat_type":         hab_type,
            "elevation_min_m":      elev_min,
            "elevation_max_m":      elev_max,
            "elev_outlier_min_m":   out_min,
            "elev_outlier_max_m":   out_max,
            "geographic_notes":     geo_notes,
            "extraction_method":    extraction_method,
        })

    # Merge updates back into the full DataFrame
    updates_df = pd.DataFrame(updates).set_index("species")
    for col in ["volume", "habitat_raw", "habitat_type", "elevation_min_m",
                "elevation_max_m", "elev_outlier_min_m", "elev_outlier_max_m",
                "geographic_notes", "extraction_method"]:
        df.loc[df["species"].isin(TARGETS), col] = df.loc[
            df["species"].isin(TARGETS), "species"
        ].map(updates_df[col])

    df.to_csv(args.output, index=False, encoding="utf-8")

    total = len(TARGETS)
    print(f"\n{'─'*60}")
    print(f"  Re-extracted: {total}")
    print(f"  exact={stats['exact']}  fuzzy={stats['fuzzy']}  failed={stats['failed']}")
    print(f"  Saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
