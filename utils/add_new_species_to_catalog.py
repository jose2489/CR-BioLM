"""
add_new_species_to_catalog.py
------------------------------
One-shot script: extract Manual de Plantas habitat data for 33 additional
species and append them to both catalog CSVs.

Strategy:
  • Species in families covered by available PDF volumes (II, VI, VIII):
    PDF extraction via helpers reused from extract_habitat_from_pdf.py.
  • Species in families A–G (Vol. IV/V, no PDFs available):
    LLM with a Costa Rica–specific knowledge prompt (no PDF block needed).
  • Newly extracted rows are cleaned with the same OCR rules used for the
    original 100 species, then appended to both catalog files.

Usage:
    python utils/add_new_species_to_catalog.py [--no-llm]
"""

import argparse
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from utils.extract_habitat_from_pdf import (
    load_pdf_text,
    find_species_block,
    extract_habitat_regex,
    parse_elevation,
    parse_habitat_type,
    parse_geographic_notes,
)
from utils.clean_species_catalog import clean_dataframe

RAW_CATALOG   = os.path.join("outputs", "picked_species_enhanced.csv")
CLEAN_CATALOG = os.path.join("outputs", "picked_species_enhanced_clean.csv")

# Families with confirmed available PDFs (alphabetical range boundaries):
# Vol. II  → Liliopsida families "Agavaceae" ≤ f ≤ "Musaceae"
# Vol. VI  → Magnoliopsida families "Haloragaceae" ≤ f ≤ "Phytolaccaceae"
# Vol. VIII→ Magnoliopsida families "Sabiaceae" ≤ f ≤ "Zygophyllaceae"
_VOL_PDF_MAP = {
    "Vol. II":   "ManualPlantasCostaRica_BHL_v2.pdf",
    "Vol. III":  "ManualPlantasCostaRica_BHL_v3.pdf",
    "Vol. VI":   "ManualPlantasCostaRica_BHL_v6.pdf",
    "Vol. VIII": "ManualPlantasCostaRica_BHL_v8.pdf",
}

# ---------------------------------------------------------------------------
# Species table — 33 additions
# volume = target PDF volume; "?" = no PDF available, go straight to LLM
# ---------------------------------------------------------------------------
SPECIES_DATA: list[dict] = [
    # ── Vol. II (Monocots A–M) ───────────────────────────────────────────────
    {"species": "Cipura cubensis",             "phylum": "Tracheophyta", "class": "Liliopsida",    "order": "Asparagales",  "family": "Iridaceae",      "occurrences": 0,   "volume": "Vol. II",   "volume_title": "Gimnospermas y Monocotiledóneas (Agavaceae–Musaceae)"},

    # ── Vol. VI (Dicots H–Ph) ────────────────────────────────────────────────
    {"species": "Batocarpus costaricensis",    "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Rosales",      "family": "Moraceae",       "occurrences": 131, "volume": "Vol. VI",   "volume_title": "Dicotiledóneas (Haloragaceae–Phytolaccaceae)"},
    {"species": "Brosimum utile",              "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Rosales",      "family": "Moraceae",       "occurrences": 117, "volume": "Vol. VI",   "volume_title": "Dicotiledóneas (Haloragaceae–Phytolaccaceae)"},
    {"species": "Cedrela odorata",             "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Sapindales",   "family": "Meliaceae",      "occurrences": 212, "volume": "Vol. VI",   "volume_title": "Dicotiledóneas (Haloragaceae–Phytolaccaceae)"},
    {"species": "Cedrela tonduzii",            "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Sapindales",   "family": "Meliaceae",      "occurrences": 133, "volume": "Vol. VI",   "volume_title": "Dicotiledóneas (Haloragaceae–Phytolaccaceae)"},
    {"species": "Ceiba pentandra",             "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Malvales",     "family": "Malvaceae",      "occurrences": 230, "volume": "Vol. VI",   "volume_title": "Dicotiledóneas (Haloragaceae–Phytolaccaceae)"},
    {"species": "Cuphea utriculosa",           "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Myrtales",     "family": "Lythraceae",     "occurrences": 208, "volume": "Vol. VI",   "volume_title": "Dicotiledóneas (Haloragaceae–Phytolaccaceae)"},
    {"species": "Lecythis ampla",              "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Ericales",     "family": "Lecythidaceae",  "occurrences": 114, "volume": "Vol. VI",   "volume_title": "Dicotiledóneas (Haloragaceae–Phytolaccaceae)"},
    {"species": "Oreamunnea pterocarpa",       "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Fagales",      "family": "Juglandaceae",   "occurrences": 0,   "volume": "Vol. VI",   "volume_title": "Dicotiledóneas (Haloragaceae–Phytolaccaceae)"},
    {"species": "Passiflora bicornis",         "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Malpighiales", "family": "Passifloraceae", "occurrences": 123, "volume": "Vol. VI",   "volume_title": "Dicotiledóneas (Haloragaceae–Phytolaccaceae)"},
    {"species": "Passiflora membranacea",      "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Malpighiales", "family": "Passifloraceae", "occurrences": 340, "volume": "Vol. VI",   "volume_title": "Dicotiledóneas (Haloragaceae–Phytolaccaceae)"},
    {"species": "Passiflora platyloba",        "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Malpighiales", "family": "Passifloraceae", "occurrences": 117, "volume": "Vol. VI",   "volume_title": "Dicotiledóneas (Haloragaceae–Phytolaccaceae)"},
    {"species": "Passiflora tica",             "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Malpighiales", "family": "Passifloraceae", "occurrences": 140, "volume": "Vol. VI",   "volume_title": "Dicotiledóneas (Haloragaceae–Phytolaccaceae)"},
    {"species": "Spirotheca rosea",            "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Malvales",     "family": "Malvaceae",      "occurrences": 90,  "volume": "Vol. VI",   "volume_title": "Dicotiledóneas (Haloragaceae–Phytolaccaceae)"},

    # ── Vol. VIII (Dicots S–Z) ───────────────────────────────────────────────
    {"species": "Macrohasseltia macroterantha", "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Malpighiales", "family": "Salicaceae",     "occurrences": 183, "volume": "Vol. VIII", "volume_title": "Dicotiledóneas (Sabiaceae–Zygophyllaceae)"},
    {"species": "Stachytarpheta calderonii",   "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Lamiales",     "family": "Verbenaceae",    "occurrences": 77,  "volume": "Vol. VIII", "volume_title": "Dicotiledóneas (Sabiaceae–Zygophyllaceae)"},

    # ── Families A–G: no PDF available → LLM fallback ───────────────────────
    # Assigned "?" volume; script goes straight to knowledge-based LLM prompt.
    {"species": "Albizia niopoides",           "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Fabales",      "family": "Fabaceae",       "occurrences": 81,  "volume": "?",         "volume_title": ""},
    {"species": "Anacardium excelsum",         "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Sapindales",   "family": "Anacardiaceae",  "occurrences": 311, "volume": "?",         "volume_title": ""},
    {"species": "Anthodiscus chocoensis",      "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Malpighiales", "family": "Caryocaraceae",  "occurrences": 88,  "volume": "?",         "volume_title": ""},
    {"species": "Astronium graveolens",        "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Sapindales",   "family": "Anacardiaceae",  "occurrences": 155, "volume": "?",         "volume_title": ""},
    {"species": "Balizia elegans",             "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Fabales",      "family": "Fabaceae",       "occurrences": 57,  "volume": "?",         "volume_title": ""},
    {"species": "Campnosperma panamensis",     "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Sapindales",   "family": "Anacardiaceae",  "occurrences": 1,   "volume": "?",         "volume_title": ""},
    {"species": "Caryocar costaricense",       "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Malpighiales", "family": "Caryocaraceae",  "occurrences": 188, "volume": "?",         "volume_title": ""},
    {"species": "Chaunochiton kappleri",       "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Santalales",   "family": "Aptandraceae",   "occurrences": 83,  "volume": "?",         "volume_title": ""},
    {"species": "Chiangiodendron mexicanum",   "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Malpighiales", "family": "Achariaceae",    "occurrences": 32,  "volume": "?",         "volume_title": ""},
    {"species": "Copaifera aromatica",         "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Fabales",      "family": "Fabaceae",       "occurrences": 38,  "volume": "?",         "volume_title": ""},
    {"species": "Cordia gerascanthus",         "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Boraginales",  "family": "Cordiaceae",     "occurrences": 35,  "volume": "?",         "volume_title": ""},
    {"species": "Crescentia cujete",           "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Lamiales",     "family": "Bignoniaceae",   "occurrences": 366, "volume": "?",         "volume_title": ""},
    {"species": "Dipteryx oleifera",           "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Fabales",      "family": "Fabaceae",       "occurrences": 126, "volume": "?",         "volume_title": ""},
    {"species": "Hymenolobium mesoamericanum", "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Fabales",      "family": "Fabaceae",       "occurrences": 76,  "volume": "?",         "volume_title": ""},
    {"species": "Nelsonia canescens",          "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Lamiales",     "family": "Acanthaceae",    "occurrences": 191, "volume": "?",         "volume_title": ""},
    {"species": "Peltogyne purpurea",          "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Fabales",      "family": "Fabaceae",       "occurrences": 114, "volume": "?",         "volume_title": ""},
    {"species": "Tachigali versicolor",        "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Fabales",      "family": "Fabaceae",       "occurrences": 107, "volume": "?",         "volume_title": ""},
]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
_llm_client = None


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI
        _llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,
        )
    return _llm_client


def _llm_from_pdf_block(species: str, block: str) -> str | None:
    """Existing fallback: extract from a PDF text block."""
    client = _get_llm_client()
    prompt = (
        f"Below is a botanical entry for '{species}' from the Manual de Plantas de Costa Rica.\n"
        "Extract ONLY the habitat/location sentence — the one that describes the vegetation type, "
        "elevation (in metres), and geographic distribution within Costa Rica. "
        "Return just that sentence, nothing else.\n\n"
        f"TEXT:\n{block[:1200]}"
    )
    return _llm_call(client, species, prompt)


def _llm_from_knowledge(species: str) -> str | None:
    """Knowledge-based fallback for species not in available PDF volumes."""
    client = _get_llm_client()
    prompt = (
        f"You are a botanical reference assistant specialised in Costa Rican flora.\n"
        f"For the species '{species}', provide the habitat/distribution sentence exactly as it "
        f"would appear in the Manual de Plantas de Costa Rica (Hammel, Grayum, Herrera, Zamora).\n"
        "Include: vegetation type (Bosque húmedo/muy húmedo/pluvial, Matorral, etc.), "
        "elevation range in metres (e.g. 0–800 m), and geographic distribution "
        "(vert. Pac., vert. Carib., specific cordilleras, llanuras, or peninsulas).\n"
        "Format: 'Bosque [type], [elevation] m; [distribution]. [country range].'\n"
        "Return ONLY the habitat sentence."
    )
    return _llm_call(client, species, prompt)


def _llm_call(client, species: str, prompt: str) -> str | None:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-3.2-3b-instruct:free",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            result = response.choices[0].message.content.strip()
            time.sleep(8)
            return result if result else None
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 20 * (attempt + 1)
                print(f"  [rate-limit] waiting {wait}s ...")
                time.sleep(wait)
            else:
                print(f"  [!] LLM error for {species}: {e}")
                return None
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM fallback")
    args = parser.parse_args()

    # Guard: don't add duplicates
    existing_raw   = pd.read_csv(RAW_CATALOG)
    existing_clean = pd.read_csv(CLEAN_CATALOG)
    already_in     = set(existing_clean["species"].tolist())

    to_process = [r for r in SPECIES_DATA if r["species"] not in already_in]
    if not to_process:
        print("[OK] All 33 species already present in catalog — nothing to add.")
        return

    skipped = len(SPECIES_DATA) - len(to_process)
    if skipped:
        print(f"[INFO] Skipping {skipped} species already in catalog.")

    stats = {"pdf_exact": 0, "pdf_fuzzy": 0, "llm_block": 0, "llm_knowledge": 0, "failed": 0}
    results = []

    for row in to_process:
        species = row["species"]
        volume  = row["volume"]

        habitat_raw       = None
        extraction_method = "failed"

        if volume in _VOL_PDF_MAP:
            # ── PDF extraction path ──────────────────────────────────────────
            pdf_text = load_pdf_text(volume)
            block, find_method = find_species_block(pdf_text, species)

            if block:
                habitat_raw = extract_habitat_regex(block)
                if habitat_raw:
                    extraction_method = find_method
                    stats[f"pdf_{find_method}"] = stats.get(f"pdf_{find_method}", 0) + 1
                elif not args.no_llm:
                    print(f"  [LLM-block] {species}")
                    habitat_raw = _llm_from_pdf_block(species, block)
                    extraction_method = "llm" if habitat_raw else "failed"
                    if habitat_raw:
                        stats["llm_block"] += 1
            elif not args.no_llm:
                # Not found in PDF → fall through to knowledge-based LLM
                print(f"  [LLM-knowledge] {species} (not in {volume} PDF)")
                habitat_raw = _llm_from_knowledge(species)
                extraction_method = "llm" if habitat_raw else "failed"
                if habitat_raw:
                    stats["llm_knowledge"] += 1

        else:
            # ── No PDF available → knowledge-based LLM ───────────────────────
            if not args.no_llm:
                print(f"  [LLM-knowledge] {species} (no PDF for family {row['family']})")
                habitat_raw = _llm_from_knowledge(species)
                extraction_method = "llm" if habitat_raw else "failed"
                if habitat_raw:
                    stats["llm_knowledge"] += 1
            else:
                print(f"  [SKIP] {species} — no PDF, --no-llm set")

        if extraction_method == "failed":
            stats["failed"] += 1

        elev_min, elev_max, outlier_min, outlier_max = parse_elevation(habitat_raw or "")

        results.append({
            "volume":            row["volume"] if row["volume"] != "?" else "",
            "volume_title":      row["volume_title"],
            "species":           species,
            "phylum":            row["phylum"],
            "class":             row["class"],
            "order":             row["order"],
            "family":            row["family"],
            "occurrences":       row["occurrences"],
            "habitat_raw":       habitat_raw or "",
            "habitat_type":      parse_habitat_type(habitat_raw) if habitat_raw else "",
            "elevation_min_m":   elev_min,
            "elevation_max_m":   elev_max,
            "elev_outlier_min_m": outlier_min,
            "elev_outlier_max_m": outlier_max,
            "geographic_notes":  parse_geographic_notes(habitat_raw) if habitat_raw else "",
            "extraction_method": extraction_method,
        })

        status = "✓" if habitat_raw else "✗"
        vol_label = volume if volume != "?" else "LLM"
        print(f"  {status} [{vol_label}] {species:<48s} → {extraction_method}")

    if not results:
        print("[INFO] Nothing new to append.")
        return

    new_df = pd.DataFrame(results)

    # Append to raw catalog
    updated_raw = pd.concat([existing_raw, new_df], ignore_index=True)
    updated_raw.to_csv(RAW_CATALOG, index=False, encoding="utf-8")

    # Clean and append to clean catalog
    new_clean = clean_dataframe(new_df)
    updated_clean = pd.concat([existing_clean, new_clean], ignore_index=True)
    updated_clean.to_csv(CLEAN_CATALOG, index=False, encoding="utf-8")

    total = len(results)
    succeeded = total - stats["failed"]
    print(f"\n{'─'*60}")
    print(f"  Added species     : {total}")
    print(f"  Extracted         : {succeeded}")
    print(f"    PDF (exact)     : {stats.get('pdf_exact', 0)}")
    print(f"    PDF (fuzzy)     : {stats.get('pdf_fuzzy', 0)}")
    print(f"    LLM (pdf block) : {stats['llm_block']}")
    print(f"    LLM (knowledge) : {stats['llm_knowledge']}")
    print(f"  Failed            : {stats['failed']}")
    print(f"  Catalog now has   : {len(updated_clean)} species")
    print(f"  → {os.path.abspath(CLEAN_CATALOG)}")


if __name__ == "__main__":
    main()
