"""
Replace failed-extraction species with verified alternatives.

For each species in picked_species_enhanced.csv where extraction_method == 'failed',
this script:
  1. Finds eligible alternates from the same volume pool (gbif_species_summary.csv)
  2. Verifies each candidate actually has a habitat line in the PDF
  3. Swaps in the first verified replacement
  4. Re-runs habitat extraction on all replacements
  5. Saves updated picked_species.csv and picked_species_enhanced.csv

Usage:
    python utils/replace_failed_species.py [--seed 99] [--min-occ 150]
"""

import argparse
import os
import re
import sys

import fitz
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUT_BASE_DIR

# Reuse logic from the two existing scripts
from pick_species_by_volume import VOLUMES, is_binomial
from extract_habitat_from_pdf import (
    PDF_DIR, VOLUME_TO_PDF,
    load_pdf_text, find_species_block, extract_habitat_regex,
    parse_elevation, parse_habitat_type, parse_geographic_notes,
)

INPUT_SUMMARY  = os.path.join(OUTPUT_BASE_DIR, "gbif_species_summary.csv")
INPUT_PICKED   = os.path.join(OUTPUT_BASE_DIR, "picked_species.csv")
INPUT_ENHANCED = os.path.join(OUTPUT_BASE_DIR, "picked_species_enhanced.csv")
OUTPUT_PICKED   = INPUT_PICKED    # overwrite in-place
OUTPUT_ENHANCED = INPUT_ENHANCED  # overwrite in-place


def find_verified_replacement(
    volume_label: str,
    excluded_species: set[str],
    summary_df: pd.DataFrame,
    vol_def: dict,
    min_occ: int,
    seed_offset: int,
) -> tuple[pd.Series | None, str | None]:
    """
    Returns (gbif_row, habitat_raw) for the first candidate that has a
    verifiable habitat entry in the PDF, or (None, None) if none found.
    """
    pool = vol_def["filter"](summary_df)
    pool = pool[pool["occurrences"] >= min_occ]
    pool = pool[pool["species"].apply(is_binomial)]
    pool = pool[~pool["species"].isin(excluded_species)]
    pool = pool.sample(frac=1, random_state=42 + seed_offset)  # shuffle

    text = load_pdf_text(volume_label)

    for _, row in pool.iterrows():
        block, _ = find_species_block(text, row["species"])
        if not block:
            continue
        habitat = extract_habitat_regex(block)
        if habitat:
            return row, habitat

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",    type=int, default=99,  help="Random seed offset for replacements")
    parser.add_argument("--min-occ", type=int, default=150, help="Min occurrences threshold")
    args = parser.parse_args()

    summary_df = pd.read_csv(INPUT_SUMMARY, dtype={"class": str, "family": str, "phylum": str, "order": str})
    enhanced   = pd.read_csv(INPUT_ENHANCED)
    picked     = pd.read_csv(INPUT_PICKED)

    failed = enhanced[enhanced["extraction_method"] == "failed"]
    print(f"Found {len(failed)} failed species to replace.\n")

    if len(failed) == 0:
        print("Nothing to do.")
        return

    # Build vol_def lookup by label
    vol_by_label = {v["volume"]: v for v in VOLUMES}

    # Track all currently picked species (to avoid duplicates)
    excluded = set(picked["species"].tolist())

    replacements = []

    for _, fail_row in failed.iterrows():
        vol_label = fail_row["volume"]
        vol_def   = vol_by_label.get(vol_label)
        if vol_def is None:
            print(f"  [!] Unknown volume label: {vol_label}")
            continue

        print(f"  Replacing [{vol_label}] {fail_row['species']} ...")
        new_row, habitat_raw = find_verified_replacement(
            vol_label, excluded, summary_df, vol_def, args.min_occ, len(replacements)
        )

        if new_row is None:
            print(f"  [!] No verified replacement found for {fail_row['species']}")
            continue

        excluded.add(new_row["species"])
        elev_min, elev_max = parse_elevation(habitat_raw)

        enhanced_row = {
            "volume":            vol_label,
            "volume_title":      fail_row["volume_title"],
            "species":           new_row["species"],
            "phylum":            new_row.get("phylum", ""),
            "class":             new_row.get("class", ""),
            "order":             new_row.get("order", ""),
            "family":            new_row.get("family", ""),
            "occurrences":       new_row["occurrences"],
            "habitat_raw":       habitat_raw,
            "habitat_type":      parse_habitat_type(habitat_raw),
            "elevation_min_m":   elev_min,
            "elevation_max_m":   elev_max,
            "geographic_notes":  parse_geographic_notes(habitat_raw),
            "extraction_method": "exact",
        }

        replacements.append({
            "old_species": fail_row["species"],
            "new_species": new_row["species"],
            "volume":      vol_label,
            "enhanced_row": enhanced_row,
            "picked_row": {
                "volume":       vol_label,
                "volume_title": fail_row["volume_title"],
                "species":      new_row["species"],
                "phylum":       new_row.get("phylum", ""),
                "class":        new_row.get("class", ""),
                "order":        new_row.get("order", ""),
                "family":       new_row.get("family", ""),
                "occurrences":  new_row["occurrences"],
            }
        })
        print(f"    → {new_row['species']}  ({habitat_raw[:80]}...)")

    if not replacements:
        print("\nNo replacements made.")
        return

    # Apply replacements to enhanced dataframe
    for r in replacements:
        mask = (enhanced["species"] == r["old_species"]) & (enhanced["volume"] == r["volume"])
        for col, val in r["enhanced_row"].items():
            if col in enhanced.columns:
                enhanced.loc[mask, col] = val
            else:
                enhanced[col] = enhanced.get(col, "")
                enhanced.loc[mask, col] = val

    # Apply replacements to picked dataframe
    for r in replacements:
        mask = (picked["species"] == r["old_species"]) & (picked["volume"] == r["volume"])
        for col, val in r["picked_row"].items():
            if col in picked.columns:
                picked.loc[mask, col] = val

    enhanced.to_csv(OUTPUT_ENHANCED, index=False, encoding="utf-8")
    picked.to_csv(OUTPUT_PICKED, index=False, encoding="utf-8")

    print(f"\n{'─'*60}")
    print(f"  Replaced : {len(replacements)}/{len(failed)}")
    print(f"  Still failed: {len(failed) - len(replacements)}")
    print(f"\n  Replacements summary:")
    for r in replacements:
        print(f"    [{r['volume']}] {r['old_species']:<40s} → {r['new_species']}")
    print(f"\n  Saved: {os.path.abspath(OUTPUT_ENHANCED)}")
    print(f"  Saved: {os.path.abspath(OUTPUT_PICKED)}")


if __name__ == "__main__":
    main()
