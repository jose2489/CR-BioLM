"""
audit_unmatched_tokens.py — V3
--------------------------------
Audits which text fragments in geographic_notes are NOT resolved by the
gazetteer (after normalization). Uses build_ficha's unresolved_tokens field.

Run: python utils/audit_unmatched_tokens.py
"""

import csv
import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from utils.distribution_map import build_ficha

_CATALOG = os.path.join("outputs", "picked_species_enhanced_clean.csv")
_OUTPUT  = os.path.join("outputs", "audit_unmatched_v3.csv")


def main():
    if not os.path.exists(_CATALOG):
        print(f"[ERROR] Catalog not found: {_CATALOG}")
        sys.exit(1)

    df = pd.read_csv(_CATALOG)
    counts: dict[str, int]   = {}
    examples: dict[str, str] = {}

    for _, row in df.iterrows():
        species = str(row.get("species", "")).strip()
        geo     = str(row.get("geographic_notes", "") or "")
        raw     = str(row.get("habitat_raw", "") or "")
        if not geo.strip() and not raw.strip():
            continue
        ficha = build_ficha(habitat_raw=raw, geographic_notes=geo, species=species)
        for tok in ficha.unresolved_tokens:
            counts[tok]   = counts.get(tok, 0) + 1
            if tok not in examples:
                examples[tok] = species

    if not counts:
        print("[INFO] No unresolved tokens found.")
        return

    sorted_tokens = sorted(counts.items(), key=lambda x: -x[1])[:30]

    os.makedirs(os.path.dirname(_OUTPUT), exist_ok=True)
    with open(_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "count", "example_species"])
        for tok, cnt in sorted_tokens:
            writer.writerow([tok, cnt, examples.get(tok, "")])

    print(f"[OK] Top {len(sorted_tokens)} unresolved tokens written → {_OUTPUT}")
    print("\nTop 15 unresolved tokens:")
    for tok, cnt in sorted_tokens[:15]:
        print(f"  {cnt:3d}x  {tok!r}")


if __name__ == "__main__":
    main()
