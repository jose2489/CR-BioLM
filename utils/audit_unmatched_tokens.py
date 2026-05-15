"""
audit_unmatched_tokens.py — V2
--------------------------------
Audits which tokens/phrases in geographic_notes are NOT matched by any
pattern in TRANSLATION_TABLE_V2 (after stripping vertiente markers).

Iterates every row in outputs/picked_species_enhanced_clean.csv, strips known
matched substrings, tokenizes the remainder, counts frequencies, and writes
the top-30 unmatched tokens to outputs/audit_unmatched_v2.csv.

Run: python utils/audit_unmatched_tokens.py
"""

import csv
import os
import re
import sys

sys.path.insert(0, ".")
from utils.map_gen.habitat_map import TRANSLATION_TABLE_V2, _normalize

_CATALOG = os.path.join("outputs", "picked_species_enhanced_clean.csv")
_OUTPUT  = os.path.join("outputs", "audit_unmatched_v2.csv")

# Vertiente phrases to strip before tokenizing
_VERT_PATTERNS = [
    r"vert\.?\s*carib(?:e|ibe)?",
    r"vert\.?\s*pac(?:ific[ao])?",
    r"vertiente\s*carib(?:e|ibe)?",
    r"vertiente\s*pacif\w*",
    r"ambas\s*vert\w*",
    r"toda\s*(?:la\s*)?vert\w*",
]

# Noise words to ignore after stripping
_STOPWORDS = {
    "y", "o", "de", "del", "la", "el", "los", "las", "a", "en", "con", "por",
    "entre", "cerca", "s", "n", "fl", "ene", "feb", "mar", "abr", "may",
    "jun", "jul", "ago", "sep", "oct", "nov", "dic", "m", "mex", "pan",
    "arg", "antillas", "cr", "mo", "se", "por", "muy", "su", "al",
}


def _strip_matched(text: str, norm: str) -> str:
    """Remove matched pattern substrings from normalized text."""
    for vert_pat in _VERT_PATTERNS:
        norm = re.sub(vert_pat, " ", norm)
    for pattern, _ in TRANSLATION_TABLE_V2:
        norm = re.sub(pattern, " ", norm)
    return norm


def _tokenize(text: str) -> list[str]:
    """Split on commas, semicolons, dots, parens, and whitespace; clean up."""
    parts = re.split(r"[,;.()\[\]/]", text)
    tokens = []
    for p in parts:
        word = p.strip()
        if len(word) >= 3 and word not in _STOPWORDS and not word.isdigit():
            tokens.append(word)
    return tokens


def main():
    if not os.path.exists(_CATALOG):
        print(f"[ERROR] Catalog not found: {_CATALOG}")
        sys.exit(1)

    counts: dict[str, int]               = {}
    examples: dict[str, str]             = {}

    with open(_CATALOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            species = row.get("species", "").strip()
            notes   = row.get("geographic_notes", "") or ""
            if not notes.strip():
                continue

            norm      = _normalize(notes)
            remainder = _strip_matched(notes, norm)
            tokens    = _tokenize(remainder)

            for tok in tokens:
                counts[tok]  = counts.get(tok, 0) + 1
                if tok not in examples:
                    examples[tok] = species

    if not counts:
        print("[INFO] No unmatched tokens found.")
        return

    sorted_tokens = sorted(counts.items(), key=lambda x: -x[1])[:30]

    os.makedirs(os.path.dirname(_OUTPUT), exist_ok=True)
    with open(_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "count", "example_species"])
        for tok, cnt in sorted_tokens:
            writer.writerow([tok, cnt, examples.get(tok, "")])

    print(f"[OK] Top {len(sorted_tokens)} unmatched tokens written → {_OUTPUT}")
    print("\nTop 15 unmatched tokens:")
    for tok, cnt in sorted_tokens[:15]:
        print(f"  {cnt:3d}x  {tok}")


if __name__ == "__main__":
    main()
