"""
clean_species_catalog.py
------------------------
Cleans OCR artifacts from picked_species_enhanced.csv.

All patterns were catalogued directly from the raw text fields.
Produces outputs/picked_species_enhanced_clean.csv.

Usage:
    python utils/clean_species_catalog.py
"""

import re
import pandas as pd
from pathlib import Path

INPUT  = Path("outputs/picked_species_enhanced.csv")
OUTPUT = Path("outputs/picked_species_enhanced_clean.csv")


# ---------------------------------------------------------------------------
# Ordered substitution rules — (pattern, replacement)
# Applied sequentially to every text cell.
# ---------------------------------------------------------------------------
RULES: list[tuple[str, str]] = [

    # ── Hyphenated line-breaks from PDF column wrapping ───────────────────
    # "Gua- nacaste" → "Guanacaste",  "Tala- manca" → "Talamanca", etc.
    (r'(\w+)-\s*\n?\s*([a-záéíóúüñ])', r'\1\2'),   # hyphen + lowercase continuation
    (r'(\w+)¬\s*([a-záéíóúüñ])',        r'\1\2'),   # soft-hyphen variant (¬)

    # ── OCR digit/letter confusions ───────────────────────────────────────
    # "Divisi6n" → "División"
    (r'\bDivisi6n\b',    'División'),
    (r'\bdivisi6n\b',    'división'),
    # "regi6n" → "región"
    (r'\bregi6n\b',      'región'),
    (r'\bRegiOn\b',      'Región'),
    (r'\bregiOn\b',      'región'),
    # "regidn" → "región"  (d instead of ó)
    (r'\bregidn\b',      'región'),
    # "0—" at start of elevation range  "O—1100" → "0–1100"
    (r'\bO—',            '0–'),
    (r'\bO-',            '0–'),
    # "0Q—" artifact
    (r'\b0Q—',           '0–'),
    # Isolated "O" as zero in elevation strings like "(O—)400"
    (r'\(O—\)',          '(0–)'),
    (r'\(O-\)',          '(0–)'),

    # ── "FI." / "FI " → "Fl." (flowering abbreviation) ──────────────────
    (r'\bFI\.',          'Fl.'),
    (r'\bFI ',           'Fl. '),

    # ── "fi" → "ñ" in Spanish words ─────────────────────────────────────
    # "Costefia" → "Costeña",  "Montafia" → "Montaña"
    (r'Costefi([ao])',   r'Costeñ\1'),
    (r'costefi([ao])',   r'costeñ\1'),
    (r'Montafi([ao])',   r'Montañ\1'),
    (r'montafi([ao])',   r'montañ\1'),
    (r'Tamafi([ao])',    r'Tamañ\1'),

    # ── "Ilanuras" → "llanuras" (capital I instead of l) ─────────────────
    (r'\bIlanuras\b',    'llanuras'),
    (r'\blanuras\b',     'llanuras'),   # missing first l

    # ── "htmedo" / "htimedo" → "húmedo" ─────────────────────────────────
    (r'\bht[i]?medo\b',  'húmedo'),
    (r'\bht[i]?meda\b',  'húmeda'),
    (r'\bhimedo\b',      'húmedo'),
    (r'\bhiimedo\b',     'húmedo'),

    # ── "humedo" → "húmedo" (missing accent, common OCR miss) ────────────
    (r'\bhumedo\b',      'húmedo'),
    (r'\bhumeda\b',      'húmeda'),
    (r'\bhumedad\b',     'humedad'),   # keep this one as-is

    # ── em-dash variants: normalize to en-dash for elevation ranges ───────
    # "650—2000" → "650–2000"  (U+2014 → U+2013)
    (r'(\d)\u2014(\d)',  r'\1–\2'),
    # Also fix "0— 1100" with space
    (r'(\d)—\s+(\d)',    r'\1–\2'),
    (r'(\d)-—(\d)',      r'\1–\2'),

    # ── "himedos" / "htmedos" → "húmedos" ────────────────────────────────
    (r'\bh[ti]+medos\b', 'húmedos'),

    # ── "Tarrazt" → "Tarrazú" ─────────────────────────────────────────────
    (r'\bTarrazt\b',     'Tarrazú'),

    # ── "Divisién" / "Divisién" (é instead of ó) ─────────────────────────
    (r'\bDivisién\b',    'División'),
    (r'\bdivisién\b',    'división'),

    # ── "Triarán" → "Tilarán" (misread T for Til) ────────────────────────
    (r'\bTriarán\b',     'Tilarán'),

    # ── Remove stray £ and ± symbols ─────────────────────────────────────
    (r'[£±]',            ''),

    # ── "SO—" at start → "0–" (S misread as leading char) ───────────────
    (r'\bSO—\)',          '0–)'),
    (r'\(SO—\)',          '(0–)'),

    # ── Trailing/leading whitespace cleanup ───────────────────────────────
    (r'[ \t]+',          ' '),      # collapse multiple spaces
    (r'^\s+|\s+$',       ''),       # strip
]


def clean_text(value) -> str:
    if pd.isna(value):
        return value
    text = str(value)
    for pattern, replacement in RULES:
        text = re.sub(pattern, replacement, text)
    return text.strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    TEXT_COLS = ["geographic_notes", "habitat_raw", "habitat_type",
                 "species", "family", "order", "class", "phylum"]
    df = df.copy()
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT)
    print(f"[INFO] Loaded {len(df)} rows from {INPUT}")

    df_clean = clean_dataframe(df)

    # Report changes
    changes = 0
    TEXT_COLS = ["geographic_notes", "habitat_raw", "habitat_type"]
    for col in TEXT_COLS:
        if col not in df.columns:
            continue
        for i, (old, new) in enumerate(zip(df[col].fillna(""), df_clean[col].fillna(""))):
            if old != new:
                changes += 1
                if changes <= 10:
                    print(f"\n  [{col}] row {i}")
                    print(f"    BEFORE: {old[:100]}")
                    print(f"    AFTER : {new[:100]}")

    print(f"\n[INFO] Total cells changed: {changes}")

    df_clean.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"[OK] Saved clean file → {OUTPUT}")
