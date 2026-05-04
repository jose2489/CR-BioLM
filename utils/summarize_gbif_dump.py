"""
Summarize a GBIF Darwin Core Archive occurrence dump.

Reads occurrence.txt in chunks, filters for Kingdom Plantae / Country Costa Rica,
groups by species, counts occurrences, and outputs a CSV sorted from highest to lowest.

Output columns: species, phylum, class, order, family, occurrences

Usage:
    python utils/summarize_gbif_dump.py [--top N] [--output PATH]

Filtering examples (post-processing the CSV):
    Monocots       -> class == "Liliopsida"
    Gymnosperms    -> class in ["Pinopsida", "Cycadopsida", "Gnetopsida"]
    Ferns          -> class == "Polypodiopsida"
    Dicots         -> class == "Magnoliopsida"
"""

import argparse
import os
import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUT_BASE_DIR

INPUT_FILE = r"C:\Users\Jose\Documents\Tesis\raw_data\GBIF\0106080-260226173443078\occurrence.txt"
OUTPUT_CSV = os.path.join(OUTPUT_BASE_DIR, "gbif_species_summary.csv")

TAXON_COLS = ["phylum", "class", "order", "family"]
COLS = ["species", "acceptedScientificName", "kingdom", "countryCode"] + TAXON_COLS
DTYPES = {"kingdom": "category", "countryCode": "category"}
CHUNK_SIZE = 100_000


def summarize(input_file: str, output_csv: str, top_n: int = 20) -> pd.DataFrame:
    counts: dict[str, int] = {}
    # Store first-seen taxonomy per resolved species name
    taxonomy: dict[str, dict] = {}
    total_rows = 0
    filtered_rows = 0

    print(f"Reading: {input_file}")
    reader = pd.read_csv(
        input_file,
        sep="\t",
        usecols=COLS,
        dtype=DTYPES,
        chunksize=CHUNK_SIZE,
        on_bad_lines="skip",
        encoding="utf-8",
    )

    for chunk in reader:
        total_rows += len(chunk)
        chunk = chunk[
            (chunk["kingdom"].str.lower() == "plantae") &
            (chunk["countryCode"].str.upper() == "CR")
        ]
        filtered_rows += len(chunk)

        resolved = chunk["species"].str.strip()
        mask_empty = resolved.isna() | (resolved == "")
        resolved = resolved.where(~mask_empty, chunk["acceptedScientificName"].str.strip())
        chunk = chunk[resolved.notna() & (resolved != "")].copy()
        resolved = resolved[resolved.notna() & (resolved != "")]
        chunk["resolved_species"] = resolved

        # Count occurrences
        for name, count in resolved.value_counts().items():
            counts[name] = counts.get(name, 0) + count

        # Capture taxonomy for new species (first non-null seen)
        new_species = set(resolved.unique()) - taxonomy.keys()
        if new_species:
            subset = chunk[chunk["resolved_species"].isin(new_species)]
            for name, grp in subset.groupby("resolved_species"):
                row = grp.iloc[0]
                taxonomy[name] = {col: row[col] for col in TAXON_COLS}

        print(f"  processed {total_rows:,} rows, kept {filtered_rows:,} Plantae/CR so far...", end="\r")

    print()

    df = pd.DataFrame(
        [{"species": name, **taxonomy.get(name, {col: None for col in TAXON_COLS}), "occurrences": cnt}
         for name, cnt in counts.items()]
    )
    df = df.sort_values("occurrences", ascending=False).reset_index(drop=True)

    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\nTotal rows read   : {total_rows:,}")
    print(f"Plantae / CR rows : {filtered_rows:,}")
    print(f"Unique species    : {len(df):,}")
    print(f"Output saved to   : {os.path.abspath(output_csv)}")
    print(f"\nTop {top_n} species by occurrence count:")
    print(df.head(top_n).to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(description="Summarize GBIF occurrence dump by species.")
    parser.add_argument("--input", default=INPUT_FILE, help="Path to occurrence.txt")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Path for output CSV")
    parser.add_argument("--top", type=int, default=20, help="Number of top species to print (default: 20)")
    args = parser.parse_args()

    summarize(args.input, args.output, args.top)


if __name__ == "__main__":
    main()
