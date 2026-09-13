"""Field-by-field regression report between two catalog builds.

Every re-ingest changes text the maps and the BIP evaluation depend on, so a rebuild
is only accepted after this report shows which species and fields moved, and why.

Run:  python -m mpcr_rag.eval.catalog_diff OLD.sqlite NEW.sqlite [--examples 5]
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from ..schema import Ficha

# Fields that drive maps, filters and the BIP evaluation.
GEO_FIELDS = ["elev_min", "elev_max", "elev_outlier_min", "elev_outlier_max",
              "vertientes", "regions", "forest_types", "distribution_paragraph"]
ENRICH_FIELDS = ["family", "habits", "endemic_cr", "flowering_months", "fruiting_months"]
TEXT_FIELDS = ["morphology", "discussion", "genus_description"]


def load(path: Path) -> dict[str, Ficha]:
    conn = sqlite3.connect(path)
    out = {vid: Ficha.from_json(js)
           for vid, js in conn.execute("SELECT vector_id, ficha_json FROM fichas")}
    conn.close()
    return out


def _norm(v):
    return sorted(v) if isinstance(v, list) else v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--examples", type=int, default=5)
    a = ap.parse_args()

    old, new = load(a.old), load(a.new)
    only_old, only_new = sorted(old.keys() - new.keys()), sorted(new.keys() - old.keys())
    both = sorted(old.keys() & new.keys())
    print(f"species: old={len(old)} new={len(new)} common={len(both)} "
          f"removed={len(only_old)} added={len(only_new)}")
    for label, ids in (("removed", only_old), ("added", only_new)):
        if ids:
            print(f"  {label}: {', '.join(ids[:a.examples])}{' …' if len(ids) > a.examples else ''}")

    changed: dict[str, list[str]] = defaultdict(list)
    for vid in both:
        fo, fn = old[vid], new[vid]
        for fld in GEO_FIELDS + ENRICH_FIELDS:
            if _norm(getattr(fo, fld, None)) != _norm(getattr(fn, fld, None)):
                changed[fld].append(vid)

    print("\nchanged fields (common species):")
    for fld in GEO_FIELDS + ENRICH_FIELDS:
        ids = changed.get(fld, [])
        print(f"  {fld:24} {len(ids):5}")
    for fld in GEO_FIELDS + ENRICH_FIELDS:
        for vid in changed.get(fld, [])[:a.examples]:
            print(f"    [{fld}] {vid}: {_norm(getattr(old[vid], fld, None))!s:.90} "
                  f"-> {_norm(getattr(new[vid], fld, None))!s:.90}")

    n = len(new)
    print("\ndescriptive text coverage (new):")
    for fld in TEXT_FIELDS:
        vals = [getattr(f, fld, "") or "" for f in new.values()]
        filled = [len(v) for v in vals if v]
        med = sorted(filled)[len(filled) // 2] if filled else 0
        print(f"  {fld:18} {len(filled):5} ({len(filled)/n:.0%})  median {med} chars")

    # Genus-level text must be constant within a genus. A genus with several distinct
    # values means preamble text leaked across a boundary.
    by_genus: dict[str, Counter] = defaultdict(Counter)
    for f in new.values():
        by_genus[f.genus][(f.genus_description or "")[:80]] += 1
    inconsistent = {g: c for g, c in by_genus.items() if len(c) > 1}
    genera_by_text: dict[str, set] = defaultdict(set)
    for f in new.values():
        if f.genus_description:
            genera_by_text[f.genus_description[:80]].add(f.genus)
    multi_genus = sum(1 for g in genera_by_text.values() if len(g) > 1)
    print(f"\ngenus_description: {len(by_genus)} genera, {len(inconsistent)} with >1 distinct "
          f"value, {multi_genus} values shared by >1 genus")


if __name__ == "__main__":
    main()
