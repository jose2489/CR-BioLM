"""Frozen, stratified species sample for the pilot and the main benchmark.

Two axes, because they measure different things and correlate only moderately
(Spearman 0.46 over 6,917 species, 2026-09-13):

  G  global familiarity   — GBIF occurrence records worldwide (accepted key). LLM
                            knowledge should follow this: a pantropical weed or a
                            European pasture grass is well known even with few Costa
                            Rican records.
  L  local evidence       — records in the frozen Costa Rican snapshot (DOI
                            10.15468/dl.8yhee8). How much occurrence data exists to
                            ground an answer.

Design: tertiles of G x tertiles of L = 9 cells, EQUAL allocation per cell, so the
off-diagonal cells (known globally but sparse here; well recorded here but obscure)
are as well represented as the diagonal. Population estimates must reweight by the
frame counts per cell recorded in the output.

Frame (all counted in the output):
  - catalog species resolved to a GBIF accepted key
  - excluded: unnamed morphospecies ("Licaria sp. 3") — cannot be asked about
  - excluded: species sharing an accepted key with another Manual species — the Manual
    separates what GBIF lumps, so occurrence evidence cannot be attributed

Pilot and main samples are DISJOINT: prompts and rubrics get tuned on the pilot.

Run:  python -m benchmark.sampler            (writes benchmark/data/species_sample.json)
"""
from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

import numpy as np

from mpcr_rag import config
from mpcr_rag.store import pg_store

OUT = Path(__file__).resolve().parent / "data" / "species_sample.json"
SEED = 20260913
SNAPSHOT_DOI = "10.15468/dl.8yhee8"
PILOT_PER_CELL = 2
MAIN_PER_CELL = 11
GLOBAL_COUNTS = config.DATA_DIR / "global_counts_gbif.json"


def _frame(cur, global_counts: dict) -> tuple[list[dict], dict]:
    cur.execute("""
        WITH shared AS (
            SELECT taxon_key FROM mpcr.taxa
            WHERE taxon_key IS NOT NULL GROUP BY taxon_key HAVING count(*) > 1)
        SELECT f.species, f.family, f.volume, e.taxon_key, e.n_records, e.n_cells,
               (f.species ~ ' sp\\. ?[0-9]')                 AS morpho,
               (e.taxon_key IN (SELECT taxon_key FROM shared)) AS shared_key
        FROM mpcr.fichas f JOIN mpcr.species_evidence e USING (species)
        ORDER BY f.species""")
    cols = [c.name for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    counts = {"catalog": len(rows),
              "morphospecies": sum(r["morpho"] for r in rows),
              "no_gbif_key": sum(r["taxon_key"] is None and not r["morpho"] for r in rows),
              "shared_gbif_key": sum(bool(r["shared_key"]) and not r["morpho"] for r in rows)}
    frame = []
    for r in rows:
        if r["morpho"] or r["shared_key"] or r["taxon_key"] is None:
            continue
        r["n_global"] = int(global_counts[str(int(r["taxon_key"]))])
        frame.append(r)
    counts["frame"] = len(frame)
    return frame, counts


def _tertile_edges(values: list[int]) -> list[float]:
    return [float(np.quantile(values, 1 / 3)), float(np.quantile(values, 2 / 3))]


def _tier(v: int, edges: list[float]) -> int:
    return 1 if v <= edges[0] else (2 if v <= edges[1] else 3)


def build() -> dict:
    gc = json.loads(GLOBAL_COUNTS.read_text(encoding="utf-8"))
    conn = pg_store.connect()
    with conn.cursor() as cur:
        frame, counts = _frame(cur, gc["counts"])
    conn.close()

    g_edges = _tertile_edges([r["n_global"] for r in frame])
    l_edges = _tertile_edges([r["n_records"] for r in frame])
    cells: dict[str, list[dict]] = {}
    for r in frame:
        cid = f"G{_tier(r['n_global'], g_edges)}L{_tier(r['n_records'], l_edges)}"
        cells.setdefault(cid, []).append(r)

    rng = random.Random(SEED)
    pilot, main = [], []
    for cid in sorted(cells):
        pool = sorted(cells[cid], key=lambda r: r["species"])
        picked = rng.sample(pool, PILOT_PER_CELL + MAIN_PER_CELL)
        for i, r in enumerate(picked):
            item = {"species": r["species"], "family": r["family"], "volume": r["volume"],
                    "stratum": cid, "taxon_key": int(r["taxon_key"]),
                    "n_records": r["n_records"], "n_cells": r["n_cells"], "n_global": r["n_global"]}
            (pilot if i < PILOT_PER_CELL else main).append(item)

    out = {
        "created": dt.date.today().isoformat(),
        "seed": SEED,
        "axes": {
            "G": f"global GBIF occurrence records (live count retrieved {gc.get('retrieved')})",
            "L": f"Costa Rican records in the frozen snapshot, DOI {SNAPSHOT_DOI}",
        },
        "tertile_edges": {"G": g_edges, "L": l_edges},
        "allocation": f"equal: {PILOT_PER_CELL} pilot + {MAIN_PER_CELL} main per cell; "
                      "reweight by cell frame_size for population estimates",
        "counts": counts,
        "cells": [{"id": cid, "frame_size": len(cells[cid])} for cid in sorted(cells)],
        "pilot": pilot,
        "main": main,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    s = build()
    print(json.dumps(s["counts"]), "edges", s["tertile_edges"])
    print("cells:", [(c["id"], c["frame_size"]) for c in s["cells"]])
    print(f"\npilot ({len(s['pilot'])}):")
    for x in s["pilot"]:
        print(f"  {x['stratum']}  CR={x['n_records']:5} global={x['n_global']:8}  {x['species']:32} {x['family']}")
    print(f"main: {len(s['main'])} species")
