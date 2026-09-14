"""Global GBIF occurrence count per catalog species (accepted key, all countries).

A familiarity proxy that is independent of Costa Rica: a pantropical weed can have no
Costa Rican records in the snapshot and still be very well known. Used next to the
local snapshot count, which measures how much occurrence evidence exists here.

Note: this is a LIVE count (it grows over time), unlike the frozen snapshot; the
retrieval date is stored with the values.

Run:  python -m mpcr_rag.evidence.global_counts
"""
from __future__ import annotations

import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from pygbif import occurrences

from .. import config
from . import taxa

CACHE = config.DATA_DIR / "global_counts_gbif.json"


def _count(key: int, attempts: int = 4) -> int | None:
    for i in range(attempts):
        try:
            return int(occurrences.count(taxonKey=key))
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def fetch(workers: int = 8) -> dict:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {"counts": {}}
    keys = sorted({v["taxon_key"] for v in taxa.load().values() if v["taxon_key"]})
    todo = [k for k in keys if str(k) not in cache["counts"]]
    print(f"[global] {len(keys)} keys, {len(todo)} to fetch", flush=True)
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_count, k): k for k in todo}
        for fut in as_completed(futures):
            n = fut.result()
            if n is None:
                failed += 1
            else:
                cache["counts"][str(futures[fut])] = n
            done += 1
            if done % 1000 == 0:
                print(f"[global] {done}/{len(todo)}", flush=True)
    cache["retrieved"] = dt.date.today().isoformat()
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    print(f"[global] done; {failed} failures left for a rerun", flush=True)
    return cache


if __name__ == "__main__":
    fetch()
