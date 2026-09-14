"""Manual name -> GBIF accepted taxon, resolved once for the whole catalog and cached.

``gbif_map.resolve_taxon`` calls the backbone API on every use; the evaluation needs
the mapping for all species at once (to join occurrences, to stratify by record
count), so it is resolved in parallel and persisted.

Match rules are those of ``resolve_taxon``: EXACT backbone match only; synonyms map
to their accepted key. Unnamed morphospecies ("Genus sp. 3") never match.

Run:  python -m mpcr_rag.evidence.taxa            # resolve missing entries, write cache
"""
from __future__ import annotations

import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from pygbif import species as gbif_species

from .. import config

CACHE = config.DATA_DIR / "taxa_gbif.json"


_SPECIES_RANKS = {"SPECIES", "SUBSPECIES", "VARIETY", "FORM"}


def _lookup(name: str, attempts: int = 4, **kw) -> dict | None:
    for i in range(attempts):
        try:
            return gbif_species.name_backbone(scientificName=name, **kw)
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def _entry(nb: dict, via: str) -> dict:
    match = nb.get("diagnostics", {}).get("matchType")
    usage = nb.get("usage") or {}
    if nb.get("synonym") and nb.get("acceptedUsage"):
        acc = nb["acceptedUsage"]
        return {"taxon_key": int(acc["key"]), "accepted_name": acc.get("canonicalName"),
                "synonym": True, "match": match, "via": via}
    return {"taxon_key": int(usage["key"]), "accepted_name": usage.get("canonicalName"),
            "synonym": False, "match": match, "via": via}


def _resolve(sp: str, authority: str = "") -> dict | None:
    """Manual name -> accepted GBIF taxon. None means a network failure (retry later),
    never "no match".

    1. Name only, EXACT match (the rule of gbif_map.resolve_taxon).
    2. If that fails, name + the Manual's authority. Homonyms ("Festuca arundinacea"
       under several authors, each a synonym of a different accepted name) make GBIF
       fall back to the genus or family; the authority disambiguates. A spelling
       VARIANT at species rank is accepted here ("ferdinandii" / "ferdinandi") and
       flagged in ``match``.
    """
    nb = _lookup(sp)
    if nb is None:
        return None
    if nb.get("usage") and nb.get("diagnostics", {}).get("matchType") == "EXACT":
        return _entry(nb, "name")
    first_match = nb.get("diagnostics", {}).get("matchType")
    if authority:
        nb2 = _lookup(f"{sp} {authority}", kingdom="Plantae")
        if nb2 is None:
            return None
        usage = nb2.get("usage") or {}
        if (nb2.get("diagnostics", {}).get("matchType") in ("EXACT", "VARIANT")
                and usage.get("rank") in _SPECIES_RANKS):
            return _entry(nb2, "name+authority")
    return {"taxon_key": None, "accepted_name": None, "synonym": False,
            "match": first_match, "via": "unresolved"}


def _authorities() -> dict[str, str]:
    """Authority as printed in the Manual header: the text between the binomial and the
    first comma ("(L.) Klotzsch", "Schltdl. & Cham.")."""
    import json as _json
    import re
    conn = sqlite3.connect(config.SQLITE_PATH)
    out = {}
    for (js,) in conn.execute("SELECT ficha_json FROM fichas"):
        f = _json.loads(js)
        head = (f.get("full_text") or "").split("\n")[0]
        rest = head[len(f["species"]):].strip()
        out[f["species"]] = re.split(r",\s", rest, maxsplit=1)[0].strip()
    conn.close()
    return out


def load() -> dict[str, dict]:
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def resolve_catalog(workers: int = 8, verbose: bool = True) -> dict[str, dict]:
    """Resolve every catalog species not yet in the cache. Network failures are not
    cached, so a rerun retries them."""
    conn = sqlite3.connect(config.SQLITE_PATH)
    species = [s for (s,) in conn.execute("SELECT species FROM fichas ORDER BY species")]
    conn.close()

    cache = load()
    # also retry earlier non-matches that predate the authority fallback
    todo = [s for s in species
            if s not in cache or (cache[s]["taxon_key"] is None and cache[s].get("via") != "unresolved")]
    if verbose:
        print(f"[taxa] {len(species)} species, {len(todo)} to resolve", flush=True)

    auth = _authorities()
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_resolve, s, auth.get(s, "")): s for s in todo}
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                failed += 1
            else:
                cache[futures[fut]] = res
            done += 1
            if verbose and done % 500 == 0:
                print(f"[taxa] {done}/{len(todo)}", flush=True)
                CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    if verbose:
        n_key = sum(1 for v in cache.values() if v["taxon_key"])
        n_syn = sum(1 for v in cache.values() if v["synonym"])
        print(f"[taxa] resolved {n_key}/{len(cache)} to an accepted key "
              f"({n_syn} via synonym); {failed} network failures left for a rerun",
              flush=True)
    return cache


if __name__ == "__main__":
    resolve_catalog()
