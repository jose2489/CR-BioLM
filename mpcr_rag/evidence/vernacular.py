"""Common names -> species, federated from existing providers.

Nothing here invents a lexicon. Names are collected from providers and stored in the
shape of the Darwin Core VernacularName extension (vernacularName, language,
countryCode, isPreferredName, source), so every candidate an answer rests on keeps its
provenance and can be exported as a checklist later.

Sources, in order of trust:
  MPCR       names printed in the Manual entry              -> cite Tomo and page
  TROPICOS   Missouri Botanical Garden (needs TROPICOS_API_KEY; MOBOT publishes the
             Manual, so its names are expected to mirror MPCR)
  INAT_CR    iNaturalist name preferred for Costa Rica      -> what people say today
  INAT_ES    other Spanish iNaturalist names                -> candidates
  GBIF       GBIF vernacular names (Spanish)                -> recall, no locality

A common name is ambiguous by nature ("poro" is several Erythrina; "roble" is Quercus
and also Tabebuia rosea), so ``resolve`` returns RANKED CANDIDATES with provenance and
never silently picks one.

Run:  python -m mpcr_rag.evidence.vernacular fetch --pilot --expert
      python -m mpcr_rag.evidence.vernacular resolve "poro" [--region ...] [--elev 200]
      python -m mpcr_rag.evidence.vernacular report
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import requests

from ..store import pg_store
from . import taxa

INAT_CR_PLACE = 6924                    # iNaturalist place id for Costa Rica
UA = {"User-Agent": "CR-BioLM thesis research"}
TROPICOS_KEY = os.environ.get("TROPICOS_API_KEY", "")

TRUST = {"MPCR": 0, "TROPICOS": 1, "INAT_CR": 2, "INAT_ES": 3, "GBIF": 4}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mpcr.vernacular (
    name_norm     text NOT NULL,        -- lowercase, unaccented, for lookup
    vernacular    text NOT NULL,        -- as published
    species       text NOT NULL,        -- Manual name
    language      text,                 -- ISO 639 where known
    country_code  text,                 -- 'CR' when the source says so
    is_preferred  boolean NOT NULL DEFAULT false,
    source        text NOT NULL,        -- MPCR | TROPICOS | INAT_CR | INAT_ES | GBIF
    source_detail text,                 -- dataset, taxon id, Tomo/page
    fetched       date NOT NULL DEFAULT current_date,
    PRIMARY KEY (name_norm, species, source)
);
CREATE INDEX IF NOT EXISTS vernacular_name ON mpcr.vernacular (name_norm);
CREATE INDEX IF NOT EXISTS vernacular_species ON mpcr.vernacular (species);
"""


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------- providers

def from_inat(species: str, accepted: str | None = None) -> list[dict]:
    """iNaturalist names; the Costa Rica-preferred one is marked INAT_CR."""
    queries = [species] + ([accepted] if accepted and accepted != species else [])
    for query in queries:
        try:
            r = requests.get("https://api.inaturalist.org/v1/taxa",
                             params={"q": query, "rank": "species", "locale": "es",
                                     "preferred_place_id": INAT_CR_PLACE, "all_names": "true"},
                             headers=UA, timeout=30)
        except Exception:
            return []
        if not r.ok:
            continue
        hit = next((x for x in r.json().get("results", []) if x.get("name") == query), None)
        if not hit:
            continue
        out, preferred = [], hit.get("preferred_common_name")
        detail = f"iNat taxon {hit['id']}"
        if preferred:
            out.append({"vernacular": preferred, "language": "spa", "country_code": "CR",
                        "is_preferred": True, "source": "INAT_CR", "source_detail": detail})
        for n in hit.get("names") or []:
            if (n.get("lexicon") or "").lower() not in ("spanish", "espanol", "español", "es", "spa"):
                continue
            if preferred and norm(n.get("name")) == norm(preferred):
                continue
            out.append({"vernacular": n["name"], "language": "spa", "country_code": None,
                        "is_preferred": False, "source": "INAT_ES", "source_detail": detail})
        return out
    return []


_YEAR_CITE = re.compile(r"(?:1[6789]\d\d|20\d\d)(?:\[[^\]]*\])?\s*\.\s*")
_SYN_CUE = re.compile(r"[A-Z]\.\s|\bsensu\b|\bnon\b|\bvar\.|\bsubsp\.|\bnom\.|\bex\b|\d|&")
_AUTHOR_TAIL = re.compile(
    r"\b(Kunth|Lindl|Schltr|Pittier|Standl|Vahl|Jacq|Rudd|Urb|Mez|Donn|Sm|Griseb|DC|Benth|"
    r"Hook|Willd|Poepp|Endl|Miq|Sw|Poir|Rchb|Cogn|Klotzsch|Planch|Baker|Presl|Steud|Nees|"
    r"Aubl|Gaertn|Blume|Wedd|Seem|Oerst|Hemsl|Rose|Britton|Nash|Trel|Woodson|Croat|Grayum|"
    r"Hammel|Burger|Werff|G[oó]mez|Grimes|Pollard|Fl)\b\.?$")
_HYPHEN_SPLIT = re.compile(r"([a-záéíóúñ])-\s+([a-záéíóúñ])")
_SMALL_CAPS = re.compile(r"\b([A-ZÁÉÍÓÚÑ])\s+([A-ZÁÉÍÓÚÑ]{2,})")


def from_mpcr(ficha) -> list[dict]:
    """PROTOTYPE extractor for the names printed in the Manual header.

    The names sit after the last citation year, comma separated:
    "... Standl. Cortez , Cortez amarillo , Guayacan ."

    Measured on the full catalog: ~36% of species yield at least one name, and
    author surnames, flora abbreviations and synonyms still leak through. This is a
    placeholder so the MPCR layer exists end to end; the bachelor team's extractor,
    validated on a gold set (target precision >= 0.95), replaces this function
    without changing anything downstream.
    """
    head = (ficha.full_text or "").splitlines()[0] if ficha.full_text else ""
    head = _HYPHEN_SPLIT.sub(lambda m: m.group(1) + m.group(2), head)
    parts = _YEAR_CITE.split(head)
    if len(parts) < 2:
        return []
    # "A NTORCHA DE B RASIL" -> "Antorcha de Brasil" (small caps lost by the OCR)
    tail = _SMALL_CAPS.sub(lambda m: m.group(1) + m.group(2).lower(), parts[-1]).strip(" .")
    out = []
    for piece in re.split(r"\s*,\s*", tail):
        p = piece.strip(" .")
        if not p:
            continue
        if _SYN_CUE.search(p) or _AUTHOR_TAIL.search(p):
            # a real name can trail a synonym: "... non Kunth. Poro"
            p = re.split(r"(?<=[a-zA-Z])\.\s+", p)[-1].strip(" .")
            if not p or _SYN_CUE.search(p) or _AUTHOR_TAIL.search(p):
                continue
        p = re.sub(r"\s*[\[\]]\s*", " ", p).strip()
        if len(p) < 3 or len(p.split()) > 4 or not re.match(r"^[A-ZÁÉÍÓÚÑ]", p):
            continue
        out.append({"vernacular": p, "language": "spa", "country_code": "CR",
                    "is_preferred": False, "source": "MPCR",
                    "source_detail": f"Manual de Plantas de Costa Rica, "
                                     f"Tomo {ficha.volume}, p. {ficha.pages}"})
    return out


def from_gbif(taxon_key: int | None) -> list[dict]:
    if not taxon_key:
        return []
    from pygbif import species as gbif_species
    try:
        rows = gbif_species.name_usage(key=taxon_key, data="vernacularNames").get("results", [])
    except Exception:
        return []
    out, seen = [], set()
    for v in rows:
        if (v.get("language") or "").lower() not in ("spa", "es"):
            continue
        n = v.get("vernacularName")
        if not n or norm(n) in seen:
            continue
        seen.add(norm(n))
        out.append({"vernacular": n, "language": "spa", "country_code": v.get("country"),
                    "is_preferred": False, "source": "GBIF",
                    "source_detail": (v.get("source") or "")[:200]})
    return out


def from_tropicos(species: str) -> list[dict]:
    """Missouri Botanical Garden. Needs TROPICOS_API_KEY: a website account is not
    enough, unkeyed calls are rejected with "You are not allowed to make this request".
    MOBOT publishes the Manual, so these names should mirror MPCR."""
    if not TROPICOS_KEY:
        return []
    try:
        s = requests.get("https://services.tropicos.org/Name/Search",
                         params={"name": species, "type": "exact", "format": "json",
                                 "apikey": TROPICOS_KEY}, headers=UA, timeout=30).json()
        nid = s[0].get("NameId") if isinstance(s, list) and s else None
        if not nid:
            return []
        rows = requests.get(f"https://services.tropicos.org/Name/{nid}/CommonNames",
                            params={"format": "json", "apikey": TROPICOS_KEY},
                            headers=UA, timeout=30).json()
    except Exception:
        return []
    out = []
    for v in rows if isinstance(rows, list) else []:
        n = v.get("CommonName")
        if not n:
            continue
        country = (v.get("Country") or "")
        out.append({"vernacular": n, "language": (v.get("Language") or "")[:3].lower() or None,
                    "country_code": "CR" if country.lower().startswith("costa") else None,
                    "is_preferred": False, "source": "TROPICOS",
                    "source_detail": f"Tropicos name {nid}"})
    return out


# ----------------------------------------------------------------------- store

def fetch(species_list: list[str], verbose: bool = True, mpcr: bool = True) -> dict:
    tx = taxa.load()
    conn = pg_store.connect()
    cur = conn.cursor()
    cur.execute(_SCHEMA)
    conn.commit()
    stats = {"species": 0, "rows": 0, "with_cr_preferred": 0, "no_names": 0}
    for i, sp in enumerate(species_list, 1):
        t = tx.get(sp) or {}
        rows = (from_inat(sp, t.get("accepted_name")) + from_gbif(t.get("taxon_key"))
                + from_tropicos(sp))
        if mpcr:
            from ..store import local_store
            from .. import config as _cfg
            f = local_store.get(local_store.connect(_cfg.SQLITE_PATH), sp.replace(" ", "_"))
            if f:
                rows = from_mpcr(f) + rows
        for r in rows:
            cur.execute("""INSERT INTO mpcr.vernacular
                (name_norm, vernacular, species, language, country_code, is_preferred,
                 source, source_detail)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (norm(r["vernacular"]), r["vernacular"], sp, r["language"],
                         r["country_code"], r["is_preferred"], r["source"], r["source_detail"]))
        conn.commit()
        stats["species"] += 1
        stats["rows"] += len(rows)
        stats["no_names"] += not rows
        pref = next((r["vernacular"] for r in rows if r["source"] == "INAT_CR"), None)
        stats["with_cr_preferred"] += bool(pref)
        if verbose:
            print(f"  [{i:3}/{len(species_list)}] {sp:34} {len(rows):3} names | "
                  f"CR-preferred: {pref or '-'}", flush=True)
        time.sleep(1.0)            # iNaturalist asks for about one request per second
    conn.close()
    return stats


# -------------------------------------------------------------------- resolver

def resolve(name: str, *, conn=None, region: str | None = None,
            elev: int | None = None, limit: int = 8) -> list[dict]:
    """Ranked candidate species for a common name, with provenance.

    Ranking: source trust first (the Manual and the Costa Rica-preferred iNaturalist
    name beat a generic Spanish name), then how often the species is actually recorded
    in Costa Rica. Constraints taken from the question (region, elevation) filter the
    candidates when that leaves any. Never returns a single species silently.
    """
    own = conn is None
    conn = conn or pg_store.connect()
    sql = """
        SELECT v.species,
               bool_or(v.is_preferred)            AS preferred,
               array_agg(DISTINCT v.source)       AS sources,
               max(v.vernacular)                  AS as_published,
               f.ficha->>'family'                 AS family,
               (f.ficha->>'elev_min')::int        AS elev_min,
               (f.ficha->>'elev_max')::int        AS elev_max,
               f.regions                          AS regions,
               e.n_records                        AS n_records
        FROM mpcr.vernacular v
        JOIN mpcr.fichas f ON f.species = v.species
        LEFT JOIN mpcr.species_evidence e ON e.species = v.species
        WHERE {clause}
        GROUP BY v.species, f.ficha, f.regions, e.n_records"""
    try:
        with conn.cursor() as cur:
            # Accent-sensitive first: "poró" (Erythrina, Costa Rica) and "poro" (leek,
            # Mexico) are different plants, so the unaccented match is only a fallback.
            cur.execute(sql.format(clause="lower(v.vernacular) = lower(%s)"), (name.strip(),))
            cols = [c.name for c in cur.description]
            cands = [dict(zip(cols, r), match="exact") for r in cur.fetchall()]
            if not cands:
                cur.execute(sql.format(clause="v.name_norm = %s"), (norm(name),))
                cands = [dict(zip(cols, r), match="unaccented") for r in cur.fetchall()]
    finally:
        if own:
            conn.close()

    for c in cands:
        c["trust"] = min(TRUST.get(s, 9) for s in c["sources"])
    if region:
        kept = [c for c in cands if region in (c["regions"] or [])]
        cands = kept or cands
    if elev is not None:
        kept = [c for c in cands if c["elev_min"] is not None
                and c["elev_min"] - 200 <= elev <= (c["elev_max"] or 0) + 200]
        cands = kept or cands
    cands.sort(key=lambda c: (c["trust"], -(c["n_records"] or 0)))
    return cands[:limit]


def resolve_decision(name: str, **kw) -> dict:
    """resolve() plus an explicit decision the caller must act on.

      none        the name is not in the store -> say so, do not guess
      unique      one candidate -> proceed, but name the species in the answer
      preferred   several candidates, but exactly one comes from a Costa Rica-specific
                  authority (Manual, Tropicos CR, iNaturalist CR-preferred) AND is at
                  least as well recorded here as the others -> proceed with it, saying
                  which and why. Both conditions are needed: iNaturalist marks "Ceiba"
                  as the Costa Rica name of Spirotheca rosea (76 records) because
                  Ceiba pentandra (200 records) is filed under the indigenous "Shkuli",
                  so trust alone would pick the less likely plant.
      ambiguous   several equally plausible -> ask back or answer for the group; never
                  pick silently

    Candidates are always species of THIS catalog: a global name index answers a
    different question (Tropicos returns Allium porrum, the leek, for "Poro").
    """
    cands = resolve(name, **kw)
    if not cands:
        return {"decision": "none", "name": name, "candidates": []}
    if len(cands) == 1:
        return {"decision": "unique", "name": name, "species": cands[0]["species"],
                "candidates": cands}
    best, rest = cands[0], cands[1:]
    best_records = best["n_records"] or 0
    if (best["trust"] <= TRUST["INAT_CR"]
            and all(c["trust"] > best["trust"] for c in rest)
            and best_records >= max((c["n_records"] or 0) for c in rest)):
        return {"decision": "preferred", "name": name, "species": best["species"],
                "candidates": cands}
    return {"decision": "ambiguous", "name": name, "candidates": cands}


def report(conn=None) -> None:
    conn = conn or pg_store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*), count(DISTINCT name_norm), count(DISTINCT species) "
                    "FROM mpcr.vernacular")
        rows, names, sp = cur.fetchone()
        print(f"vernacular: {rows} rows, {names} distinct names, {sp} species")
        cur.execute("SELECT source, count(*) FROM mpcr.vernacular GROUP BY 1 ORDER BY 2 DESC")
        print("  by source:", dict(cur.fetchall()))
        cur.execute("""SELECT name_norm, count(DISTINCT species) n, array_agg(DISTINCT species) sp
                       FROM mpcr.vernacular GROUP BY 1 HAVING count(DISTINCT species) > 1
                       ORDER BY n DESC, 1 LIMIT 12""")
        amb = cur.fetchall()
        print(f"  ambiguous names (within the fetched species): {len(amb)}")
        for n, k, sps in amb:
            print(f"    {n:24} -> {k}: {', '.join(sorted(sps)[:4])}")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--pilot", action="store_true", help="the 18 pilot species")
    f.add_argument("--expert", action="store_true", help="the 21 expert-validated species")
    f.add_argument("--species-file")
    r = sub.add_parser("resolve")
    r.add_argument("name")
    r.add_argument("--region")
    r.add_argument("--elev", type=int)
    sub.add_parser("report")
    a = ap.parse_args()

    if a.cmd == "fetch":
        names: list[str] = []
        if a.pilot:
            s = json.loads((Path(__file__).resolve().parents[2] / "benchmark" / "data"
                            / "species_sample.json").read_text(encoding="utf-8"))
            names += [x["species"] for x in s["pilot"]]
        if a.expert:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from utils.compare_expert_maps import SPECIES
            names += list(SPECIES)
        if a.species_file:
            names += [l.strip() for l in Path(a.species_file).read_text(encoding="utf-8").splitlines()
                      if l.strip()]
        seen: set[str] = set()
        names = [n for n in names if not (n in seen or seen.add(n))]
        print(json.dumps(fetch(names), ensure_ascii=False))
    elif a.cmd == "resolve":
        for c in resolve(a.name, region=a.region, elev=a.elev):
            print(f"  {c['species']:32} {(c['family'] or ''):18} "
                  f"[{c['elev_min']}-{c['elev_max']} m] CR={c['n_records'] or 0:5} "
                  f"via {','.join(sorted(c['sources']))}{'  (CR-preferred)' if c['preferred'] else ''}")
    else:
        report()
