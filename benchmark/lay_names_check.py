"""Do the free models' maps score near chance because the parser misses their place names?

The map parser (mapcompare.from_text) reads the Manual's region names. Asked freely, a
model may instead name a province ("Guanacaste", "Limón"), an informal zone ("zona
norte", "Pacífico Central") or a lowland ("tierras bajas del Caribe"), which the parser
reads as nothing. This check adds a gazetteer of such lay names, each mapped to the
botanical regions it covers, and rescores the free models' maps on all records of the
300-species sample exactly as null_models.py does ("map_all"). It is generous to the
models: a province is read as every region it touches.

The gazetteer below is the authors' judgement of which regions each name covers.

Run:  python -m benchmark.lay_names_check
"""
from __future__ import annotations

import collections
import json
import re
import statistics as st

import numpy as np

from . import map_refine as R
from . import mapcompare as M
from . import null_models as N

PACIFIC_S = ["Península de Osa - Golfito", "Llanuras del Diquís", "Coto Colorado",
             "Fila Costeña Sur", "Valle del General", "Valle del Coto Brus", "Punta Burica"]
LAY = [
    (r"\bGuanacaste\b(?!\s*(?:y|,)?\s*cordillera)", ["Llanuras de Guanacaste", "Cordillera de Guanacaste",
                                                    "Península de Nicoya", "Cordillera de Tilarán"]),
    (r"\bLim[oó]n\b", ["Llanuras de Tortuguero / Santa Clara", "Baja Talamanca",
                       "Cordillera de Talamanca"]),
    (r"\bPuntarenas\b", ["Península de Nicoya", "Tárcoles - Térraba", "Fila Costeña Norte"]
     + PACIFIC_S),
    (r"\bAlajuela\b", ["Llanuras de San Carlos", "LLanura de los Guatusos",
                       "Valle Central Occidental", "Cordillera Central"]),
    (r"\bHeredia\b", ["Llanuras de Tortuguero / Santa Clara", "Cordillera Central",
                      "Valle Central Oriental"]),
    (r"\bCartago\b", ["Valle Central Oriental", "Cordillera de Talamanca"]),
    (r"\bSan Jos[eé]\b", ["Valle Central Occidental", "Valle Central Oriental",
                          "Puriscal - Los Santos", "Valle del General"]),
    (r"zona norte|norte del pa[ií]s|Sarapiqu[ií]|San Carlos",
     ["Llanuras de San Carlos", "LLanura de los Guatusos", "Llanuras de Tortuguero / Santa Clara"]),
    (r"(tierras bajas|llanuras?|bajura)s? (costeras? )?(del|de la vertiente del) (Caribe|Atl[aá]ntico)",
     ["Llanuras de Tortuguero / Santa Clara", "Llanuras de San Carlos",
      "LLanura de los Guatusos", "Baja Talamanca"]),
    (r"Pac[ií]fico Norte|Pac[ií]fico seco|noroeste", ["Llanuras de Guanacaste", "Península de Nicoya"]),
    (r"Pac[ií]fico Central", ["Tárcoles - Térraba", "Turrubares", "Fila Costeña Norte"]),
    (r"Pac[ií]fico Sur|zona sur|sur del pa[ií]s", PACIFIC_S),
    (r"(tierras bajas|llanuras?) (costeras? )?del Pac[ií]fico",
     ["Llanuras de Guanacaste", "Tárcoles - Térraba", "Llanuras del Diquís", "Coto Colorado"]),
]


def from_text_lay(text: str) -> dict:
    d = M.from_text(text)
    if d["everywhere"]:
        return d
    extra = set()
    for pat, regs in LAY:
        if re.search(pat, text, re.I):
            extra.update(regs)
    d = dict(d)
    d["regions"] = sorted(set(d["regions"]) | extra)
    return d


def main() -> None:
    from mpcr_rag.store import pg_store
    N.background()
    texts = collections.defaultdict(dict)
    for r in M._jsonl(M.DATA / "answers_mapq.jsonl"):
        if not r.get("error") and r.get("answer_text"):
            texts[r["species"]][r["model"]] = r["answer_text"]
    rows = []
    conn = pg_store.connect()
    with conn.cursor() as cur:
        for sp in N._species():
            cur.execute("""SELECT o.lon, o.lat, o.region, o.vertiente, o.elev_dem
                           FROM mpcr.occurrences o
                           JOIN mpcr.species_evidence e ON e.taxon_key = o.species_key
                           WHERE e.species = %s""", (sp,))
            got = cur.fetchall()
            if len(got) < M.MIN_RECORDS:
                continue
            lonlat = [(r[0], r[1]) for r in got]
            recs = [(r[2], None if r[3] is None else ("Caribe" if "arib" in r[3] else "Pacífico"), r[4])
                    for r in got]
            pix = R.record_pixels(lonlat)
            for model, text in texts.get(sp, {}).items():
                for parser, fn in (("published", M.from_text), ("with lay names", from_text_lay)):
                    ing = fn(text)
                    mk = M.mask(ing)
                    if mk is None:
                        continue
                    r = N.row(sp, parser, model, N._inside_map(ing, recs), lonlat, pix, [mk])
                    rows.append(r)
    conn.close()
    out = {}
    for model in ("paid", "open"):
        for parser in ("published", "with lay names"):
            s = [r for r in rows if r["source"] == model and r["analysis"] == parser and r["area"] > 0]
            k = f"{model}, {parser}"
            out[k] = {"maps": len(s),
                      "median_capture": st.median(r["capture"] for r in s),
                      "median_area": st.median(r["area"] for r in s),
                      "median_lift": st.median(r["capture"] / r["area"] for r in s),
                      "median_site_lift": st.median(r["capture"] / r["p_sites"] for r in s if r["p_sites"] > 0),
                      "beats_blocks": sum(r["boot_q05"] > r["p_sites"] for r in s)}
            v = out[k]
            print(f"{k:28s} maps {v['maps']:3d}  capture {100*v['median_capture']:5.1f}%  area "
                  f"{100*v['median_area']:5.1f}%  lift {v['median_lift']:.2f}  site lift "
                  f"{v['median_site_lift']:.2f}  beat chance (blocks) {v['beats_blocks']}")
    (M.DATA / "lay_names_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
