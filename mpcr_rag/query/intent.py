"""Natural-language → structured query intent.

Turns a free Spanish question ("arbustos de bosque nuboso sobre 2000 m en Talamanca")
into the structured constraints that ``retriever.pattern_b`` consumes. An LLM maps
the phrasing onto the catalog's CONTROLLED VOCABULARY (the distinct habits, forest
types, vertientes, regions, families actually present), so the resulting Pinecone
metadata filter uses exact stored values. Whatever isn't a structured field becomes
``semantic_text`` for the vector search.
"""
from __future__ import annotations

import json
import os

import requests

from .. import config
from ..store import local_store
from ..schema import Ficha

_HABITS = ["árbol", "arbusto", "hierba", "bejuco", "epífita", "hemiepífita",
           "palma", "terrestre", "acuática", "sufrútice", "parásita"]
_MONTHS = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
           "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10,
           "noviembre": 11, "diciembre": 12}


def load_vocab(conn=None) -> dict:
    """Distinct controlled values present in the catalog."""
    conn = conn or local_store.connect(config.SQLITE_PATH)
    rows = [Ficha.from_json(r["ficha_json"])
            for r in conn.execute("SELECT ficha_json FROM fichas")]
    forests, regions, families = set(), set(), set()
    for f in rows:
        forests.update(f.forest_types)
        regions.update(f.regions)
        families.add(f.family)
    return {
        "habits": _HABITS,
        "forest_types": sorted(forests),
        "vertientes": ["Caribe", "Pacífico"],
        "regions": sorted(regions),
        "families": sorted(families),
    }


_SYSTEM = """Eres un parser que convierte preguntas en español sobre plantas de Costa
Rica en un filtro estructurado JSON. Usa ÚNICAMENTE valores de las listas permitidas
(copia exactamente; si no aplica, null). Reglas de elevación (metros):
- "entre X y Y" -> elev_lo=X, elev_hi=Y
- "sobre/arriba de/más de/encima de X" -> elev_lo=X, elev_hi=null
- "bajo/debajo de/menos de X" -> elev_lo=null, elev_hi=X
- "a X m" o "alrededor de X" -> elev_lo=X-150, elev_hi=X+150
endemic=true si la pregunta dice "endémica(s)/endémico(s)/solo de Costa Rica/exclusivas de CR".
flowering_month = número de mes si dice "florece(n) en <mes>" (enero=1 … diciembre=12).
vertiente: "del Caribe/caribeña/atlántica" -> Caribe; "del Pacífico/pacífica" -> Pacífico.
semantic_text = la parte descriptiva libre que NO es un campo (hábitat, olor, usos),
o "" si todo quedó en campos. Responde SOLO el JSON."""


def parse_intent(question: str, vocab: dict | None = None) -> dict:
    """LLM-map a question to {habit, elev_lo, elev_hi, vertiente, region,
    forest_type, family, flowering_month, endemic, semantic_text}."""
    vocab = vocab or load_vocab()
    schema = {
        "habit": f"uno de {vocab['habits']} o null",
        "forest_type": f"uno de {vocab['forest_types']} o null",
        "vertiente": "Caribe, Pacífico o null",
        "region": f"uno de {vocab['regions']} o null",
        "family": f"una de {vocab['families']} o null",
        "elev_lo": "int o null", "elev_hi": "int o null",
        "flowering_month": "1-12 o null", "endemic": "true/false/null",
        "semantic_text": "string",
    }
    prompt = (f"Listas permitidas y formato:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
              f"Pregunta: {question}")
    key = os.environ.get("OPENROUTER_API_KEY", "")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": config.ENRICH_MODEL, "temperature": 0,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": _SYSTEM},
                           {"role": "user", "content": prompt}]},
        timeout=30,
    )
    data = json.loads(resp.json()["choices"][0]["message"]["content"])
    return _validate(data, vocab)


def _validate(d: dict, vocab: dict) -> dict:
    """Keep only values that exist in the controlled vocabulary."""
    def pick(key, allowed):
        v = d.get(key)
        return v if v in allowed else None

    out = {
        "habit": pick("habit", vocab["habits"]),
        "forest_type": pick("forest_type", vocab["forest_types"]),
        "vertiente": pick("vertiente", vocab["vertientes"]),
        "region": pick("region", vocab["regions"]),
        "family": pick("family", vocab["families"]),
        "elev_lo": d.get("elev_lo") if isinstance(d.get("elev_lo"), int) else None,
        "elev_hi": d.get("elev_hi") if isinstance(d.get("elev_hi"), int) else None,
        "flowering_month": d.get("flowering_month")
            if isinstance(d.get("flowering_month"), int) else None,
        "endemic": d.get("endemic") if isinstance(d.get("endemic"), bool) else None,
        "semantic_text": d.get("semantic_text") or "",
    }
    return out


def ask(question: str, *, top_k: int = 15, conn=None, index=None):
    """End-to-end: NL question → parsed intent + ranked (Ficha, score) results."""
    from . import retriever as R
    from ..store import pinecone_client as pc

    conn = conn or local_store.connect(config.SQLITE_PATH)
    index = index or pc.ensure_index()
    intent = parse_intent(question)
    constraints = {k: v for k, v in intent.items()
                   if k != "semantic_text" and v is not None}
    query_text = intent["semantic_text"] or question
    results = R.pattern_b(query_text, top_k=top_k, conn=conn, index=index, **constraints)
    return intent, results


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "arbustos endémicos de bosque nuboso sobre 2000 m en Talamanca"
    intent, results = ask(q)
    print(f"PREGUNTA: {q}\nINTENT: {json.dumps(intent, ensure_ascii=False)}\n")
    print(f"{len(results)} resultados:")
    for f, s in results[:12]:
        print(f"  rel={s:.3f}  {f.species:30} [{f.family}] [{f.elev_min}-{f.elev_max}m] {f.habits}")
