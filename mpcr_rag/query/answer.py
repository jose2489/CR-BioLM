"""End-to-end answer layer — the system figure.

A natural-language question → (1) routed to species lookup or geospatial filter,
(2) a *grounded* text answer composed only from the retrieved Manual fichas, with
page citations, (3) a GBIF occurrence map of the matching species filtered by the
query constraints, and (4) the collection-bias disclaimer + 'insufficient evidence'
flags. Renders a Markdown report (map embedded) for presentation.
"""
from __future__ import annotations

import json
import os
import re
import textwrap

import requests

from .. import config
from ..schema import Ficha
from ..store import local_store, pinecone_client as pc
from . import gbif_map
from . import retriever as R
from .intent import parse_intent

_ANSWERS = config.DATA_DIR / "answers"
_ANSWERS.mkdir(exist_ok=True)

_DISCLAIMER = (
    "*Los puntos son registros de GBIF y reflejan **esfuerzo de colecta**, no abundancia; "
    "la ausencia de puntos no implica ausencia de la especie. El rango de referencia "
    "proviene del Manual de Plantas de Costa Rica. Especies con <5 registros se marcan "
    "como evidencia insuficiente.*"
)


def _detect_species(question: str, conn) -> str | None:
    """If the question names a catalog species (binomial), return it."""
    for m in re.finditer(r"\b([A-Z][a-záéíóúñ]+)\s+([a-záéíóúñ]{3,})\b", question):
        cand = f"{m.group(1)} {m.group(2)}"
        if local_store.get(conn, cand.replace(" ", "_")):
            return cand
    return None


def _compose(question: str, results: list[tuple[Ficha, float]]) -> str:
    """LLM-compose a grounded answer using ONLY the retrieved fichas (with cites)."""
    ctx = "\n".join(
        f"- {f.species} (Tomo {f.volume}, p.{f.pages}): {f.distribution_paragraph}"
        for f, _ in results[:12])
    sys = ("Eres un botánico de Costa Rica. Responde la pregunta USANDO ÚNICAMENTE las "
           "fichas dadas; no agregues datos externos. Cita la especie y (Tomo, página) "
           "para cada afirmación. Sé conciso. Si no hay fichas, dilo.")
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY','')}",
                     "Content-Type": "application/json"},
            json={"model": config.ENRICH_MODEL, "temperature": 0,
                  "messages": [{"role": "system", "content": sys},
                               {"role": "user",
                                "content": f"Fichas:\n{ctx}\n\nPregunta: {question}"}]},
            timeout=40)
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(síntesis no disponible: {e})"


def answer(question: str, *, top_k: int = 12, conn=None, index=None) -> dict:
    conn = conn or local_store.connect(config.SQLITE_PATH)
    index = index or pc.ensure_index()

    sp = _detect_species(question, conn)
    if sp:
        f = local_store.get(conn, sp.replace(" ", "_"))
        results, constraints, mode = [(f, 1.0)], {}, "A (especie)"
    else:
        intent = parse_intent(question)
        constraints = {k: v for k, v in intent.items()
                       if k != "semantic_text" and v is not None}
        results = R.pattern_b(intent.get("semantic_text") or question,
                              top_k=top_k, conn=conn, index=index, **constraints)
        mode = "B (geoespacial)"

    map_c = {k: constraints.get(k) for k in ("elev_lo", "elev_hi", "vertiente", "region")}
    map_path, n_pts = gbif_map.most_likely_map(results, query_text=question, **map_c)
    text = _compose(question, results)
    return {"question": question, "mode": mode, "constraints": constraints,
            "results": results, "text": text, "map_path": map_path, "n_pts": n_pts}


def render_markdown(a: dict) -> str:
    rel = a["map_path"].name
    lines = [f"# {a['question']}", "",
             f"**Modo:** {a['mode']}  ·  **Filtro:** `{json.dumps(a['constraints'], ensure_ascii=False)}`", "",
             "## Respuesta (grounded)", "", a["text"], "",
             "## Especies y evidencia", "",
             "| especie | familia | elev (m) | vertiente | Tomo·pág | GBIF n |",
             "|---|---|---|---|---|---|"]
    for f, _ in a["results"][:12]:
        n = len(gbif_map.get_points(f.species))
        flag = " ⚠️<5" if n < 5 else ""
        lines.append(f"| *{f.species}* | {f.family} | {f.elev_min}–{f.elev_max} | "
                     f"{', '.join(f.vertientes) or '–'} | {f.volume}·{f.pages} | {n}{flag} |")
    lines += ["", f"## Mapa GBIF ({a['n_pts']} registros filtrados)", "",
              f"![mapa](../maps/{rel})", "", "---", _DISCLAIMER]
    md = "\n".join(lines)
    out = _ANSWERS / f"answer_{abs(hash(a['question'])) % 10**8}.md"
    out.write_text(md, encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "arbustos endémicos de bosque nuboso sobre 2000 m en Talamanca"
    a = answer(q)
    path = render_markdown(a)
    print(f"\nPREGUNTA: {q}\nMODO: {a['mode']}  |  {len(a['results'])} especies  |  "
          f"{a['n_pts']} registros GBIF\n")
    print(textwrap.shorten(a["text"], 400))
    print(f"\nreporte → {path}\nmapa    → {a['map_path']}")
