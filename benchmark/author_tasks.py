"""The validation work that needs a careful reader but not a botanist, in one workbook.

Three sheets, each answering a reviewer's request with labels the checker never sees:

  1. Checker (60 answers). For each real answer the reader gives the verdict a careful
     reader would give against the stated key, whether the answer declines, and its
     stance on a yes-or-no question. The last two validate the two things the language
     model in the checker extracts, separately from the rules that turn them into a
     verdict. 37 answers come from the first validation sheet (its three rows the author
     had already read and commented on are dropped, being no longer blind); 23 new ones
     are drawn from the confirmatory set, from the types whose verdict turns on the
     stance, equally across the stances the checker extracted.

  2. Contradicted claims (50). Claims the checker marked as contradicted in the grounded
     answers (25 from each generator, confirmatory set), with the answer, the key and
     the species' record. The reader says whose error it is: the checker's, the key's or
     record's, or the answer's. No suggestion is shown, so the labels are not anchored.

  3. The use pattern (100 entries). Whether the Manual entry says the plant is edible or
     medicinal, against what the key's pattern found: 30 entries the pattern flagged
     (its precision), 50 it did not flag but that use words such as "consumido" or
     "medicina popular" (where misses are likeliest), and 20 drawn at random from the
     rest.

Seeds recorded. Checker verdicts, stances and pattern results go to author_tasks_key.json.

Run:  python -m benchmark.author_tasks
Writes tareas_validacion_jose.xlsx and author_tasks_key.json
"""
from __future__ import annotations

import collections
import io
import json
import random
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

from . import map_preference_sheet as MP
from .expert_sheet import TIPO
from .questions import _EDIBLE_MED_CUE
from .validation_sheet import key_text, MESES

DATA = Path(__file__).resolve().parent / "data"
SEED = "20261025-author"
DROPPED = {"V06", "V07", "V09"}            # read and commented on: no longer blind
STANCE_TYPES = ("park_confirmed", "park_possible", "park_unlikely", "cultivation",
                "endemism", "outside_record")
GROUNDED = ("D", "F")
VERDICTS = ("correcto", "parcialmente correcto", "exagerado (más certeza de la debida)",
            "subestimado (menos certeza de la debida)", "incorrecto", "se abstiene",
            "afirma algo que el registro no contiene")
# Words that point at food or medicine without the key's own vocabulary.
_BROAD = re.compile(
    r"\b(consumid[oa]s?|consum[eo]n?|se comen?|aliment\w*|sabor\w*|dulces?|bebida|fresco|"
    r"t[ée]|medicina\w*|remedios?|cura\w*|dolor\w*|fiebre|enfermedad\w*|tratar|usad[oa]s?|"
    r"se usa|utiliza\w*|us[oa]s?|venen\w*|t[óo]xic\w*)\b", re.I)


def _jsonl(name: str) -> list[dict]:
    return [json.loads(l) for l in io.open(DATA / name, encoding="utf-8") if l.strip()]


def _confirmatory() -> set[str]:
    return (set(json.loads((DATA / "species_tranche1.json").read_text(encoding="utf-8")))
            | set(json.loads((DATA / "species_tranche2.json").read_text(encoding="utf-8"))))


def _record_summary(f) -> str:
    """The species' record in a few lines, for judging a claim."""
    out = []
    if f.elev_min is not None:
        s = f"Altitud: {f.elev_min}-{f.elev_max} m"
        if f.elev_outlier_min is not None or f.elev_outlier_max is not None:
            s += (f" (atípicos {f.elev_outlier_min if f.elev_outlier_min is not None else f.elev_min}"
                  f"-{f.elev_outlier_max if f.elev_outlier_max is not None else f.elev_max} m)")
        out.append(s)
    if f.vertientes:
        out.append("Vertientes: " + ", ".join(f.vertientes))
    if f.regions:
        out.append("Regiones (Manual): " + ", ".join(f.regions))
    if f.flowering_months:
        out.append("Floración: " + ", ".join(MESES[m - 1] for m in f.flowering_months))
    if f.habits:
        out.append("Hábito: " + ", ".join(f.habits))
    out.append("Endémica de CR: " + ("sí" if f.endemic_cr else "no indicada"))
    return "\n".join(out) + f"\n\nDistribución (Manual): {f.distribution_paragraph}"


# ------------------------------------------------------------------ sheet 1, checker

def checker_items() -> list[dict]:
    qs = {q["qid"]: q for q in _jsonl("questions_main.jsonl")}
    ans = {(a["qid"], a["arm"]): a["answer_text"] for a in _jsonl("answers_main.jsonl")
           if not a.get("error") and a.get("answer_text")}
    ver = {(v["qid"], v["arm"]): v for v in _jsonl("verdicts_main.jsonl")
           if not v.get("error") and v.get("direct")}
    old = [k for k in _jsonl("checker_validation_key.jsonl") if k["id"] not in DROPPED]
    items = [{"qid": k["qid"], "arm": k["arm"], "origin": f"first sheet {k['id']}"} for k in old]
    taken = {(i["qid"], i["arm"]) for i in items}

    conf = _confirmatory()
    rng = random.Random(SEED + "-checker")
    by_stance = collections.defaultdict(list)
    for (q, a), v in sorted(ver.items()):
        if (qs.get(q, {}).get("species") in conf and qs[q]["qtype"] in STANCE_TYPES
                and a not in ("G",) and (q, a) in ans and (q, a) not in taken
                and v["direct"] not in ("not_scored",)):
            by_stance[v.get("stance") or "none"].append((q, a))
    stances = ("yes", "possible", "no", "uncertain")
    need = 60 - len(items)
    per = need // len(stances)
    for i, s in enumerate(stances):
        pool = by_stance[s]
        rng.shuffle(pool)
        k = per + (1 if i < need - per * len(stances) else 0)
        items += [{"qid": q, "arm": a, "origin": "new"} for q, a in pool[:k]]
    rng.shuffle(items)
    for n, it in enumerate(items, 1):
        q, v = qs[it["qid"]], ver[(it["qid"], it["arm"])]
        it.update({"id": f"C{n:02d}", "qtype": q["qtype"], "question": q["question"],
                   "key": key_text(q), "answer": ans[(it["qid"], it["arm"])],
                   "checker": v["direct"], "stance": v.get("stance"),
                   "abstains": v.get("abstains")})
    return items


# ------------------------------------------------------------- sheet 2, contradicted

def claim_items(cur) -> list[dict]:
    from mpcr_rag.schema import Ficha
    conf = _confirmatory()
    qs = {q["qid"]: q for q in _jsonl("questions_main.jsonl")}
    ans = {(a["qid"], a["arm"]): a["answer_text"] for a in _jsonl("answers_main.jsonl")
           if not a.get("error") and a.get("answer_text")}
    rows = {}
    for v in _jsonl("verdicts_main.jsonl"):
        if v["arm"] in GROUNDED and v["species"] in conf and not v.get("error"):
            rows[(v["qid"], v["arm"])] = v
    rng = random.Random(SEED + "-claims")
    out = []
    for arm in GROUNDED:
        pool = [(q, a, i, c) for (q, a), v in sorted(rows.items()) if a == arm
                for i, c in enumerate(v.get("claims") or []) if c.get("verdict") == "contradicted"]
        rng.shuffle(pool)
        # One claim per answer, so a single long answer cannot fill the sample.
        seen = set()
        for q, a, i, c in pool:
            if (q, a) in seen:
                continue
            seen.add((q, a))
            out.append({"qid": q, "arm": a, "claim_index": i, "claim": c["text"],
                        "claim_type": c.get("type")})
            if sum(o["arm"] == arm for o in out) == 25:
                break
    rng.shuffle(out)
    fichas = {}
    for n, it in enumerate(out, 1):
        sp = qs[it["qid"]]["species"]
        if sp not in fichas:
            cur.execute("SELECT ficha FROM mpcr.fichas WHERE species = %s", (sp,))
            fichas[sp] = Ficha(**cur.fetchone()[0])
        it.update({"id": f"K{n:02d}", "species": sp, "qtype": qs[it["qid"]]["qtype"],
                   "question": qs[it["qid"]]["question"], "key": key_text(qs[it["qid"]]),
                   "answer": ans[(it["qid"], it["arm"])],
                   "record": _record_summary(fichas[sp])})
    return out


# ------------------------------------------------------------------ sheet 3, uses

def _sentences(text: str, rx) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text or "") if rx.search(s)]


def use_items(cur) -> list[dict]:
    cur.execute("SELECT species, ficha->>'discussion', ficha->>'distribution_paragraph' "
                "FROM mpcr.fichas ORDER BY species")
    flagged, broad, rest = [], [], []
    for sp, disc, dist in cur.fetchall():
        text = f"{disc or ''} {dist or ''}"
        row = {"species": sp, "text": text.strip()}
        if _EDIBLE_MED_CUE.search(text):
            flagged.append(row)
        elif _BROAD.search(text):
            broad.append(row)
        else:
            rest.append(row)
    rng = random.Random(SEED + "-uses")
    picked = ([dict(r, stratum="flagged") for r in rng.sample(flagged, 30)]
              + [dict(r, stratum="broad_unflagged") for r in rng.sample(broad, 50)]
              + [dict(r, stratum="random_unflagged") for r in rng.sample(rest, 20)])
    rng.shuffle(picked)
    for n, it in enumerate(picked, 1):
        hits = _sentences(it["text"], _EDIBLE_MED_CUE) + _sentences(it["text"], _BROAD)
        it.update({"id": f"U{n:03d}",
                   "sentences": " | ".join(dict.fromkeys(hits)) or "(ninguna oración menciona usos)"})
    counts = {"flagged": len(flagged), "broad_unflagged": len(broad),
              "random_unflagged": len(rest)}
    return picked, counts


# -------------------------------------------------------------------- the workbook

INSTRUCTIONS = [
    ("Validación: tres tareas", True, 14),
    ("", False, 10),
    ("Ninguna requiere conocimiento botánico: se trata de leer con cuidado contra lo que dice "
     "el registro. Solo llene las celdas amarillas. Lo que el verificador dijo de cada caso "
     "no está en el libro, para que las etiquetas sean ciegas.", False, 10),
    ("", False, 10),
    ("1. Verificador (60 respuestas, ~1,5 h)", True, 11),
    ("Para cada respuesta: (a) el veredicto que daría un lector cuidadoso frente a lo que "
     "respalda el registro; (b) si la respuesta se niega a responder o dice no tener base; "
     "(c) su postura, en las preguntas de sí o no: sí, posible, no, incierto, o «no aplica». "
     "Juzgue la respuesta contra el registro, no contra lo que usted sabe de la planta.", False, 10),
    ("", False, 10),
    ("2. Afirmaciones contradichas (50, ~1 h)", True, 11),
    ("El verificador marcó cada afirmación como contradicha por el registro. Diga de quién es "
     "el error: del verificador (la afirmación en realidad no contradice el registro, o fue "
     "mal leída), de la clave o del registro (el registro está mal o incompleto), o de la "
     "respuesta (la afirmación sí es falsa según el registro).", False, 10),
    ("", False, 10),
    ("3. Patrón de usos (100 entradas, ~45 min)", True, 11),
    ("Para cada entrada del Manual: ¿dice que la planta es comestible o medicinal (incluido "
     "uso medicinal popular)? Se muestran primero las oraciones que mencionan usos y, al lado, "
     "el texto completo por si hace falta.", False, 10),
]


def _sheet(wb, title, heads, rows, fill_from, validations, height=150):
    ws = wb.create_sheet(title)
    for col, (t, w) in enumerate(heads, 1):
        c = ws.cell(row=1, column=col, value=t)
        c.font = Font(name=MP.FONT, bold=True, color="FFFFFF", size=10)
        c.fill = MP.HEAD
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        ws.column_dimensions[c.column_letter].width = w
    ws.row_dimensions[1].height = 45
    ws.freeze_panes = "C2"
    for r, values in enumerate(rows, 2):
        for col, v in enumerate(values, 1):
            c = ws.cell(row=r, column=col, value=v)
            c.font = Font(name=MP.FONT, size=10)
            c.alignment = MP.WRAP
            c.border = MP.BOX
            if col >= fill_from:
                c.fill = MP.FILL
        ws.row_dimensions[r].height = height
    for col, options in validations.items():
        dv = DataValidation(type="list", formula1='"' + ",".join(options) + '"',
                            allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{len(rows) + 1}")
    return ws


def build() -> None:
    from mpcr_rag.store import pg_store
    checker = checker_items()
    conn = pg_store.connect()
    with conn.cursor() as cur:
        claims = claim_items(cur)
        uses, use_counts = use_items(cur)
    conn.close()

    wb = Workbook()
    ins = wb.active
    ins.title = "Instrucciones"
    ins.column_dimensions["A"].width = 110
    for i, (t, bold, size) in enumerate(INSTRUCTIONS, 1):
        MP._cell(ins, f"A{i}", t, bold=bold, size=size)
        if len(t) > 110:
            ins.row_dimensions[i].height = 15 * (len(t) // 100 + 1)

    _sheet(wb, "1 Verificador",
           [("ID", 6), ("Tema", 14), ("Pregunta", 32), ("Lo que respalda el registro", 40),
            ("Respuesta", 70), ("Veredicto", 18), ("¿Se niega a responder?", 11),
            ("Postura", 11), ("¿Duda?", 8), ("Nota", 30)],
           [[i["id"], TIPO[i["qtype"]], i["question"], i["key"], i["answer"], "", "", "", "", ""]
            for i in checker], 6,
           {"F": VERDICTS, "G": ("sí", "no"),
            "H": ("sí", "posible", "no", "incierto", "no aplica"), "I": ("sí", "no")})
    _sheet(wb, "2 Contradichas",
           [("ID", 6), ("Especie", 18), ("Pregunta", 28), ("Afirmación marcada", 32),
            ("Respuesta completa", 55), ("Registro de la especie", 50),
            ("Lo que respalda el registro para la pregunta", 34), ("¿De quién es el error?", 18),
            ("Nota", 30)],
           [[i["id"], i["species"], i["question"], i["claim"], i["answer"], i["record"],
             i["key"], "", ""] for i in claims], 8,
           {"H": ("del verificador", "de la clave o registro", "de la respuesta", "no sé")},
           height=200)
    _sheet(wb, "3 Usos",
           [("ID", 6), ("Especie", 22), ("Oraciones que mencionan usos", 60),
            ("Texto completo (discusión y distribución)", 80),
            ("¿Dice que es comestible o medicinal?", 14), ("Nota", 30)],
           [[i["id"], i["species"], i["sentences"], i["text"][:3000], "", ""] for i in uses], 5,
           {"E": ("sí", "no", "dudoso")}, height=120)

    out = DATA / "tareas_validacion_jose.xlsx"
    wb.save(out)
    key = {"seed": SEED,
           "checker": [{k: i[k] for k in ("id", "qid", "arm", "origin", "checker", "stance",
                                          "abstains")} for i in checker],
           "claims": [{k: i[k] for k in ("id", "qid", "arm", "claim_index", "claim",
                                         "claim_type")} for i in claims],
           "uses": [{k: i[k] for k in ("id", "species", "stratum")} for i in uses],
           "use_pool_sizes": use_counts}
    (DATA / "author_tasks_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    print(f"{out.name}: checker {len(checker)} "
          f"({collections.Counter(i['origin'] == 'new' for i in checker)[True]} new), "
          f"claims {len(claims)} "
          f"({dict(collections.Counter(i['claim_type'] for i in claims))}), uses {len(uses)}")
    print("  checker verdicts (hidden):", dict(collections.Counter(i["checker"] for i in checker)))
    print("  use pools:", use_counts)


if __name__ == "__main__":
    build()
