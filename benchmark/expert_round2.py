"""The second expert round: a blind choice among four maps, the same sheet for both
botanists.

For each of 15 species the expert sees four maps, unlabelled and in random order: the
system's (the Manual's range plus regions holding at least five records), the Manual's
stated range alone, and the ranges Mistral Large 4 and Gemma 4 31B describe when asked
in plain Spanish. The expert picks the one that best shows where the species grows and
marks which are acceptable. Only the family and growth form are shown beside the maps:
the Manual's entry, shown in the first round, favours the two maps drawn from it.

Answers are not reviewed in this round. The grounded-against-free comparison was
reviewed in the first round, and a comparison of two answers written from the same
facts (template against model) turned out to show near-identical texts on most types,
which the paper reports from the texts themselves.

Both experts see the same species, so their agreement can be reported (Cohen's kappa).

How the species are chosen, so it can be reported as it was done: from the 300-species
map sample (>= 30 records, disjoint from the benchmark), the species for which all four
sources give a drawable map; 5 drawn at random from each tertile of worldwide record
count. Seed recorded. Which letter is which goes to expert_round2_key.json, not sent.

Run:  python -m benchmark.expert_round2
Writes revision_mapas_<expert>.xlsx and expert_round2_key.json
"""
from __future__ import annotations

import io
import json
import random
import tempfile
from pathlib import Path

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

from . import map_preference_sheet as MP

DATA = Path(__file__).resolve().parent / "data"
SEED = "20261025-round2"
EXPERTS = ("Armando", "Nelson")
N_MAPS_PER_TIER = 5
SOURCES = ("ours", "manual", "paid", "open")


def _jsonl(name: str) -> list[dict]:
    return [json.loads(l) for l in io.open(DATA / name, encoding="utf-8") if l.strip()]


def select_maps() -> list[str]:
    sample = json.loads((DATA / "maps_sample.json").read_text(encoding="utf-8"))["sample"]
    rows = _jsonl("mapcompare_scores_maps.jsonl")
    drawable = {(r["species"], r["source"]) for r in rows
                if r["has_map"] and r.get("area", 0) > 0}
    rng = random.Random(SEED + "-maps")
    picked = []
    for g in ("G1", "G2", "G3"):
        pool = sorted(r["species"] for r in sample if r["stratum"] == g
                      and all((r["species"], s) in drawable
                              for s in ("manual", "paid", "open")))
        picked += rng.sample(pool, N_MAPS_PER_TIER)
    rng.shuffle(picked)
    return picked


def _growth_form(species: str) -> str:
    from .cooccurrence import _habits, _norm
    forms = sorted(_habits().get(_norm(species), set()))
    return ", ".join(forms) if forms else "no indicada"


INSTRUCTIONS = [
    ("Elección de mapas de distribución", True, 14),
    ("", False, 10),
    ("Son 15 especies; calculamos entre 15 y 20 minutos. Solo hay que llenar las celdas "
     "amarillas de la hoja «Mapas».", False, 10),
    ("", False, 10),
    ("Para cada especie hay cuatro mapas, A, B, C y D, dibujados de la misma forma. El área "
     "verde es donde cada mapa dice que la especie puede encontrarse en Costa Rica. Los mapas "
     "salen de fuentes distintas y el orden cambia de una especie a otra; no se indica cuál es "
     "cuál. Junto a la especie solo se indica la familia y la forma de crecimiento.", False, 10),
    ("", False, 10),
    ("Qué le pedimos, por especie:", True, 11),
    ("1. ¿Cuál mapa representa mejor dónde se encuentra la especie? Elija una letra. Si dos o "
     "más le parecen igual de buenos, elija «Empate» y escriba las letras en el comentario. "
     "Si no conoce la especie lo suficiente, elija «No la conozco».", False, 10),
    ("2. ¿Cuáles mapas considera aceptables? Escriba las letras (por ejemplo «A, C») o "
     "«Ninguno».", False, 10),
    ("3. Un comentario, si quiere: qué le falta o le sobra al mejor mapa.", False, 10),
    ("", False, 10),
    ("A veces dos mapas son idénticos; es normal. No necesita consultar fuentes: buscamos su "
     "criterio como botánico. La primera fila, en gris, es un ejemplo ya llenado.", False, 10),
]


def _row(ws, r, values, height, grey=False):
    for col, v in enumerate(values, 1):
        c = ws.cell(row=r, column=col, value=v)
        c.font = Font(name=MP.FONT, size=10, italic=grey)
        c.alignment = MP.WRAP
        c.border = MP.BOX
        if grey:
            c.fill = MP.GREY
        elif col >= 4:
            c.fill = MP.FILL
    ws.row_dimensions[r].height = height


def build() -> None:
    from mpcr_rag.store import pg_store
    maps = select_maps()
    rng = random.Random(SEED + "-letters")
    letters = [{"species": sp, "letters": dict(zip("ABCD", rng.sample(SOURCES, 4)))}
               for sp in maps]
    conn = pg_store.connect()
    with conn.cursor() as cur, tempfile.TemporaryDirectory() as tmp:
        ings = {sp: MP._ingredients(cur, sp) for sp in maps}
        pngs = {}
        for i, e in enumerate(letters, 1):
            e["identical_ours_manual"] = ings[e["species"]]["ours"] == ings[e["species"]]["manual"]
            png = Path(tmp) / f"m{i}.png"
            MP._panel_png(list(e["letters"].values()), ings[e["species"]], png)
            pngs[e["species"]] = png
        forms = {sp: _growth_form(sp) for sp in maps}

        for expert in EXPERTS:
            wb = Workbook()
            ins = wb.active
            ins.title = "Instrucciones"
            ins.column_dimensions["A"].width = 110
            for i, (t, bold, size) in enumerate(INSTRUCTIONS, 1):
                MP._cell(ins, f"A{i}", t, bold=bold, size=size)
                if len(t) > 110:
                    ins.row_dimensions[i].height = 15 * (len(t) // 100 + 1)

            ws = wb.create_sheet("Mapas")
            heads = [("N.º", 5), ("Especie", 26), ("Mapas", 150),
                     ("¿Cuál representa mejor dónde se encuentra?", 16),
                     ("¿Cuáles son aceptables?", 16), ("Comentario", 34)]
            for col, (t, w) in enumerate(heads, 1):
                c = ws.cell(row=1, column=col, value=t)
                c.font = Font(name=MP.FONT, bold=True, color="FFFFFF", size=10)
                c.fill = MP.HEAD
                c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
                ws.column_dimensions[c.column_letter].width = w
            ws.row_dimensions[1].height = 40
            ws.freeze_panes = "C2"
            _row(ws, 2, ["Ej.", "Especie de ejemplo\nFamilia. Árbol.",
                         "(aquí van los cuatro mapas)", "C", "B, C",
                         "EJEMPLO. C es el mejor; B también es aceptable aunque le falta "
                         "la Península de Osa."], 40, grey=True)
            for i, e in enumerate(letters, 1):
                sp = e["species"]
                f = ings[sp]["ficha"]
                _row(ws, i + 2, [i, f"{sp}\n{f.family}. Forma de crecimiento: {forms[sp]}.",
                                 "", "", "", ""], 255)
                img = XLImage(str(pngs[sp]))
                img.width, img.height = 1050, 330
                ws.add_image(img, f"C{i + 2}")
            dv = DataValidation(type="list", formula1='"A,B,C,D,Empate,No la conozco"',
                                allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"D3:D{len(letters) + 2}")

            out = DATA / f"revision_mapas_{expert}.xlsx"
            wb.save(out)
            print(f"{expert}: {out.name} ({out.stat().st_size // 1024} KB)")
    conn.close()
    key = {"seed": SEED, "maps": letters}
    (DATA / "expert_round2_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    print(f"maps: {len(maps)} species "
          f"({sum(e['identical_ours_manual'] for e in letters)} where ours equals the Manual's)")


if __name__ == "__main__":
    build()
