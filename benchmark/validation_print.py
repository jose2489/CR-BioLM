"""The checker validation sheet as a fillable PDF, to label on a tablet or on paper.

One answer per page, with the question, what the record supports and the answer, then
a tick box per verdict and fields for the quote and a note. The species entries
(checker_validation_fichas.md) follow as an appendix, each card linking to its entry.
The checker's verdict stays out, as on the CSV.

The PDF's form fields can be read back:

    python -m benchmark.validation_print --read path/to/filled.pdf

which writes the labels into checker_validation_sheet.csv (columns veredicto_humano,
cita_que_lo_justifica, duda, nota), leaving any row with no tick untouched.

Run:  python -m benchmark.validation_print        (writes checker_validation_sheet.pdf)
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
ROOT = DATA.parent.parent
SHEET = DATA / "checker_validation_sheet.csv"
FICHAS = DATA / "checker_validation_fichas.md"
OUT = DATA / "checker_validation_sheet.pdf"

VERDICTS = [
    ("correct", "correct", "coincide con lo que respalda el registro (con tolerancia)"),
    ("partially_correct", "partially correct",
     "se traslapa, pero se extiende más allá u omite una parte"),
    ("overstated", "overstated",
     "afirma con más certeza de la que el registro respalda (\\emph{sí} donde la clave es \\emph{posible})"),
    ("understated", "understated",
     "duda donde el registro es definitivo (\\emph{posible} donde hay registros que lo confirman)"),
    ("wrong", "wrong", "contradice la clave"),
    ("abstained", "abstained", "no responde una pregunta que el registro sí resuelve"),
    ("beyond_record", "beyond record", "afirma algo sobre lo que el registro no dice nada"),
]
TYPES = {"elevation": "altitud", "vertiente": "vertiente", "flowering": "floración",
         "growth_form": "forma de crecimiento", "endemism": "endemismo",
         "park_possible": "área protegida (posible)", "park_present": "área protegida (presente)",
         "park_unlikely": "área protegida (improbable)", "cultivation": "cultivo",
         "outside_record": "uso fuera del registro", "light": "luz"}


def tex(s: str) -> str:
    s = s or ""
    s = (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
         .replace("$", "\\$").replace("#", "\\#").replace("_", "\\_").replace("{", "\\{")
         .replace("}", "\\}").replace("~", "\\textasciitilde{}").replace("^", "\\^{}"))
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"\*(.+?)\*", r"\\emph{\1}", s)
    s = s.replace("\n\n", "\\par ").replace("\n", " \\newline ")
    return s


def _field(s: str) -> str:
    """A note already on the CSV, prefilled in its field (plain text only)."""
    return re.sub(r"[\\{}%#&$^_~()]", " ", s or "").strip()


def _fichas() -> tuple[str, dict[str, str]]:
    """The appendix in LaTeX, and row id -> label of its species' entry."""
    rows_of: dict[str, str] = {}
    out = []
    label = None
    in_list = False
    for line in io.open(FICHAS, encoding="utf-8").read().splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            if in_list:
                out.append("\\end{itemize}")
                in_list = False
            name = line[3:].strip()
            label = "f:" + re.sub(r"\W+", "-", name)
            out.append(f"\\clearpage\\subsection*{{\\emph{{{tex(name)}}}}}\\label{{{label}}}")
            continue
        if label is None:
            continue
        m = re.match(r"Filas:\s*(.+)", line)
        if m:
            ids = [x.strip() for x in m.group(1).split(",")]
            for i in ids:
                rows_of[i] = label
            out.append("Filas: " + ", ".join(f"\\hyperref[c:{i}]{{{i}}}" for i in ids) + "\\par")
            continue
        if line.startswith("Archivo:"):
            continue
        m = re.match(r"!\[.*?\]\((.+?)\)", line)
        if m:
            p = (FICHAS.parent / m.group(1)).resolve()
            if p.exists():
                out.append("\\begin{center}\\includegraphics[width=.85\\linewidth,height=.45\\textheight,"
                           f"keepaspectratio]{{{p.as_posix()}}}\\end{{center}}")
            continue
        if line.startswith("- "):
            if not in_list:
                out.append("\\begin{itemize}\\setlength\\itemsep{1pt}")
                in_list = True
            out.append("\\item " + tex(line[2:]))
            continue
        if in_list and not line.strip():
            out.append("\\end{itemize}")
            in_list = False
            continue
        if line.startswith("> "):
            out.append("\\begin{quote}\\small " + tex(line[2:]) + "\\end{quote}")
            continue
        if line.strip():
            out.append(tex(line) + "\\par")
    if in_list:
        out.append("\\end{itemize}")
    return "\n".join(out), rows_of


def build() -> Path:
    rows = list(csv.DictReader(io.open(SHEET, encoding="utf-8-sig")))
    appendix, rows_of = _fichas()
    legend = "\n".join(f"\\item[\\fbox{{\\rule{{0pt}}{{6pt}}\\hspace{{6pt}}}}] \\textbf{{{lab}}}: {desc}"
                       for _k, lab, desc in VERDICTS)
    cards = []
    for r in rows:
        i = r["id"]
        boxes = " \\quad ".join(
            f"\\mbox{{\\CheckBox[name={i}-{k},width=1.2em,height=1.2em,bordercolor=0 0 0]{{}}"
            f"\\,{lab}}}" for k, lab, _d in VERDICTS)
        ficha = (f" \\hfill \\hyperref[{rows_of[i]}]{{ver ficha}}" if i in rows_of else "")
        cards.append(f"""
\\clearpage
\\phantomsection\\label{{c:{i}}}
{{\\Large\\textbf{{{i}}}}} \\quad {tex(TYPES.get(r['tipo'], r['tipo']))}{ficha}

\\medskip
\\textbf{{Pregunta.}} {tex(r['pregunta'])}

\\medskip
\\noindent\\colorbox{{gray!12}}{{\\parbox{{\\dimexpr\\linewidth-2\\fboxsep}}{{\\textbf{{Lo que respalda el registro.}} {tex(r['lo_que_respalda_el_registro'])}}}}}

\\medskip
\\textbf{{Respuesta.}} {tex(r['respuesta'])}

\\vfill
\\noindent\\rule{{\\linewidth}}{{.4pt}}

\\textbf{{Veredicto}} (marque uno):\\par\\smallskip
\\begin{{flushleft}}{boxes}\\end{{flushleft}}

\\mbox{{\\CheckBox[name={i}-duda,width=1.2em,height=1.2em,bordercolor=0 0 0]{{}}\\, tengo duda sobre este veredicto}}

\\smallskip
\\textbf{{Cita que lo justifica}} (frase de la respuesta):\\par
\\TextField[name={i}-cita,multiline=true,width=\\linewidth,height=2.2cm,bordercolor=0.6 0.6 0.6]{{}}

\\smallskip
\\textbf{{Nota}}:\\par
\\TextField[name={i}-nota,multiline=true,width=\\linewidth,height=1.8cm,bordercolor=0.6 0.6 0.6,value={{{_field(r['nota'])}}}]{{}}
""")
    index = " ".join(f"\\hyperref[c:{r['id']}]{{{r['id']}}}" for r in rows)
    doc = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[margin=1.8cm]{{geometry}}
\\usepackage{{fontspec}}
\\setmainfont{{Calibri}}
\\usepackage[spanish]{{babel}}
\\usepackage{{graphicx,xcolor,enumitem}}
\\usepackage[colorlinks=true,linkcolor=blue!60!black]{{hyperref}}
\\setlength\\parindent{{0pt}}
\\pagestyle{{plain}}
\\begin{{document}}
\\begin{{Form}}
{{\\LARGE\\textbf{{Validación del verificador}}}}\\par\\medskip
40 respuestas reales de la corrida. Para cada una, dé el veredicto que daría un lector
cuidadoso \\textbf{{contra el recuadro gris}} (lo que respalda el registro), no contra lo
que usted sabe de la planta. El veredicto del verificador y el modelo que escribió la
respuesta no aparecen, para que el juicio sea ciego.

\\medskip
\\textbf{{Cómo llenarla.}} En el iPad, toque las casillas y los campos (Archivos, Acrobat o
GoodNotes los aceptan); así las etiquetas se pueden leer automáticamente del PDF. También
puede marcar con el lápiz o imprimir: en ese caso las transcribo yo.

\\medskip
\\textbf{{Veredictos}} (marque uno por respuesta):
\\begin{{itemize}}[leftmargin=2.2em]
{legend}
\\end{{itemize}}

\\textbf{{Tolerancias.}} Altitud: margen de 200 m. Floración: \\emph{{correct}} si al menos
tres cuartos de los meses nombrados están en la clave y nombra al menos la mitad de los
meses de la clave; \\emph{{wrong}} si menos de un cuarto. Vertiente: \\emph{{correct}} si
nombra las vertientes que acepta la clave; \\emph{{partially correct}} si se traslapa.
En cultivo, sembrar fuera del rango natural no se evalúa.

\\medskip
\\textbf{{Cita.}} La frase de la respuesta en que se basa su veredicto. \\textbf{{Duda.}}
Marque si el caso es ambiguo; no lo salte.

\\medskip
\\textbf{{Ir a:}} {index}

\\medskip
Al final están las fichas de las especies (texto del Manual, campos extraídos y mapa),
con un enlace desde cada respuesta.
{''.join(cards)}
\\clearpage
\\section*{{Anexo: fichas de las especies}}
Úselas para comprobar el recuadro gris; el veredicto se sigue dando contra ese recuadro.
{appendix}
\\end{{Form}}
\\end{{document}}
"""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "sheet.tex"
        src.write_text(doc, encoding="utf-8")
        for _ in range(2):
            p = subprocess.run(["lualatex", "-interaction=nonstopmode", "sheet.tex"],
                               cwd=tmp, capture_output=True, text=True, errors="replace")
        pdf = Path(tmp) / "sheet.pdf"
        if not pdf.exists():
            raise SystemExit(p.stdout[-3000:])
        OUT.write_bytes(pdf.read_bytes())
    return OUT


def read(path: Path) -> None:
    """Labels from a filled PDF into the CSV."""
    from pypdf import PdfReader
    f = {k: (v.get("/V") if isinstance(v, dict) else v)
         for k, v in (PdfReader(str(path)).get_fields() or {}).items()}
    on = lambda v: v not in (None, "/Off", "Off", "")
    rows = list(csv.DictReader(io.open(SHEET, encoding="utf-8-sig")))
    fields = [k for k in rows[0].keys() if k]       # the CSV carries empty trailing columns
    done = 0
    for r in rows:
        i = r["id"]
        ticked = [k for k, _l, _d in VERDICTS if on(f.get(f"{i}-{k}"))]
        if not ticked:
            continue
        if len(ticked) > 1:
            print(f"  {i}: more than one verdict ticked {ticked}; left blank")
            continue
        r["veredicto_humano"] = ticked[0]
        r["cita_que_lo_justifica"] = str(f.get(f"{i}-cita") or "").strip()
        r["duda (s/n)"] = "s" if on(f.get(f"{i}-duda")) else "n"
        r["nota"] = str(f.get(f"{i}-nota") or "").strip()
        done += 1
    with io.open(SHEET, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{done} of {len(rows)} rows labelled from {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", default="", help="a filled PDF: write its labels into the CSV")
    a = ap.parse_args()
    if a.read:
        read(Path(a.read))
    else:
        print(f"wrote {build()}")
