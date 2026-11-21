"""The third expert round: the climate a species grows in, and where to look for it next.

Two short blocks on the second round's species, which both botanists have already seen.

Clima (10 species): four statements of the climate where the species grows, A-D,
unlabelled and in random order: the system's profile (5th-95th percentile of annual
mean temperature and annual rainfall over its records, rounded as a person would state
them), what Mistral Large 4 and Gemma 4 31B say when asked freely, and what Mistral
says when given the Manual's entry. The questions are the map round's: which is best,
which are acceptable, a comment.

Dónde buscarla (5 species): a map of every record of the species and five numbered
5 km cells in which it has never been recorded: all five at least half
outside the system's map, where the model goes beyond the record's stated range: the
three the forecast model (benchmark.forecast, without collecting effort) scores highest
among those cells, and two it scores in the lowest quarter of those that lie inside the
Manual's elevation band, as near as possible as far from a known record as the first
two (controls; the map shows all five outside the range and the controls are at a
plausible altitude, so neither gives them away); no two cells within about 15 km; in random
order. Per cell:
would you expect to find it there? The model's scores use what was known by 2009; the
map shows every record to date, and no cell shown has a record at any date.

How the species are chosen, so it can be reported as it was done: of the 15 species of
the second round, Syzygium malaccense is left out (an introduced tree, which one expert
excluded). Climate: 10 drawn at random from those for which all four sources state both
ranges. Dónde buscarla: the species not drawn for climate, plus enough drawn at random
from the climate set to make 5. Seed recorded. Which letter and which cell is which goes
to expert_round3_key.json, not sent.

Run:  python -m benchmark.expert_round3
Writes revision_clima_prediccion_<expert>.xlsx and expert_round3_key.json
"""
from __future__ import annotations

import io
import json
import math
import random
import tempfile
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

from . import climate_profile as C
from . import expert_round2 as E2
from . import forecast as F
from . import map_preference_sheet as MP
from . import map_refine as R

DATA = Path(__file__).resolve().parent / "data"
SEED = "20261103-round3"
EXPERTS = E2.EXPERTS
EXCLUDED = {"Syzygium malaccense"}
N_CLIMATE, N_FORECAST = 10, 5
CLIMATE_SOURCES = ("ours", "paid", "open", "paid_manual")
N_TOP, N_CONTROL = 3, 2
MIN_APART = 3                 # cells (about 15 km) between any two cells shown


def _jsonl(name: str) -> list[dict]:
    return [json.loads(l) for l in io.open(DATA / name, encoding="utf-8") if l.strip()]


def select() -> tuple[list[str], list[str]]:
    key = json.loads((DATA / "expert_round2_key.json").read_text(encoding="utf-8"))
    pool = [m["species"] for m in key["maps"] if m["species"] not in EXCLUDED]
    has = {(r["species"], r["source"]) for r in _jsonl("climate_compare_scores.jsonl")
           if r["has_env"]}
    eligible = sorted(s for s in pool if all((s, src) in has for src in CLIMATE_SOURCES))
    rng = random.Random(SEED + "-species")
    climate = rng.sample(eligible, N_CLIMATE)
    rest = sorted(s for s in pool if s not in climate)
    forecast = rest + rng.sample(sorted(climate), max(0, N_FORECAST - len(rest)))
    rng.shuffle(climate), rng.shuffle(forecast)
    return climate, forecast[:N_FORECAST]


# ------------------------------------------------------------------------ climate

def _num(x: float, step: int) -> str:
    return f"{round(x / step) * step:,.0f}".replace(",", ".")


def _range(rng, unit: str, step: int) -> str:
    if not rng:
        return "no lo indica"
    lo, hi = float(rng[0]), float(rng[1])
    if math.isinf(lo):
        return f"menos de {_num(hi, step)} {unit}"
    if math.isinf(hi):
        return f"más de {_num(lo, step)} {unit}"
    return f"{_num(lo, step)}–{_num(hi, step)} {unit}"


def statement(env: dict) -> str:
    return (f"Temperatura media anual: {_range(env.get('bio1'), '°C', 1)}\n"
            f"Lluvia anual: {_range(env.get('bio12'), 'mm', 100)}")


def climate_envs(cur, species: list[str]) -> dict:
    stated = {(r["species"], r["source"]): r["env"] for r in _jsonl("climate_compare_scores.jsonl")
              if r["source"] != "ours" and r["has_env"]}
    out = {}
    for sp in species:
        v = C.profile(cur, sp)["vars"]
        out[sp] = {"ours": {k: (v[k]["p05"], v[k]["p95"]) for k in ("bio1", "bio12")}}
        for src in CLIMATE_SOURCES[1:]:
            out[sp][src] = stated[(sp, src)]
    return out


# ----------------------------------------------------------------------- forecast

def forecast_cells(species: list[str], G: F.Grid) -> dict:
    """Per species, the numbered cells: top-scored and matched low-scored cells with no
    record at any date, from the test table (known to 2009)."""
    names, sets, _ = F.build()
    tr, te = sets["train"], sets["test"]
    cols = F._cols(F.SUBSETS["without effort"])
    gb, _, _ = F.train(tr, cols, 3_000_000)
    index = {n: i for i, n in enumerate(names)}
    near = F.FEATURES.index("near_km")
    sysmap = F.FEATURES.index("system_map")
    band = F.FEATURES.index("manual_band")
    out = {}
    for sp in species:
        m = (te["s"] == index[sp]) & ~te["y"]          # candidates never recorded
        X, c = te["X"][m], te["c"][m]
        p = gb.predict_proba(X[:, cols])[:, 1]
        d = np.nan_to_num(X[:, near], nan=1e4)
        rc = np.c_[G.cr[c], G.cc[c]] / F.CELL
        chosen: list[int] = []

        def apart(j):
            return all(np.abs(rc[j] - rc[k]).max() >= MIN_APART for k in chosen)

        # every cell shown lies (mostly) outside the system's map: where the model goes
        # beyond the record's range, and where the map cannot tell the expert anything
        outside = np.flatnonzero(X[:, sysmap] < .5)
        top = []
        for j in outside[np.argsort(-p[outside])]:
            if len(top) == N_TOP:
                break
            if apart(j):
                top.append(j), chosen.append(j)
        # controls: also outside the map, scored in the lowest quarter of those cells,
        # as near as possible as far from a known record as the first two
        # and inside the Manual's elevation band, so altitude alone does not give them away
        ctl = outside[X[outside, band] >= .5]
        ctl = ctl if len(ctl) >= 10 else outside
        pool = list(ctl[p[ctl] <= np.quantile(p[ctl], .25)])
        controls = []
        for t in top[:N_CONTROL]:
            free = [j for j in pool if apart(j)]
            j = min(free, key=lambda j: abs(math.log1p(d[j]) - math.log1p(d[t])))
            controls.append(j), chosen.append(j)
        rank = {j: int((p > p[j]).sum()) + 1 for j in top + controls}
        cells = ([{"cell": int(c[j]), "kind": "top", "p": float(p[j]), "near_km": float(d[j]),
                   "rank_all": rank[j], "n_cand": int(len(p))}
                  for j in top] +
                 [{"cell": int(c[j]), "kind": "control", "p": float(p[j]), "near_km": float(d[j]),
                   "rank_all": rank[j], "n_cand": int(len(p))}
                  for j in controls])
        random.Random(f"{SEED}-{sp}").shuffle(cells)
        for k, e in enumerate(cells, 1):
            e["number"] = k
        out[sp] = cells
    return out


def _describe(G: F.Grid, cell: int) -> str:
    from pyproj import Transformer
    g = R.grid()
    px = np.flatnonzero(G.pc == cell)
    x, y = g["transform"] * (float(G.cc[cell]) + .5, float(G.cr[cell]) + .5)
    lon, lat = Transformer.from_crs(g["crs"], 4326, always_xy=True).transform(x, y)
    el = G.elev[px]
    el = el[~np.isnan(el)]
    region, slope = G.pair_codes[np.bincount(G.pair_id[px]).argmax()].split("|")
    where = region if slope in ("", "None", "nan") else f"{region}, vertiente {slope}"
    lo, hi = (_num(np.quantile(el, q), 50) for q in (.1, .9)) if len(el) else (None, None)
    alt = "sin dato" if lo is None else f"unos {hi} m" if lo == hi else f"{lo}–{hi} m"
    return f"{lat:.2f}° N, {abs(lon):.2f}° O; {alt}; {where}"


def cell_boxes(G: F.Grid, cells: list[int]):
    """Each cell's outline, as a polygon in the elevation model's CRS."""
    import geopandas as gpd
    from shapely.geometry import box
    g = R.grid()
    t = g["transform"]
    polys = []
    for c in cells:
        px = G.pc == c
        r0, r1 = G.rows[px].min(), G.rows[px].max() + 1
        c0, c1 = G.cols[px].min(), G.cols[px].max() + 1
        (x0, y0), (x1, y1) = t * (c0, r0), t * (c1, r1)
        polys.append(box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    return gpd.GeoSeries(polys, crs=g["crs"])


CELL_COLOR = "#ff3df5"


def system_map(cur, sp: str, path: Path, lonlat=None, overlay=None) -> None:
    """The system's distribution map (the renderer validated in the companion paper:
    regions, elevation band, protected areas, occurrence records), with an overlay."""
    import geopandas as gpd
    from mpcr_rag.ingest.field_extractor import split_distribution
    from utils.distribution_map import build_ficha, generate_distribution_map
    ficha = MP._ingredients(cur, sp)["ficha"]
    habitat, notes = split_distribution(ficha.distribution_paragraph)
    df = build_ficha(habitat_raw=habitat, geographic_notes=notes, species=sp)
    if lonlat is None:
        lonlat, _ = R._records(cur, sp)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy([p[0] for p in lonlat],
                                                       [p[1] for p in lonlat]), crs=4326)
    generate_distribution_map(df, path, presencias_gdf=pts if len(pts) else None,
                              overlay=overlay)


def draw_cells(boxes, labels, color=CELL_COLOR):
    """An overlay: cell outlines with their labels."""
    import matplotlib.patheffects as pe

    def overlay(ax, crs):
        b = boxes.to_crs(crs)
        b.plot(ax=ax, facecolor="none", edgecolor=color, linewidth=2.2, zorder=20)
        for lab, geom in zip(labels, b.geometry):
            if lab is None:
                continue
            ax.annotate(str(lab), (geom.bounds[2], geom.bounds[3]), color=color, fontsize=13,
                        fontweight="bold", zorder=21, xytext=(2, 2), textcoords="offset points",
                        path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    return overlay


def forecast_png(cur, G: F.Grid, sp: str, cells: list[dict], path: Path) -> None:
    boxes = cell_boxes(G, [e["cell"] for e in cells])
    system_map(cur, sp, path, overlay=draw_cells(boxes, [e["number"] for e in cells]))


# -------------------------------------------------------------------------- sheet

INSTRUCTIONS = [
    ("Clima y dónde buscarla", True, 14),
    ("", False, 10),
    ("Son dos hojas cortas, sobre especies de la ronda de mapas; calculamos unos 20 minutos. "
     "Solo hay que llenar las celdas amarillas.", False, 10),
    ("", False, 10),
    ("Hoja «Clima» (10 especies). Para cada especie hay cuatro descripciones, A, B, C y D, del "
     "clima donde crece: temperatura media anual y lluvia anual. Salen de fuentes distintas y "
     "el orden cambia de una especie a otra; no se indica cuál es cuál.", False, 10),
    ("1. ¿Cuál describe mejor el clima donde crece la especie? Elija una letra, «Empate» (y "
     "escriba las letras en el comentario) o «No la conozco».", False, 10),
    ("2. ¿Cuáles considera aceptables? Escriba las letras (por ejemplo «A, C») o «Ninguna».",
     False, 10),
    ("3. Un comentario, si quiere.", False, 10),
    ("", False, 10),
    ("Hoja «Dónde buscarla» (5 especies). El mapa es el del sistema que ya conoce (regiones, "
     "banda de altitud en cian, áreas protegidas y registros en rojo). Encima hay cinco cuadros "
     "fucsia numerados, de unos 5 × 5 km, donde la especie nunca se ha registrado. Debajo se "
     "indica la ubicación, la altitud y la región de cada cuadro.", False, 10),
    ("Para cada cuadro: ¿esperaría encontrar la especie ahí? «Sí», «Posiblemente», «Improbable» "
     "o «No sé». Un comentario por especie, si quiere.", False, 10),
    ("", False, 10),
    ("No necesita consultar fuentes: buscamos su criterio como botánico. La primera fila de cada "
     "hoja, en gris, es un ejemplo ya llenado.", False, 10),
]


def _heads(ws, heads):
    for col, (t, w) in enumerate(heads, 1):
        c = ws.cell(row=1, column=col, value=t)
        c.font = Font(name=MP.FONT, bold=True, color="FFFFFF", size=10)
        c.fill = MP.HEAD
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        ws.column_dimensions[c.column_letter].width = w
    ws.row_dimensions[1].height = 40


def _plain(ws, r, col, value, *, fill=None, bold=False):
    c = ws.cell(row=r, column=col, value=value)
    c.font = Font(name=MP.FONT, size=10, bold=bold)
    c.alignment = MP.WRAP
    c.border = MP.BOX
    if fill is not None:
        c.fill = fill
    return c


def build() -> None:
    from mpcr_rag.store import pg_store
    climate, forecast = select()
    rng = random.Random(SEED + "-letters")
    letters = [{"species": sp, "letters": dict(zip("ABCD", rng.sample(CLIMATE_SOURCES, 4)))}
               for sp in climate]
    G = F.Grid()
    cells = forecast_cells(forecast, G)
    conn = pg_store.connect()
    with conn.cursor() as cur, tempfile.TemporaryDirectory() as tmp:
        envs = climate_envs(cur, climate)
        fichas = {sp: MP._ingredients(cur, sp)["ficha"] for sp in climate + forecast}
        forms = {sp: E2._growth_form(sp) for sp in climate + forecast}
        pngs = {}
        for sp in forecast:
            pngs[sp] = Path(tmp) / f"{sp.replace(' ', '_')}.png"
            forecast_png(cur, G, sp, cells[sp], pngs[sp])
            for e in cells[sp]:
                e["where"] = _describe(G, e["cell"])
        label = lambda sp: f"{sp}\n{fichas[sp].family}. Forma de crecimiento: {forms[sp]}."

        for expert in EXPERTS:
            wb = Workbook()
            ins = wb.active
            ins.title = "Instrucciones"
            ins.column_dimensions["A"].width = 110
            for i, (t, bold, size) in enumerate(INSTRUCTIONS, 1):
                MP._cell(ins, f"A{i}", t, bold=bold, size=size)
                if len(t) > 110:
                    ins.row_dimensions[i].height = 15 * (len(t) // 100 + 1)

            ws = wb.create_sheet("Clima")
            _heads(ws, [("N.º", 5), ("Especie", 28)] +
                   [(f"Descripción {k}", 24) for k in "ABCD"] +
                   [("¿Cuál describe mejor el clima donde crece?", 16),
                    ("¿Cuáles son aceptables?", 16), ("Comentario", 34)])
            ex = ["Ej.", "Especie de ejemplo\nFamilia. Árbol.",
                  "Temperatura media anual: 18–24 °C\nLluvia anual: 2.000–3.500 mm",
                  "Temperatura media anual: 22–27 °C\nLluvia anual: 1.500–2.500 mm",
                  "Temperatura media anual: 15–20 °C\nLluvia anual: más de 3.000 mm",
                  "Temperatura media anual: 18–25 °C\nLluvia anual: 2.500–4.000 mm",
                  "D", "A, D", "EJEMPLO. D es la mejor; A también, aunque la lluvia se queda corta."]
            for col, v in enumerate(ex, 1):
                c = _plain(ws, 2, col, v, fill=MP.GREY)
                c.font = Font(name=MP.FONT, size=10, italic=True)
            ws.row_dimensions[2].height = 45
            for i, e in enumerate(letters, 1):
                sp, r = e["species"], i + 2
                _plain(ws, r, 1, i)
                _plain(ws, r, 2, label(sp))
                for col, k in enumerate("ABCD", 3):
                    _plain(ws, r, col, statement(envs[sp][e["letters"][k]]))
                for col in (7, 8, 9):
                    _plain(ws, r, col, "", fill=MP.FILL)
                ws.row_dimensions[r].height = 50
            dv = DataValidation(type="list", formula1='"A,B,C,D,Empate,No la conozco"',
                                allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"G3:G{len(letters) + 2}")
            ws.freeze_panes = "C2"

            ws = wb.create_sheet("Dónde buscarla")
            _heads(ws, [("N.º", 5), ("Especie y mapa", 95), ("Cuadro", 8),
                        ("Ubicación, altitud y región", 52),
                        ("¿Esperaría encontrarla ahí?", 16), ("Comentario", 34)])
            dv = DataValidation(type="list", formula1='"Sí,Posiblemente,Improbable,No sé"',
                                allow_blank=True)
            ws.add_data_validation(dv)
            r = 2
            for col, v in enumerate(["Ej.", "Especie de ejemplo (aquí va el mapa)", 1,
                                     "9.95° N, 84.05° O; 1.100–1.400 m; Valle Central",
                                     "Posiblemente", "EJEMPLO."], 1):
                c = _plain(ws, r, col, v, fill=MP.GREY)
                c.font = Font(name=MP.FONT, size=10, italic=True)
            r += 1
            for i, sp in enumerate(forecast, 1):
                _plain(ws, r, 1, i, bold=True)
                _plain(ws, r, 2, label(sp), bold=True)
                ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=6)
                ws.row_dimensions[r].height = 32
                r += 1
                img = XLImage(str(pngs[sp]))
                img.width, img.height = 640, 711
                ws.add_image(img, f"B{r}")
                first = r
                for e in sorted(cells[sp], key=lambda e: e["number"]):
                    _plain(ws, r, 3, e["number"])
                    _plain(ws, r, 4, e["where"])
                    _plain(ws, r, 5, "", fill=MP.FILL)
                    dv.add(f"E{r}")
                    ws.row_dimensions[r].height = 30
                    r += 1
                _plain(ws, first, 6, "", fill=MP.FILL)
                ws.merge_cells(start_row=first, start_column=6, end_row=r - 1, end_column=6)
                # room for the map below the five rows
                ws.row_dimensions[r].height = 545 - 30 * len(cells[sp])
                r += 1
            ws.freeze_panes = "A2"

            out = DATA / f"revision_clima_prediccion_{expert}.xlsx"
            wb.save(out)
            print(f"{expert}: {out.name} ({out.stat().st_size // 1024} KB)")
    conn.close()
    key = {"seed": SEED, "climate": letters,
           "forecast": {sp: cells[sp] for sp in forecast}}
    (DATA / "expert_round3_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    print(f"climate: {climate}\nforecast: {forecast}")
    for sp in forecast:
        print(f"  {sp}: " + "; ".join(f"{e['number']} {e['kind']} p={e['p']:.3f} #{e['rank_all']}/{e['n_cand']} {e['near_km']:.0f} km"
                                      for e in sorted(cells[sp], key=lambda e: e['number'])))


if __name__ == "__main__":
    build()
