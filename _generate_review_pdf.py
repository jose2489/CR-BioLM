"""
_generate_review_pdf.py
Genera un PDF profesional en español para revisión por Armando Estrada.

Contiene:
  • Portada con resumen estadístico
  • Tabla de especies encontradas y no encontradas
  • Una página de comparación por especie encontrada (mapa CR-BioLM vs. experto)
  • Nota de regiones inferidas por GBIF cuando aplica

Uso:
    python _generate_review_pdf.py
Salida:
    outputs/revision_mapas_CR-BioLM.pdf
"""
from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import geopandas as gpd

from data.expert_maps import ExpertMapLoader
from data.gbif_extractor import GBIFExtractor
from mpcr_rag import config as rag_config
from mpcr_rag.store import local_store
from utils.distribution_map.geodata import load_regiones_botanicas
from utils.distribution_map.parser import build_ficha
from utils.distribution_map.renderer import (
    _gbif_inferred_regions,
    _resolve_regions_with_qualifiers,
)
import config as main_config

# ── Paths ─────────────────────────────────────────────────────────────────────
EXPERT_DIR = Path(r"C:\Users\Jose\Downloads\Distribución de especies\Distribución de especies")
COMP_DIR   = ROOT / "outputs" / "expert_comparison"
OUT_PDF    = ROOT / "outputs" / "revision_mapas_CR-BioLM.pdf"

# ── Species lists ─────────────────────────────────────────────────────────────
FOUND_SPECIES = [
    "Anacardium excelsum",
    "Anthodiscus chocoensis",
    "Balizia elegans",
    "Batocarpus costaricensis",
    "Caryocar costaricense",
    "Cedrela tonduzii",
    "Ceiba pentandra",
    "Chaunochiton kappleri",
    "Chiangiodendron mexicanum",
    "Copaifera aromatica",
    "Cuphea utriculosa",
    "Hymenolobium mesoamericanum",
    "Lecythis ampla",
    "Macrohasseltia macroterantha",
    "Nelsonia canescens",
    "Oreomunnea pterocarpa",
    "Passiflora platyloba",
    "Passiflora tica",
    "Peltogyne purpurea",
    "Spirotheca rosea",
    "Tachigali versicolor",
]

NOT_FOUND = {
    "Albizia niopoides":
        "Género presente (A. adinocephala, A. carbonaria); especie diferente. "
        "Posiblemente en Tomos VII/VIII.",
    "Astronium graveolens":
        "Género ausente del catálogo (Tomos II–VI). Posiblemente en Tomo I o VII/VIII.",
    "Brosimum utile":
        "Género presente (B. costaricanum, B. guianense, B. lactescens); especie "
        "diferente. Posiblemente en Tomos VII/VIII.",
    "Campnosperma panamensis":
        "Posible variante nomenclatural: catálogo contiene C. panamense "
        "(distinto género ortográfico del epíteto).",
    "Cedrela odorata":
        "Género presente (C. salvadorensis, C. tonduzii); especie diferente. "
        "C. odorata posiblemente en Tomo I o VII/VIII.",
    "Cipura cubensis":
        "Género presente (C. campanulata); especie diferente. "
        "Posiblemente en Tomos VII/VIII.",
    "Cordia gerascanthus":
        "Género bien representado (15 especies en catálogo); esta especie no "
        "aparece en Tomos II–VI. Posiblemente en Tomo I.",
    "Crescentia cujete":
        "Género presente (C. alata); especie diferente. "
        "Posiblemente en Tomo I o VII/VIII.",
    "Dipteryx oleifera":
        "Catálogo contiene D. panamensis, tratada como sinónimo de D. oleifera "
        "en varias floras. Verificar tratamiento taxonómico utilizado.",
    "Passiflora bicornis":
        "Género ampliamente representado (39 especies); esta especie no está en "
        "Tomos II–VI.",
    "Passiflora membranacea":
        "Género ampliamente representado (39 especies); esta especie no está en "
        "Tomos II–VI.",
    "Stachytarpheta calderonii":
        "Género ausente del catálogo. Catálogo contiene S. cayennensis (especie "
        "diferente). Posiblemente en Tomos VII/VIII.",
}

# ── Color palette (light, professional) ───────────────────────────────────────
BG        = "#ffffff"
HEADER_BG = "#0f172a"
HEADER_FG = "#f8fafc"
BODY_FG   = "#1e293b"
LABEL_FG  = "#475569"
LINE_C    = "#e2e8f0"
GBIF_BG   = "#fffbeb"
GBIF_BOR  = "#f59e0b"
GBIF_FG   = "#78350f"
DIST_BG   = "#f8fafc"
FOOTER_FG = "#94a3b8"
FOUND_BG  = "#f0fdf4"
FOUND_ACC = "#16a34a"
MISS_BG   = "#fefce8"
MISS_ACC  = "#ca8a04"
COVER_ACC = "#38bdf8"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_distribution(paragraph: str) -> tuple[str, str]:
    m = re.search(r"m\s*;", paragraph)
    if m:
        hab  = paragraph[: m.start() + 1].strip()
        rest = paragraph[m.end():].strip()
    else:
        hab, rest = "", paragraph
    geo = re.split(r"\bFls?\.\s|\bFr\.\s", rest)[0].strip()
    return hab, geo


def _expert_path(species: str) -> Path | None:
    for ext in (".jpg", ".png"):
        p = EXPERT_DIR / (species + ext)
        if p.exists():
            return p
    return None


def _fmt_months(months: list[int]) -> str:
    names = ["", "ene", "feb", "mar", "abr", "may", "jun",
             "jul", "ago", "set", "oct", "nov", "dic"]
    return ", ".join(names[m] for m in months if 0 < m <= 12) or "—"


def _wrap(text: str, width: int = 90) -> str:
    return "\n".join(textwrap.wrap(text, width))


# ── GBIF inferred region detection ────────────────────────────────────────────

def get_inferred_region_names(
    presencias_gdf,
    map_ficha,
    regiones: gpd.GeoDataFrame,
) -> list[str]:
    """Return canonical names of botanical regions inferred from GBIF clusters."""
    matched_gdf = _resolve_regions_with_qualifiers(map_ficha) if map_ficha.regions else None
    inferred    = _gbif_inferred_regions(presencias_gdf, matched_gdf, regiones)
    if inferred is None or inferred.empty:
        return []
    col = "Nombre" if "Nombre" in inferred.columns else inferred.columns[0]
    return sorted(inferred[col].dropna().unique().tolist())


# ── Page builders ─────────────────────────────────────────────────────────────

def make_cover(pdf: PdfPages, n_found: int, n_total: int) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor(HEADER_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(HEADER_BG)
    ax.axis("off")

    # Top accent line
    ax.axhline(0.94, xmin=0.08, xmax=0.92, color=COVER_ACC, linewidth=2)

    # Title block
    ax.text(0.5, 0.88, "Revisión de Mapas de Distribución",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color=HEADER_FG, transform=ax.transAxes)
    ax.text(0.5, 0.83,
            "Comparación CR-BioLM vs. mapas de referencia expertos\n"
            "Manual de Plantas de Costa Rica",
            ha="center", va="center", fontsize=13, color=COVER_ACC,
            transform=ax.transAxes, linespacing=1.6)

    ax.axhline(0.78, xmin=0.08, xmax=0.92, color="#334155", linewidth=0.8)

    # Stats box
    for i, (label, val, color) in enumerate([
        ("Especies evaluadas",         str(n_total),          "#94a3b8"),
        ("Encontradas en catálogo\n(Tomos II–VI)", str(n_found),    COVER_ACC),
        ("Fuera del catálogo",         str(n_total - n_found), "#f87171"),
    ]):
        xc = 0.18 + i * 0.32
        ax.text(xc, 0.70, val,   ha="center", va="center", fontsize=36,
                fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(xc, 0.63, label, ha="center", va="center", fontsize=9,
                color="#94a3b8", transform=ax.transAxes, linespacing=1.4)

    ax.axhline(0.58, xmin=0.08, xmax=0.92, color="#334155", linewidth=0.8)

    # Methodology note
    note = (
        "Los mapas generados por CR-BioLM se obtienen extrayendo automáticamente el párrafo "
        "de distribución de cada especie del Manual de Plantas de Costa Rica (Tomos II–VI, "
        "5 791 especies indexadas) mediante el módulo MPCR-RAG. El texto es procesado por el "
        "geo-parser determinístico de CR-BioLM, que traduce las notas geográficas a polígonos "
        "fitogeográficos (Zamora 2014) filtrados por rango altitudinal (DEM SRTM 90 m). "
        "Los registros de presencia de GBIF se superponen como capa de validación."
    )
    for j, line in enumerate(textwrap.wrap(note, 78)):
        ax.text(0.5, 0.545 - j * 0.028, line, ha="center", va="center",
                fontsize=9, color="#94a3b8", transform=ax.transAxes)

    ax.axhline(0.36, xmin=0.08, xmax=0.92, color="#334155", linewidth=0.8)

    # Metadata
    meta = [
        ("Preparado para", "M.Sc. Armando Estrada A."),
        ("Preparado por",  "Jose Araya — CR-BioLM / MPCR-RAG"),
        ("Fecha",          "14 de junio de 2026"),
        ("Fuente textual", "Manual de Plantas de CR, Tomos II–VI (MBOT, 2007)"),
        ("Catálogo",       "5 791 especies · SQLite + Pinecone (multilingual-e5-large)"),
    ]
    for j, (lbl, val) in enumerate(meta):
        y = 0.32 - j * 0.044
        ax.text(0.12, y, lbl + ":", ha="left", va="center", fontsize=9,
                color=LABEL_FG, transform=ax.transAxes)
        ax.text(0.42, y, val, ha="left", va="center", fontsize=9,
                fontweight="bold", color=HEADER_FG, transform=ax.transAxes)

    ax.axhline(0.09, xmin=0.08, xmax=0.92, color="#334155", linewidth=0.8)
    ax.text(0.5, 0.055, "CR-BioLM  ·  MPCR-RAG  ·  Tesis de Maestría",
            ha="center", va="center", fontsize=8, color=FOOTER_FG,
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_summary(pdf: PdfPages, found_rows: list[dict], not_found_rows: dict) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.axis("off")

    # Header band
    band = FancyBboxPatch((0, 0.93), 1, 0.07, transform=ax.transAxes,
                          boxstyle="square,pad=0", facecolor=HEADER_BG, clip_on=False)
    ax.add_patch(band)
    ax.text(0.5, 0.965, "Resumen de Especies Evaluadas",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=HEADER_FG, transform=ax.transAxes)

    # ── Found species table ────────────────────────────────────────────────
    ax.text(0.05, 0.905, f"Especies encontradas en el catálogo ({len(found_rows)}/33)",
            ha="left", va="center", fontsize=10, fontweight="bold",
            color=FOUND_ACC, transform=ax.transAxes)
    ax.axhline(0.895, xmin=0.05, xmax=0.95, color=LINE_C, linewidth=0.8)

    headers = ["Especie", "Tomo", "Págs.", "Elevación", "Endémica"]
    col_x   = [0.05, 0.48, 0.55, 0.63, 0.82]
    y0 = 0.880

    for i, h in enumerate(headers):
        ax.text(col_x[i], y0, h, ha="left", va="center", fontsize=7.5,
                fontweight="bold", color=LABEL_FG, transform=ax.transAxes)
    ax.axhline(y0 - 0.01, xmin=0.05, xmax=0.95, color=LINE_C, linewidth=0.5)

    for j, r in enumerate(found_rows):
        y = y0 - 0.022 - j * 0.022
        bg_col = FOUND_BG if j % 2 == 0 else BG
        rect = FancyBboxPatch((0.05, y - 0.009), 0.9, 0.020,
                              transform=ax.transAxes, clip_on=False,
                              boxstyle="square,pad=0", facecolor=bg_col, linewidth=0)
        ax.add_patch(rect)

        elev = (f"{r['elev_min']}–{r['elev_max']} m"
                if r["elev_min"] is not None else "—")
        endemic = "Sí ✓" if r["endemic"] else "No"
        end_col = FOUND_ACC if r["endemic"] else BODY_FG

        ax.text(col_x[0], y + 0.001, r["species"], ha="left", va="center",
                fontsize=7, color=BODY_FG, style="italic", transform=ax.transAxes)
        ax.text(col_x[1], y + 0.001, r["volume"],  ha="left", va="center",
                fontsize=7, color=BODY_FG, transform=ax.transAxes)
        ax.text(col_x[2], y + 0.001, str(r["pages"]), ha="left", va="center",
                fontsize=7, color=BODY_FG, transform=ax.transAxes)
        ax.text(col_x[3], y + 0.001, elev, ha="left", va="center",
                fontsize=7, color=BODY_FG, transform=ax.transAxes)
        ax.text(col_x[4], y + 0.001, endemic, ha="left", va="center",
                fontsize=7, color=end_col, transform=ax.transAxes)

    sep_y = y0 - 0.022 - len(found_rows) * 0.022 - 0.01
    ax.axhline(sep_y, xmin=0.05, xmax=0.95, color=LINE_C, linewidth=0.8)

    # ── Not found species table ────────────────────────────────────────────
    nf_y = sep_y - 0.03
    ax.text(0.05, nf_y, f"Especies fuera del catálogo Tomos II–VI ({len(not_found_rows)}/33)",
            ha="left", va="center", fontsize=10, fontweight="bold",
            color=MISS_ACC, transform=ax.transAxes)
    ax.axhline(nf_y - 0.01, xmin=0.05, xmax=0.95, color=LINE_C, linewidth=0.8)

    ax.text(0.05,  nf_y - 0.022, "Especie",  ha="left", va="center", fontsize=7.5,
            fontweight="bold", color=LABEL_FG, transform=ax.transAxes)
    ax.text(0.40,  nf_y - 0.022, "Nota",     ha="left", va="center", fontsize=7.5,
            fontweight="bold", color=LABEL_FG, transform=ax.transAxes)

    for j, (sp, note) in enumerate(not_found_rows.items()):
        y = nf_y - 0.044 - j * 0.030
        bg_col = MISS_BG if j % 2 == 0 else BG
        rect = FancyBboxPatch((0.05, y - 0.011), 0.9, 0.028,
                              transform=ax.transAxes, clip_on=False,
                              boxstyle="square,pad=0", facecolor=bg_col, linewidth=0)
        ax.add_patch(rect)
        ax.text(0.05, y + 0.003, sp, ha="left", va="center", fontsize=7,
                color=BODY_FG, style="italic", transform=ax.transAxes)
        wrapped = textwrap.wrap(note, 65)
        for k, line in enumerate(wrapped[:2]):
            ax.text(0.40, y + 0.003 - k * 0.012, line, ha="left", va="center",
                    fontsize=6.5, color=LABEL_FG, transform=ax.transAxes)

    # Footer
    ax.axhline(0.04, xmin=0.05, xmax=0.95, color=LINE_C, linewidth=0.5)
    ax.text(0.5, 0.02, "CR-BioLM  ·  MPCR-RAG  ·  14 de junio de 2026",
            ha="center", va="center", fontsize=7, color=FOOTER_FG,
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_species_page(
    pdf: PdfPages,
    sp: str,
    ficha,
    comp_img_path: Path,
    inferred_names: list[str],
    page_num: int,
    total_pages: int,
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG)

    # ── Header band ───────────────────────────────────────────────────────
    ax_hdr = fig.add_axes([0, 0.88, 1, 0.12])
    ax_hdr.set_facecolor(HEADER_BG)
    ax_hdr.axis("off")
    ax_hdr.text(0.03, 0.65, sp, ha="left", va="center", fontsize=16,
                fontweight="bold", color=HEADER_FG, style="italic")
    ax_hdr.text(0.03, 0.22,
                f"Manual de Plantas de CR  ·  Tomo {ficha.volume}  ·  Pág. {ficha.pages}",
                ha="left", va="center", fontsize=9, color=COVER_ACC)
    ax_hdr.text(0.97, 0.65, f"Pág. {page_num} / {total_pages}",
                ha="right", va="center", fontsize=8, color=FOOTER_FG)
    if ficha.endemic_cr:
        ax_hdr.text(0.97, 0.22, "ENDÉMICA DE COSTA RICA",
                    ha="right", va="center", fontsize=8,
                    fontweight="bold", color="#4ade80")

    # ── Comparison image ──────────────────────────────────────────────────
    img_bottom = 0.26 if inferred_names else 0.22
    ax_img = fig.add_axes([0.01, img_bottom, 0.98, 0.61])
    ax_img.axis("off")
    ax_img.set_facecolor(BG)

    img = mpimg.imread(str(comp_img_path))
    ax_img.imshow(img, aspect="auto")

    col_labels = ["CR-BioLM  (RAG + Manual de Plantas)", "Mapa de referencia (experto)"]
    for i, lbl in enumerate(col_labels):
        ax_img.text(0.25 + i * 0.5, -0.04, lbl,
                    ha="center", va="top", fontsize=8.5,
                    color=LABEL_FG, transform=ax_img.transAxes)

    # ── GBIF inferred note ────────────────────────────────────────────────
    if inferred_names:
        ax_gbif = fig.add_axes([0.01, 0.21, 0.98, 0.055])
        ax_gbif.set_facecolor(GBIF_BG)
        ax_gbif.axis("off")
        for spine in ax_gbif.spines.values():
            spine.set_visible(False)
        rect = FancyBboxPatch((0, 0), 1, 1, transform=ax_gbif.transAxes,
                              boxstyle="square,pad=0",
                              facecolor=GBIF_BG, edgecolor=GBIF_BOR,
                              linewidth=1.2, clip_on=False)
        ax_gbif.add_patch(rect)

        names_str = " · ".join(inferred_names)
        ax_gbif.text(0.012, 0.65,
                     "⬡ Regiones inferidas por clúster GBIF (línea discontinua ámbar en mapa):",
                     ha="left", va="center", fontsize=7.5,
                     fontweight="bold", color=GBIF_FG, transform=ax_gbif.transAxes)
        ax_gbif.text(0.012, 0.22, names_str,
                     ha="left", va="center", fontsize=7.5,
                     color=GBIF_FG, transform=ax_gbif.transAxes)

    # ── Metadata grid ─────────────────────────────────────────────────────
    meta_bottom = 0.095
    ax_meta = fig.add_axes([0.01, meta_bottom, 0.98, 0.115])
    ax_meta.set_facecolor(DIST_BG)
    ax_meta.axis("off")
    rect2 = FancyBboxPatch((0, 0), 1, 1, transform=ax_meta.transAxes,
                           boxstyle="square,pad=0",
                           facecolor=DIST_BG, edgecolor=LINE_C,
                           linewidth=0.8, clip_on=False)
    ax_meta.add_patch(rect2)

    elev = (f"{ficha.elev_min}–{ficha.elev_max} m"
            if ficha.elev_min is not None else "—")
    if ficha.elev_outlier_min or ficha.elev_outlier_max:
        lo = ficha.elev_outlier_min or ficha.elev_min
        hi = ficha.elev_outlier_max or ficha.elev_max
        elev += f"  (atípico: {lo}–{hi} m)"

    verts = ", ".join(ficha.vertientes) if ficha.vertientes else "—"
    regs  = ", ".join(ficha.regions)   if ficha.regions   else "—"
    bosq  = ", ".join(ficha.forest_types) if ficha.forest_types else "—"
    hab   = ", ".join(ficha.habits)    if ficha.habits    else "—"
    fl    = _fmt_months(ficha.flowering_months)

    fields_left = [
        ("Elevación",         elev),
        ("Vertiente(s)",      verts),
        ("Tipo(s) de bosque", bosq),
        ("Hábito",            hab),
        ("Floración",         fl),
    ]
    fields_right = [
        ("Regiones Manual", ""),
        ("",                regs),
    ]

    for j, (lbl, val) in enumerate(fields_left):
        y = 0.82 - j * 0.175
        ax_meta.text(0.01, y, lbl + ":", ha="left", va="center", fontsize=7,
                     fontweight="bold", color=LABEL_FG, transform=ax_meta.transAxes)
        ax_meta.text(0.18, y, val, ha="left", va="center", fontsize=7,
                     color=BODY_FG, transform=ax_meta.transAxes)

    # Regions — right side, wrapped
    ax_meta.text(0.50, 0.88, "Regiones (Manual):",
                 ha="left", va="center", fontsize=7,
                 fontweight="bold", color=LABEL_FG, transform=ax_meta.transAxes)
    wrapped_regs = textwrap.wrap(regs, 65)
    for k, line in enumerate(wrapped_regs[:4]):
        ax_meta.text(0.50, 0.68 - k * 0.20, line,
                     ha="left", va="center", fontsize=6.8,
                     color=BODY_FG, transform=ax_meta.transAxes)

    # ── Distribution paragraph ────────────────────────────────────────────
    ax_dist = fig.add_axes([0.01, 0.01, 0.98, 0.075])
    ax_dist.axis("off")
    ax_dist.text(0.008, 0.80, "Párrafo de distribución (Manual de Plantas CR):",
                 ha="left", va="center", fontsize=6.5,
                 fontweight="bold", color=LABEL_FG, transform=ax_dist.transAxes)
    dist_text = ficha.distribution_paragraph or "—"
    wrapped_dist = textwrap.wrap(dist_text, 155)
    for k, line in enumerate(wrapped_dist[:3]):
        ax_dist.text(0.008, 0.55 - k * 0.25, line,
                     ha="left", va="center", fontsize=6,
                     color=BODY_FG, transform=ax_dist.transAxes)

    # Footer line
    ax_foot = fig.add_axes([0, 0, 1, 0.01])
    ax_foot.set_facecolor(HEADER_BG)
    ax_foot.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    print("Cargando datos base …")
    conn     = local_store.connect(rag_config.SQLITE_PATH)
    regiones = load_regiones_botanicas()
    ml       = ExpertMapLoader()
    cr_bounds    = ml.load_country_boundary(main_config.DEFAULT_COUNTRY)
    meso_bounds  = ml.load_mesoamerica_boundary()
    if meso_bounds is None or meso_bounds.empty:
        meso_bounds = cr_bounds
    extractor = GBIFExtractor()

    # Pre-load all fichas and compute inferred regions
    print("Procesando especies y consultando GBIF …")
    species_data: list[dict] = []
    for sp in FOUND_SPECIES:
        print(f"  {sp} …", end=" ", flush=True)
        ficha = local_store.get(conn, sp.replace(" ", "_"))
        if ficha is None:
            print("NO ENCONTRADA — saltando")
            continue

        hab, geo = _split_distribution(ficha.distribution_paragraph or "")
        map_ficha = build_ficha(habitat_raw=hab, geographic_notes=geo, species=sp)

        try:
            meso = extractor.fetch_occurrences_mesoamerica(sp)
            meso = extractor.clean_spatial_outliers(meso, meso_bounds)
            pres = (extractor.clean_spatial_outliers(meso, cr_bounds)
                    if meso is not None and not meso.empty else None)
            if pres is None or pres.empty:
                pres = meso
        except Exception:
            pres = None

        inferred = get_inferred_region_names(pres, map_ficha, regiones)
        n_gbif   = len(pres) if pres is not None else 0
        print(f"{n_gbif} pts GBIF | {len(inferred)} regiones inferidas")

        comp_path = COMP_DIR / sp.replace(" ", "_") / "comparison.png"

        species_data.append({
            "species":   sp,
            "ficha":     ficha,
            "inferred":  inferred,
            "comp_path": comp_path,
        })

    conn.close()

    # Summary rows for page 2
    found_rows = []
    for d in species_data:
        f = d["ficha"]
        found_rows.append({
            "species":  d["species"],
            "volume":   f.volume,
            "pages":    f.pages,
            "elev_min": f.elev_min,
            "elev_max": f.elev_max,
            "endemic":  f.endemic_cr,
        })
    # Sort alphabetically
    found_rows.sort(key=lambda r: r["species"])

    total_pages = 2 + len(species_data)   # cover + summary + species pages

    print(f"\nGenerando PDF ({total_pages} páginas) …")
    with PdfPages(OUT_PDF) as pdf:
        make_cover(pdf, len(species_data), 33)
        make_summary(pdf, found_rows, NOT_FOUND)

        for i, d in enumerate(species_data, start=1):
            print(f"  [{i}/{len(species_data)}] {d['species']}")
            make_species_page(
                pdf,
                sp          = d["species"],
                ficha       = d["ficha"],
                comp_img_path = d["comp_path"],
                inferred_names = d["inferred"],
                page_num    = i + 2,
                total_pages = total_pages,
            )

        # PDF metadata
        info = pdf.infodict()
        info["Title"]   = "Revisión de Mapas de Distribución — CR-BioLM"
        info["Author"]  = "Jose Araya — CR-BioLM / MPCR-RAG"
        info["Subject"] = "Comparación de mapas generados automáticamente vs. expertos"
        info["Keywords"] = "Costa Rica, distribución, botánica, CR-BioLM, MPCR-RAG"

    print(f"\n✓ PDF generado → {OUT_PDF}")


if __name__ == "__main__":
    main()
