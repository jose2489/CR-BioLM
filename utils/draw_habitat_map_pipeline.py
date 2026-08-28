"""
draw_habitat_map_pipeline.py
Generates a visual diagram of the habitat map construction pipeline.
Run with: python utils/draw_habitat_map_pipeline.py
Output: outputs/habitat_map_pipeline.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

# ── Color palette ─────────────────────────────────────────────────────────────
BG       = "#0f172a"
PANEL    = "#1e293b"
BORDER   = "#334155"
TEXT     = "#f1f5f9"
SUBTEXT  = "#94a3b8"
ACCENT   = "#38bdf8"   # sky blue  — main pipeline
GREEN    = "#4ade80"   # data sources
ORANGE   = "#fb923c"   # translation / processing
CYAN     = "#22d3ee"   # optimal habitat output
MUTED    = "#64748b"   # muted polygon
GRAY_BOX = "#1e3a5f"   # intermediate
RED      = "#f87171"   # GBIF points

fig, ax = plt.subplots(figsize=(18, 11), dpi=130)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")


# ── Helpers ───────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, color, alpha=0.92, radius=0.25, zorder=3):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       linewidth=1.2, edgecolor=BORDER,
                       facecolor=color, alpha=alpha, zorder=zorder)
    ax.add_patch(p)

def txt(ax, x, y, s, size=9, color=TEXT, ha="center", va="center",
        bold=False, zorder=5, wrap=False):
    weight = "bold" if bold else "normal"
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, zorder=zorder,
            wrap=wrap)

def arrow(ax, x1, y1, x2, y2, color=ACCENT, lw=1.8, style="->", zorder=4,
          connectionstyle="arc3,rad=0.0"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle=connectionstyle),
                zorder=zorder)

def section_label(ax, x, y, s):
    ax.text(x, y, s, fontsize=7.5, color=SUBTEXT, ha="left", va="center",
            style="italic", zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
txt(ax, 9, 10.55,
    "Construcción del Mapa de Hábitat Potencial — CR-BioLM",
    size=14, bold=True, color=TEXT)
txt(ax, 9, 10.2,
    "Tres fuentes de datos → traducción → intersección espacial → mapa final",
    size=9, color=SUBTEXT)


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN LAYOUT (x anchors)
# Col A: Input sources   1.0 – 4.0
# Col B: Processing      5.5 – 8.5
# Col C: Intermediate    10.0 – 13.0
# Col D: Map output      14.5 – 17.5
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# COL A — INPUT SOURCES
# ─────────────────────────────────────────────────────────────────────────────
section_label(ax, 0.35, 9.7, "① FUENTES DE DATOS")

# A1 — Manual de Plantas (notas_geograficas + altitud)
box(ax, 0.4, 7.8, 3.6, 1.65, GREEN, alpha=0.18)
txt(ax, 2.2, 9.15, "Manual de Plantas de CR", size=9, bold=True, color=GREEN)
txt(ax, 2.2, 8.77, "notas_geograficas (texto libre)", size=8, color=SUBTEXT)
txt(ax, 2.2, 8.45, '"vert. Carib., Cord. de Talamanca"', size=7.5,
    color="#86efac")
txt(ax, 2.2, 8.1,  "elev_min / elev_max  (rango altitudinal)", size=8,
    color=SUBTEXT)

# A2 — Hammel Shapefile
box(ax, 0.4, 5.6, 3.6, 1.8, ORANGE, alpha=0.18)
txt(ax, 2.2, 7.1, "Shapefile Unidades Fitogeográficas", size=9, bold=True,
    color=ORANGE)
txt(ax, 2.2, 6.72, "Nelson Zamora 2014  —  43 polígonos", size=8, color=SUBTEXT)
txt(ax, 2.2, 6.42, "Columna clave: SUBUNIDAD", size=8, color=SUBTEXT)
txt(ax, 2.2, 6.1,  '"8.1", "10.2", "11.3" …', size=8, color="#fdba74")
txt(ax, 2.2, 5.78, "CRS → reproyectado a EPSG:4326", size=7.5, color=SUBTEXT)

# A3 — DEM raster
box(ax, 0.4, 3.6, 3.6, 1.65, CYAN, alpha=0.13)
txt(ax, 2.2, 4.96, "Raster DEM  (altitud_cr.tif)", size=9, bold=True,
    color=CYAN)
txt(ax, 2.2, 4.6,  "SRTM 90 m — Costa Rica completo", size=8, color=SUBTEXT)
txt(ax, 2.2, 4.3,  "Cada píxel = elevación en metros", size=8, color=SUBTEXT)
txt(ax, 2.2, 3.88, "Enmascarado por polígonos\nen el paso de elevación", size=7.5,
    color=SUBTEXT)

# A4 — GBIF presences
box(ax, 0.4, 1.7, 3.6, 1.55, RED, alpha=0.13)
txt(ax, 2.2, 2.97, "GBIF — Registros de Presencia", size=9, bold=True,
    color=RED)
txt(ax, 2.2, 2.62, "Puntos limpios filtrados a Costa Rica", size=8,
    color=SUBTEXT)
txt(ax, 2.2, 2.3,  "GeoDataFrame (lon, lat)", size=8, color=SUBTEXT)
txt(ax, 2.2, 2.0,  "Overlay visual (no afecta polígonos)", size=7.5,
    color=SUBTEXT)

# ─────────────────────────────────────────────────────────────────────────────
# COL B — PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
section_label(ax, 4.95, 9.7, "② TRADUCCIÓN Y PROCESAMIENTO")

# B1 — Translation table (core piece)
box(ax, 4.9, 6.8, 4.0, 2.65, ORANGE, alpha=0.22)
txt(ax, 6.9, 9.13, "TRANSLATION_TABLE", size=9.5, bold=True, color=ORANGE)
txt(ax, 6.9, 8.78, "(texto Manual → códigos SUBUNIDAD)", size=8, color=SUBTEXT)

# Mini table rows
rows = [
    ('"cord.*talamanca"',     '→  11.1 11.2 11.3 11.4 11.5'),
    ('"cord.*central"',       '→  10.1 10.2 10.3 10.4'),
    ('"cord.*guanacaste"',    '→  8.1  8.2  8.3'),
    ('"pen.*osa"',            '→  17.3 17.4'),
    ('"vert.*carib"',         '→  FILTRO  (no suma)'),
    ('"vert.*pac"',           '→  FILTRO  (no suma)'),
]
for i, (pat, codes) in enumerate(rows):
    yy = 8.38 - i * 0.25
    ax.text(5.05, yy, pat,   fontsize=7, color="#fdba74", ha="left", va="center")
    ax.text(7.35, yy, codes, fontsize=7, color="#86efac", ha="left", va="center")

txt(ax, 6.9, 7.07, "~30 patrones regex; más específicos primero", size=7.5,
    color=SUBTEXT)

# B2 — Phase 1
box(ax, 4.9, 5.0, 4.0, 1.5, GRAY_BOX, alpha=0.8)
txt(ax, 6.9, 6.17, "Fase 1 — Coincidencia de Lugar", size=9, bold=True,
    color=ACCENT)
txt(ax, 6.9, 5.82, "Regex sobre texto normalizado (sin tildes,\nminúsculas)", size=8,
    color=SUBTEXT)
txt(ax, 6.9, 5.28, "Acumula todos los códigos SUBUNIDAD\nque dispara cada patrón de lugar", size=8,
    color=SUBTEXT)

# B3 — Phase 2
box(ax, 4.9, 3.3, 4.0, 1.45, GRAY_BOX, alpha=0.8)
txt(ax, 6.9, 4.44, "Fase 2 — Filtro de Vertiente", size=9, bold=True,
    color=ACCENT)
txt(ax, 6.9, 4.08, "Si solo menciona 'vert. Carib.' →\nelimina sufijos Pacífico (.1, .2) de 8.x–11.x", size=8,
    color=SUBTEXT)
txt(ax, 6.9, 3.57, "Si solo menciona 'vert. Pac.' →\nelimina sufijos Caribe (.2, .3, .4) de 8.x–11.x", size=8,
    color=SUBTEXT)

# B4 — Elevation filter label
box(ax, 4.9, 1.7, 4.0, 1.3, GRAY_BOX, alpha=0.8)
txt(ax, 6.9, 2.58, "Enmascaramiento por Elevación", size=9, bold=True,
    color=CYAN)
txt(ax, 6.9, 2.2,  "DEM recortado a polígonos coincidentes\npíxel a píxel: ¿ elev_min ≤ DEM ≤ elev_max ?", size=8,
    color=SUBTEXT)
txt(ax, 6.9, 1.87, "Rango principal + rango atípico opcional", size=8,
    color=SUBTEXT)

# ─────────────────────────────────────────────────────────────────────────────
# COL C — INTERMEDIATE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
section_label(ax, 9.5, 9.7, "③ CAPAS INTERMEDIAS")

# C1 — Matched codes set
box(ax, 9.5, 7.9, 3.5, 1.65, ORANGE, alpha=0.18)
txt(ax, 11.25, 9.22, "Conjunto de códigos coincidentes", size=9, bold=True,
    color=ORANGE)
txt(ax, 11.25, 8.87, 'p. ej.: {"11.3","11.4","10.2","4.1","4.2"}', size=8,
    color="#fdba74")
txt(ax, 11.25, 8.5,  "Si vacío → fallback: todos los 43 polígonos\n(sin cobertura geográfica en el Manual)", size=8,
    color=SUBTEXT)

# C2 — Three rendered layers
box(ax, 9.5, 4.3, 3.5, 3.25, PANEL, alpha=0.95)
txt(ax, 11.25, 7.25, "Tres capas renderizadas", size=9, bold=True,
    color=TEXT)

# Layer indicators
ly = [
    ("#4b5563", "Capa 1 — Polígonos NO coincidentes",   "color gris oscuro (contexto)"),
    ("#8a6e3e", "Capa 2 — Polígonos coincidentes",       "fuera del rango altitudinal → color tenue"),
    ("#22d3ee", "Capa 3 — Zona de hábitat óptimo",       "coincide geografía + elevación → cian saturado"),
]
for i, (col, label, desc) in enumerate(ly):
    yy = 6.85 - i * 0.9
    circ = plt.Circle((9.9, yy - 0.04), 0.15, color=col, zorder=6)
    ax.add_patch(circ)
    txt(ax, 10.1, yy,        label, size=8, bold=True,  color=TEXT,    ha="left")
    txt(ax, 10.1, yy - 0.3,  desc,  size=7.5, color=SUBTEXT, ha="left")

# outlier band
txt(ax, 11.25, 4.6,
    "Banda atípica opcional (hatch ////):\nrango extendido de registros GBIF raros",
    size=7.5, color=SUBTEXT)

# C3 — GBIF overlay
box(ax, 9.5, 2.55, 3.5, 1.45, RED, alpha=0.13)
txt(ax, 11.25, 3.65, "Capa GBIF (overlay)", size=9, bold=True, color=RED)
txt(ax, 11.25, 3.3,  "Puntos rojos sobre el mapa terminado", size=8,
    color=SUBTEXT)
txt(ax, 11.25, 3.0,  "n presencias indicado en leyenda", size=8, color=SUBTEXT)
txt(ax, 11.25, 2.72, "Validación visual: ¿coincide con la predicción?", size=7.5,
    color=SUBTEXT)

# ─────────────────────────────────────────────────────────────────────────────
# COL D — FINAL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
section_label(ax, 14.05, 9.7, "④ SALIDA")

# D1 — Map mockup (simplified schematic)
box(ax, 14.0, 2.4, 3.7, 7.1, "#0c1a2e", alpha=0.97, radius=0.3)

# Schematic CR silhouette regions (very rough)
cr_patches = [
    # unmatched gray
    (FancyBboxPatch((14.2, 7.6), 3.2, 1.4, boxstyle="round,pad=0,rounding_size=0.1",
                    facecolor="#2e3440", edgecolor="#4c566a", lw=0.4, alpha=0.7, zorder=4)),
    (FancyBboxPatch((14.2, 6.2), 1.4, 1.2, boxstyle="round,pad=0,rounding_size=0.1",
                    facecolor="#2e3440", edgecolor="#4c566a", lw=0.4, alpha=0.7, zorder=4)),
    # muted matched (outside elev range)
    (FancyBboxPatch((15.8, 6.2), 1.6, 1.2, boxstyle="round,pad=0,rounding_size=0.1",
                    facecolor="#3a5248", edgecolor="#4c7a6a", lw=0.5, alpha=0.85, zorder=4)),
    (FancyBboxPatch((14.2, 4.7), 3.2, 1.3, boxstyle="round,pad=0,rounding_size=0.1",
                    facecolor="#3a5248", edgecolor="#4c7a6a", lw=0.5, alpha=0.85, zorder=4)),
    # optimal habitat cyan
    (FancyBboxPatch((15.5, 5.85), 1.65, 0.85, boxstyle="round,pad=0,rounding_size=0.1",
                    facecolor="#22d3ee", edgecolor="#67e8f9", lw=0.6, alpha=0.82, zorder=5)),
    (FancyBboxPatch((14.4, 5.2), 2.0, 0.45, boxstyle="round,pad=0,rounding_size=0.1",
                    facecolor="#22d3ee", edgecolor="#67e8f9", lw=0.6, alpha=0.82, zorder=5)),
    # lower region
    (FancyBboxPatch((14.2, 2.65), 3.2, 1.8, boxstyle="round,pad=0,rounding_size=0.1",
                    facecolor="#2e3440", edgecolor="#4c566a", lw=0.4, alpha=0.7, zorder=4)),
]
for p in cr_patches:
    ax.add_patch(p)

# GBIF dots
np.random.seed(42)
gx = np.random.uniform(15.4, 17.1, 18)
gy = np.random.uniform(5.7,  6.95, 18)
ax.scatter(gx, gy, s=14, c=RED, edgecolors="#ffaaaa", linewidths=0.4,
           alpha=0.9, zorder=10)

# Map title
txt(ax, 15.85, 9.35, "Calanthe calanthoides", size=8.5, bold=True, color=TEXT)
txt(ax, 15.85, 9.1,  "Hábitat Potencial · 400–1400 m", size=7, color=SUBTEXT)

# Mini legend inside map
leg_items = [
    ("#2e3440", "Fuera de rango geográfico"),
    ("#3a5248", "Zona geográfica — fuera de elevación"),
    ("#22d3ee", "Hábitat óptimo (geografía + elevación)"),
    (RED,       "Presencias GBIF"),
]
for i, (col, lbl) in enumerate(leg_items):
    yleg = 4.55 - i * 0.37
    circ = plt.Circle((14.45, yleg), 0.1, color=col, zorder=8)
    ax.add_patch(circ)
    txt(ax, 14.62, yleg, lbl, size=6.5, color=SUBTEXT, ha="left")

txt(ax, 15.85, 2.6, "generate_habitat_map()  →  .png  (140 dpi)", size=7,
    color=SUBTEXT)


# ─────────────────────────────────────────────────────────────────────────────
# ARROWS  (left to right flow)
# ─────────────────────────────────────────────────────────────────────────────

# A1 Manual → B1 Translation table
arrow(ax, 4.0, 8.62, 4.9, 8.25, color=GREEN, lw=1.8)
# A2 Shapefile → B2 Phase 1
arrow(ax, 4.0, 6.75, 4.9, 5.8,  color=ORANGE, lw=1.8)
# A3 DEM → B4 Elevation
arrow(ax, 4.0, 4.5,  4.9, 2.35, color=CYAN, lw=1.8)
# A4 GBIF → map directly (skip B, goes to C overlay)
arrow(ax, 4.0, 2.5,  9.5, 3.1,  color=RED, lw=1.4,
      connectionstyle="arc3,rad=-0.25")

# B1 Translation table → B2 Phase 1
arrow(ax, 6.9, 6.8, 6.9, 6.5, color=ORANGE, lw=1.6)
# B2 Phase 1 → B3 Phase 2
arrow(ax, 6.9, 5.0, 6.9, 4.75, color=ACCENT, lw=1.6)
# B3 Phase 2 → C1 matched codes
arrow(ax, 8.9, 4.05, 9.5, 8.72, color=ACCENT, lw=1.6,
      connectionstyle="arc3,rad=-0.3")
# B4 elevation → C2 layers
arrow(ax, 8.9, 2.35, 9.5, 5.4, color=CYAN, lw=1.6,
      connectionstyle="arc3,rad=-0.2")

# C1 matched codes → C2 layers
arrow(ax, 11.25, 7.9, 11.25, 7.55, color=ORANGE, lw=1.6)

# C2 layers → D (map)
arrow(ax, 13.0, 5.92, 14.0, 6.5, color=ACCENT, lw=2.0)
# C3 GBIF → D (map)
arrow(ax, 13.0, 3.27, 14.0, 4.2, color=RED, lw=1.6)


# ─────────────────────────────────────────────────────────────────────────────
# CALLOUT: vertiente filter detail
# ─────────────────────────────────────────────────────────────────────────────
box(ax, 4.9, 0.25, 8.6, 1.25, "#1e1a2e", alpha=0.9, radius=0.2)
txt(ax, 5.05, 1.3, "Lógica del Filtro de Vertiente (Fase 2):", size=8, bold=True,
    color=ACCENT, ha="left")
txt(ax, 5.05, 1.0,
    "Para cordilleras 8.x–11.x:  sufijos Pacífico = .1 (.2 en 11.x)  |  sufijos Caribe = .2/.3/.4 (.3/.4 en 11.x)",
    size=7.5, color=SUBTEXT, ha="left")
txt(ax, 5.05, 0.7,
    '"solo vert. Carib." → elimina sufijos Pacífico  |  "solo vert. Pac." → elimina sufijos Caribe',
    size=7.5, color=TEXT, ha="left")
txt(ax, 5.05, 0.4,
    '"ambas verts." o sin mención de vertiente → se conservan todos los sufijos de la cordillera',
    size=7.5, color=SUBTEXT, ha="left")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
txt(ax, 9, 0.12,
    "CR-BioLM  ·  utils/map_gen/habitat_map.py  ·  generate_habitat_map(species, geographic_notes, elev_min, elev_max, presencias_gdf)",
    size=7, color="#475569")


# ── Save ──────────────────────────────────────────────────────────────────────
out = Path("outputs/habitat_map_pipeline.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout(pad=0.4)
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"[OK] Saved → {out}")
