"""
compare_expert_maps.py
Generate our habitat maps for the 21 species found in the catalog and produce
a side-by-side PNG comparing each against the expert-drawn reference map.

Usage:
    python utils/compare_expert_maps.py

Output:
    outputs/expert_comparison/<Species_name>/comparison.png
"""
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ── project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.expert_maps import ExpertMapLoader
from data.gbif_extractor import GBIFExtractor
from mpcr_rag.store import local_store
from mpcr_rag import config as rag_config
from utils.distribution_map.parser import build_ficha
from utils.distribution_map.renderer import generate_distribution_map
import config as main_config

# ── expert maps folder ────────────────────────────────────────────────────────
EXPERT_DIR = Path(r"C:\Users\Jose\Downloads\Distribución de especies\Distribución de especies")

# 21 species found in the catalog (exact names from the lookup)
SPECIES = [
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

OUT_DIR = ROOT / "outputs" / "expert_comparison"


def _split_distribution(paragraph: str) -> tuple[str, str]:
    """Mirror of mpcr_rag.ingest.field_extractor._split_distribution."""
    m = re.search(r"m\s*;", paragraph)
    if m:
        habitat_raw = paragraph[: m.start() + 1].strip()
        rest = paragraph[m.end():].strip()
    else:
        habitat_raw, rest = "", paragraph
    geo = re.split(r"\bFls?\.\s|\bFr\.\s", rest)[0].strip()
    return habitat_raw, geo


def _expert_path(species: str) -> Path | None:
    """Find the expert image by trying .jpg then .png."""
    for ext in (".jpg", ".png"):
        p = EXPERT_DIR / (species + ext)
        if p.exists():
            return p
    return None


def make_comparison(species: str, our_map: Path, expert_map: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=120)
    fig.patch.set_facecolor("#0f172a")

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0f172a")

    our_img = mpimg.imread(str(our_map))
    exp_img = mpimg.imread(str(expert_map))

    axes[0].imshow(our_img)
    axes[0].set_title("CR-BioLM (RAG + Manual)",
                      color="#f1f5f9", fontsize=12, pad=8, fontweight="bold")

    axes[1].imshow(exp_img)
    axes[1].set_title("Mapa de referencia (experto)",
                      color="#f1f5f9", fontsize=12, pad=8, fontweight="bold")

    fig.suptitle(f"{species}  —  comparación de distribución",
                 color="#38bdf8", fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)


def main() -> None:
    conn = local_store.connect(rag_config.SQLITE_PATH)

    map_loader = ExpertMapLoader()
    cr_bounds  = map_loader.load_country_boundary(main_config.DEFAULT_COUNTRY)
    meso_bounds = map_loader.load_mesoamerica_boundary()
    if meso_bounds is None or meso_bounds.empty:
        meso_bounds = cr_bounds

    extractor = GBIFExtractor()

    errors: list[tuple[str, str]] = []
    ok = 0

    for sp in SPECIES:
        print(f"\n=== {sp} ===")

        # ── fetch ficha from SQLite ───────────────────────────────────────────
        ficha_rec = local_store.get(conn, sp.replace(" ", "_"))
        if ficha_rec is None:
            print(f"  SKIP — not in SQLite (shouldn't happen)")
            errors.append((sp, "not found in SQLite"))
            continue

        # ── expert reference map ──────────────────────────────────────────────
        expert_path = _expert_path(sp)
        if expert_path is None:
            print(f"  SKIP — no expert map found in {EXPERT_DIR}")
            errors.append((sp, "expert map missing"))
            continue

        # ── split distribution paragraph → map ficha ──────────────────────────
        dist = ficha_rec.distribution_paragraph or ""
        habitat_raw, geo_notes = _split_distribution(dist)
        print(f"  habitat_raw : {habitat_raw[:80]!r}")
        print(f"  geo_notes   : {geo_notes[:80]!r}")

        map_ficha = build_ficha(
            habitat_raw=habitat_raw,
            geographic_notes=geo_notes,
            species=sp,
        )
        print(f"  regions     : {[r.canonical_name for r in map_ficha.regions]}")
        print(f"  elevation   : {map_ficha.elevation}")

        # ── GBIF occurrences ──────────────────────────────────────────────────
        try:
            meso = extractor.fetch_occurrences_mesoamerica(sp)
            meso = extractor.clean_spatial_outliers(meso, meso_bounds)
            pres = (
                extractor.clean_spatial_outliers(meso, cr_bounds)
                if meso is not None and not meso.empty
                else None
            )
            if pres is None or pres.empty:
                pres = meso
            print(f"  GBIF        : {len(pres) if pres is not None else 0} pts")
        except Exception as e:
            print(f"  GBIF err    : {e}")
            pres = None

        # ── generate our map ──────────────────────────────────────────────────
        sp_slug  = sp.replace(" ", "_")
        sp_dir   = OUT_DIR / sp_slug
        our_path = sp_dir / "our_map.png"
        comp_path = sp_dir / "comparison.png"

        try:
            generate_distribution_map(map_ficha, our_path, presencias_gdf=pres)
            print(f"  map         : {our_path}")
        except Exception as e:
            print(f"  MAP ERR     : {e}")
            errors.append((sp, f"map generation: {e}"))
            continue

        # ── side-by-side comparison ───────────────────────────────────────────
        try:
            make_comparison(sp, our_path, expert_path, comp_path)
            print(f"  comparison  : {comp_path}")
            ok += 1
        except Exception as e:
            print(f"  COMP ERR    : {e}")
            errors.append((sp, f"comparison: {e}"))

    conn.close()

    print("\n" + "─" * 60)
    print(f"Done. {ok}/{len(SPECIES)} comparisons generated → {OUT_DIR}")
    if errors:
        print("\nFailed:")
        for sp, msg in errors:
            print(f"  {sp}: {msg}")


if __name__ == "__main__":
    main()
