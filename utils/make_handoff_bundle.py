"""
Empaqueta los datos NO versionados que hacen falta para correr mpcr_rag localmente.

El código va por GitHub; este script arma el zip con lo que .gitignore excluye
(shapefiles, DEM, catálogo SQLite y cachés). Ejecutar:

    python utils/make_handoff_bundle.py                 # bundle completo
    python utils/make_handoff_bundle.py --sin-cache     # sin cachés (más liviano)
    python utils/make_handoff_bundle.py -o D:/salida    # destino alternativo

El destino por defecto queda FUERA del repo: el bundle pesa ~100 MB y el proyecto
vive en una carpeta sincronizada con Google Drive.

Quien lo reciba solo tiene que descomprimirlo en la raíz del repo clonado; la
estructura de carpetas ya viene correcta.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (ruta relativa, obligatorio, descripción)
REQUERIDOS = [
    ("mpcr_rag/data/fichas.sqlite", True,
     "Catálogo: 5.791 fichas del Manual (species, elevación, regiones, texto)"),
    ("data_raw/regiones_botanicas", True,
     "Regiones Botánicas de Costa Rica (José Araya, 2026) — base del mapa"),
    ("data_raw/topography/altitud_cr.tif", True,
     "DEM ~1 km: la máscara de elevación (relleno cyan) sale de aquí"),
    ("data_raw/vectors/areas_protegidas_v2.shp", True,
     "Áreas Silvestres Protegidas (SINAC)"),
    ("data_raw/Cartografia/IGN_5_limite_Provincial.shp", True,
     "Provincias IGN — capa base, se carga en TODO render"),
    ("data_raw/Cartografia/IGN_5_limite_cantonal.shp", True,
     "Cantones IGN — solo para especies con ámbito cantonal"),
    ("data_raw/Cartografia/IGN_5_limite_distrital.shp", True,
     "Distritos IGN — solo para especies con ámbito distrital"),
]

OPCIONALES = [
    ("mpcr_rag/data/gbif_cache", "Puntos GBIF ya descargados (519 especies). "
     "Sin esto cada mapa nuevo consulta la API de GBIF: funciona igual, pero "
     "más lento y requiere red."),
    ("mpcr_rag/data/maps", "Mapas ya renderizados — útiles como referencia visual."),
]


def _añadir(zf: zipfile.ZipFile, rel: str) -> tuple[int, int]:
    """Agrega un archivo (con sidecars si es .shp) o un directorio completo."""
    p = REPO / rel
    total = n = 0
    if p.is_dir():
        objetivos = [f for f in p.rglob("*") if f.is_file()]
    elif p.suffix.lower() == ".shp":
        objetivos = sorted(p.parent.glob(p.stem + ".*"))   # shp + shx/dbf/prj/cpg
    else:
        objetivos = [p]
    for f in objetivos:
        zf.write(f, f.relative_to(REPO).as_posix())
        total += f.stat().st_size
        n += 1
    return total, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=None,
                    help="Directorio destino (por defecto: junto al repo, fuera de él)")
    ap.add_argument("--sin-cache", action="store_true",
                    help="Omitir los cachés opcionales de GBIF y mapas")
    args = ap.parse_args()

    destino = Path(args.out) if args.out else REPO.parent / "crbiolm_handoff"
    destino.mkdir(parents=True, exist_ok=True)
    zip_path = destino / "mpcr_rag_data_bundle.zip"

    faltantes = [r for r, obl, _ in REQUERIDOS if obl and not (REPO / r).exists()]
    if faltantes:
        print("[ERROR] Faltan archivos obligatorios; no se puede armar el bundle:")
        for f in faltantes:
            print(f"        {f}")
        return 1

    incluidos = list(REQUERIDOS)
    if not args.sin_cache:
        incluidos += [(r, False, d) for r, d in OPCIONALES if (REPO / r).exists()]

    print(f"[INFO] Escribiendo {zip_path}")
    total = archivos = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel, _obl, desc in incluidos:
            if not (REPO / rel).exists():
                print(f"       (omitido, no existe) {rel}")
                continue
            b, n = _añadir(zf, rel)
            total += b
            archivos += n
            print(f"       {rel:52} {b/1e6:7.1f} MB  ({n} archivo/s)")

        zf.writestr("LEEME.txt", _leeme(incluidos))

    print(f"[LISTO] {archivos} archivos, {total/1e6:.1f} MB sin comprimir "
          f"-> {zip_path.stat().st_size/1e6:.1f} MB comprimido")
    print(f"[INFO] Enviar: {zip_path}")
    return 0


def _leeme(incluidos) -> str:
    lineas = [
        "BUNDLE DE DATOS — mpcr_rag",
        "=" * 60,
        "",
        "1. Clonar el repositorio desde GitHub.",
        "2. Descomprimir ESTE zip en la raíz del repo (respeta las carpetas).",
        "3. pip install -r requirements.txt",
        "4. Probar:",
        "     python -c \"from mpcr_rag.store import local_store as L; \"",
        "                \"from mpcr_rag import config as C; \"",
        "                \"c=L.connect(C.SQLITE_PATH); \"",
        "                \"print(c.execute('SELECT COUNT(*) FROM fichas').fetchone())\"",
        "",
        "NO hacen falta llaves de API para consultar el catálogo ni para generar",
        "mapas. Solo las herramientas semantic_search y answer_question requieren",
        "PINECONE_API_KEY y OPENROUTER_API_KEY, y sin ellas responden",
        "'API key not configured' en vez de fallar.",
        "",
        "Contenido:",
    ]
    for rel, _obl, desc in incluidos:
        lineas.append(f"  {rel}")
        lineas.append(f"      {desc}")
    return "\n".join(lineas)


if __name__ == "__main__":
    sys.exit(main())
