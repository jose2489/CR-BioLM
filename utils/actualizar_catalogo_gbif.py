"""
actualizar_catalogo_gbif.py
----------------------------
Regenera el catálogo de ocurrencias de plantas en Costa Rica desde GBIF.
Usa la API de facetas (facets) para obtener conteos por especie directamente,
sin necesidad de paginar todos los registros individuales.

Uso:
    python utils/actualizar_catalogo_gbif.py
    python utils/actualizar_catalogo_gbif.py --output utils/catalogo_ocurrencias_gbif.csv
"""

import argparse
import csv
import os
import time
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
COUNTRY_CODE   = "CR"
KINGDOM_KEY    = 6             # Plantae en GBIF
FACET_LIMIT    = 500            # especies por página de facetas (conservador para evitar 503s)
SLEEP_BETWEEN  = 1.0            # segundos entre peticiones
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "catalogo_ocurrencias_gbif.csv")
GBIF_API       = "https://api.gbif.org/v1/occurrence/search"


# ---------------------------------------------------------------------------
# Lógica principal
# ---------------------------------------------------------------------------
def fetch_species_counts(verbose=True):
    """
    Usa el endpoint de facetas de GBIF para obtener conteos por speciesKey.
    Luego resuelve los nombres científicos. Mucho más rápido que paginar 1M+ registros.
    """
    conteos = {}
    offset  = 0

    print(f"[INFO] Consultando GBIF (facetas) — Reino: Plantae, País: {COUNTRY_CODE}")

    while True:
        params = {
            "country":       COUNTRY_CODE,
            "kingdomKey":    KINGDOM_KEY,
            "hasCoordinate": "true",
            "limit":         0,  # no queremos registros individuales, solo facetas
            "facet":         "speciesKey",
            "facetLimit":    FACET_LIMIT,
            "facetOffset":   offset,
        }

        try:
            resp = requests.get(GBIF_API, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] Fallo en facetOffset={offset}: {e}")
            print("[INFO]  Reintentando en 10 segundos...")
            time.sleep(10)
            continue

        facetas = data.get("facets", [])
        if not facetas:
            break

        counts_list = facetas[0].get("counts", [])
        if not counts_list:
            break

        for entry in counts_list:
            species_key = entry["name"]
            count       = entry["count"]
            conteos[species_key] = count

        total_so_far = len(conteos)
        if verbose:
            print(f"  → Especies acumuladas: {total_so_far:,}  "
                  f"(página facetOffset={offset})", flush=True)

        # Si devolvió menos del límite, ya no hay más
        if len(counts_list) < FACET_LIMIT:
            break

        offset += FACET_LIMIT
        time.sleep(SLEEP_BETWEEN)

    print(f"[INFO] Se obtuvieron conteos para {len(conteos):,} speciesKeys.")
    return conteos


def _resolver_un_nombre(species_key):
    """Resuelve un speciesKey → nombre científico."""
    try:
        resp = requests.get(
            f"https://api.gbif.org/v1/species/{species_key}",
            timeout=15
        )
        resp.raise_for_status()
        info = resp.json()
        nombre = info.get("species") or info.get("canonicalName") or info.get("scientificName", f"key_{species_key}")
        return species_key, nombre
    except Exception:
        return species_key, f"speciesKey_{species_key}"


def resolver_nombres(conteos_por_key, verbose=True, max_workers=20):
    """
    Resuelve speciesKey → nombre científico usando peticiones concurrentes.
    Con 20 hilos en paralelo es ~20x más rápido que secuencial.
    """
    total = len(conteos_por_key)
    print(f"[INFO] Resolviendo nombres científicos para {total:,} especies ({max_workers} hilos)...")

    key_to_nombre = {}
    resueltos = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_resolver_un_nombre, key): key
            for key in conteos_por_key.keys()
        }

        for future in as_completed(futures):
            species_key, nombre = future.result()
            key_to_nombre[species_key] = nombre
            resueltos += 1
            if verbose and resueltos % 1000 == 0:
                print(f"  → Resueltos: {resueltos:,} / {total:,} ({resueltos/total*100:.0f}%)", flush=True)

    # Reconstruir conteos con nombres
    conteos_por_nombre = {}
    for key, count in conteos_por_key.items():
        nombre = key_to_nombre.get(key, f"speciesKey_{key}")
        conteos_por_nombre[nombre] = conteos_por_nombre.get(nombre, 0) + count

    return conteos_por_nombre


def guardar_catalogo(conteos, ruta_salida):
    """Escribe el CSV ordenado de mayor a menor ocurrencias."""
    ordenado = sorted(conteos.items(), key=lambda x: x[1], reverse=True)

    os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)), exist_ok=True)

    with open(ruta_salida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["especie", "ocurrencias"])
        writer.writerows(ordenado)

    return len(ordenado)


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Actualiza el catálogo GBIF de plantas de CR.")
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Ruta del CSV de salida (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    inicio = datetime.now()
    print(f"[INICIO] {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO]   Salida → {args.output}")
    print("-" * 60)

    # Paso 1: obtener conteos por speciesKey (rápido, ~10 peticiones)
    conteos_keys = fetch_species_counts()

    # Paso 2: resolver speciesKey → nombre científico (~5-10 min para ~10k especies)
    conteos_nombres = resolver_nombres(conteos_keys)

    # Paso 3: guardar
    print(f"\n[INFO] Guardando {len(conteos_nombres):,} especies en CSV...")
    n = guardar_catalogo(conteos_nombres, args.output)

    duracion = datetime.now() - inicio
    print(f"[EXITO] {n:,} especies guardadas en: {args.output}")
    print(f"[INFO]  Tiempo total: {duracion}")
    print(f"[INFO]  Top 10:")
    top10 = sorted(conteos_nombres.items(), key=lambda x: x[1], reverse=True)[:10]
    for nombre, cnt in top10:
        print(f"         {nombre:<45} {cnt:>6}")
