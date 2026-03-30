import random
import time


def muestrear_especies_gbif(ruta_csv, n, min_ocurrencias=50, seed=42):
    especies_validas = []

    with open(ruta_csv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Ignorar encabezado o líneas basura
            if not line or line.startswith("---"):
                continue

            # Separar por el último espacio (clave)
            parts = line.rsplit(maxsplit=1)

            if len(parts) != 2:
                continue

            nombre, ocurrencias = parts

            try:
                ocurrencias = int(ocurrencias)
            except ValueError:
                continue

            if ocurrencias >= min_ocurrencias:
                especies_validas.append(nombre)

    if len(especies_validas) < n:
        raise ValueError(
            f"No hay suficientes especies con >= {min_ocurrencias} ocurrencias. "
            f"Disponibles: {len(especies_validas)}"
        )

    random.seed(time.time())
    seleccion = random.sample(especies_validas, n)

    return seleccion