"""
Selección de variables climáticas y particionamiento espacial para el SDM.

Dos problemas concretos que este módulo resuelve, ambos detectados el 2026-09-04
sobre los rasters de Costa Rica:

1. COLINEALIDAD. En el trópico las 7 variables de temperatura colapsan en un solo
   eje —bio_1 ~ bio_10 ~ bio_11 con r = 1.000— que es esencialmente la ELEVACIÓN.
   Lo mismo ocurre en precipitación (bio_14 ~ bio_17 con r = 0.991;
   bio_12 ~ bio_15 con r = -0.885). Entrenar con variables redundantes hace que la
   atribución de SHAP se reparta de forma arbitraria entre gemelas: por eso el
   "factor limitante" alternaba entre bio_16 y bio_19 sin relación con la especie.

2. AUTOCORRELACIÓN ESPACIAL. Una partición aleatoria pone celdas vecinas —casi
   idénticas en clima— a ambos lados del split, así que el modelo se evalúa sobre
   filas que ya vio. Es una de las causas del AUC ~1.0. La partición por BLOQUES
   espaciales retiene bloques completos y da una estimación honesta.
"""
from __future__ import annotations

import re

import numpy as np

# Orden de preferencia para la selección voraz. Encabeza bio_1 porque la
# temperatura media anual es el mejor proxy de elevación y hasta ahora estaba
# AUSENTE del modelo: con un país de 0 a 3820 m, omitirla impedía descubrir
# cualquier límite térmico o altitudinal. Después va precipitación anual, que es
# la más interpretable al redactar una respuesta ("requiere >3000 mm/año").
# Editar esta lista es una decisión metodológica explícita, no un detalle interno.
DEFAULT_PRIORITY = [
    "bio_1",    # temperatura media anual   (eje térmico / elevación)
    "bio_12",   # precipitación anual       (agua total; muy interpretable)
    "bio_2",    # rango diurno medio        (carácter de tierras altas vs bajas)
    "bio_15",   # estacionalidad de precip. (señal de vertiente Pac./Carib.)
    "bio_3",    # isotermalidad
    "bio_18",   # precipitación del trimestre cálido
    "bio_4",    # estacionalidad térmica
    "bio_17",   # precipitación del trimestre seco
    "bio_19",   # precipitación del trimestre frío
]

CORR_THRESHOLD = 0.80

LABELS = {
    "bio_1":  "temperatura media anual",
    "bio_2":  "rango diurno medio de temperatura",
    "bio_3":  "isotermalidad",
    "bio_4":  "estacionalidad térmica",
    "bio_5":  "temperatura máxima del mes más cálido",
    "bio_6":  "temperatura mínima del mes más frío",
    "bio_7":  "rango anual de temperatura",
    "bio_8":  "temperatura media del trimestre más húmedo",
    "bio_9":  "temperatura media del trimestre más seco",
    "bio_10": "temperatura media del trimestre más cálido",
    "bio_11": "temperatura media del trimestre más frío",
    "bio_12": "precipitación anual",
    "bio_13": "precipitación del mes más húmedo",
    "bio_14": "precipitación del mes más seco",
    "bio_15": "estacionalidad de la precipitación",
    "bio_16": "precipitación del trimestre más húmedo",
    "bio_17": "precipitación del trimestre más seco",
    "bio_18": "precipitación del trimestre más cálido",
    "bio_19": "precipitación del trimestre más frío",
}

UNITS = {**{f"bio_{i}": "°C" for i in (1, 2, 5, 6, 7, 8, 9, 10, 11)},
         **{f"bio_{i}": "mm" for i in (12, 13, 14, 16, 17, 18, 19)},
         "bio_3": "%", "bio_4": "°C x100", "bio_15": "CV %"}


def _var_num(name: str) -> int:
    m = re.search(r"bio_(\d+)", name)
    return int(m.group(1)) if m else 0


def select_decorrelated(raster_paths: dict, priority=None,
                        threshold: float = CORR_THRESHOLD, verbose: bool = True):
    """
    Selección voraz de variables poco correlacionadas a partir de los rasters ya
    recortados a la región de estudio.

    ``raster_paths``: {"bio_1": ruta, ...} tal como lo devuelve ClimateLoader.
    Devuelve la lista de nombres seleccionados, en orden de prioridad.
    """
    import rasterio

    priority = priority or DEFAULT_PRIORITY
    disponibles = [v for v in priority if v in raster_paths]
    if not disponibles:
        return []

    datos, mascaras = {}, []
    for v in disponibles:
        with rasterio.open(raster_paths[v]) as s:
            a = s.read(1).astype("float64")
            nod = s.nodata if s.nodata is not None else -3.4e38
        m = np.isfinite(a) & (a != nod) & (a > -1e30)
        datos[v] = a
        mascaras.append(m)

    comun = np.logical_and.reduce(mascaras)
    n_celdas = int(comun.sum())
    if n_celdas < 2:
        return disponibles

    vectores = {v: datos[v][comun] for v in disponibles}

    seleccion: list[str] = []
    for cand in disponibles:
        if all(abs(np.corrcoef(vectores[cand], vectores[s])[0, 1]) < threshold
               for s in seleccion):
            seleccion.append(cand)

    if verbose:
        print(f"[INFO] Selección de variables climáticas sobre {n_celdas} celdas "
              f"(|r| < {threshold}): {len(seleccion)} de {len(disponibles)} candidatas")
        for v in seleccion:
            print(f"       {v:7} {LABELS.get(v, '')}")
        descartadas = [v for v in disponibles if v not in seleccion]
        if descartadas:
            print(f"       descartadas por colinealidad: {', '.join(descartadas)}")
    return seleccion


def spatial_block_split(matriz, *, n_blocks: int = 8, test_frac: float = 0.25,
                        seed: int = 42, lon_col: str = "lon", lat_col: str = "lat",
                        clase_col: str = "clase"):
    """
    Partición por BLOQUES espaciales, ESTRATIFICADA por clase.

    Evita que celdas vecinas (casi idénticas climáticamente) queden repartidas
    entre entrenamiento y prueba, que es lo que infla el AUC bajo una partición
    aleatoria.

    La estratificación es imprescindible: las presencias suelen estar agrupadas en
    un área pequeña (p. ej. 45 registros de *Chaunochiton kappleri* concentrados en
    Osa/Coto Brus), así que sortear bloques sin estratificar deja el conjunto de
    prueba SIN presencias — el ROC-AUC queda indefinido (nan) y la exactitud sale
    1.0 de forma trivial. Aquí se sortean por separado los bloques CON presencia y
    los que solo tienen fondo, de modo que ambos lados reciben las dos clases.

    Devuelve (idx_train, idx_test) como arreglos booleanos.
    """
    rng = np.random.default_rng(seed)
    lon = np.asarray(matriz[lon_col], dtype=float)
    lat = np.asarray(matriz[lat_col], dtype=float)
    y = np.asarray(matriz[clase_col], dtype=int)

    def _bins(v):
        lo, hi = np.nanmin(v), np.nanmax(v)
        if hi <= lo:
            return np.zeros_like(v, dtype=int)
        return np.clip(((v - lo) / (hi - lo) * n_blocks).astype(int), 0, n_blocks - 1)

    bloque = _bins(lon) * n_blocks + _bins(lat)

    con_presencia = {b for b in np.unique(bloque) if y[bloque == b].sum() > 0}
    solo_fondo = [b for b in np.unique(bloque) if b not in con_presencia]
    con_presencia = list(con_presencia)

    def _muestra_test(bloques):
        if not bloques:
            return []
        bloques = list(bloques)
        rng.shuffle(bloques)
        n = max(1, int(round(len(bloques) * test_frac))) if len(bloques) > 1 else 0
        return bloques[:n]

    # Con un solo bloque de presencia no se puede reservar ninguno sin dejar el
    # entrenamiento sin presencias: se avisa y se cae a una partición aleatoria.
    if len(con_presencia) < 2:
        print(f"[WARN] Solo {len(con_presencia)} bloque(s) espacial(es) contienen "
              f"presencias: la especie está demasiado agrupada para una partición "
              f"por bloques. Se usa partición aleatoria estratificada (el AUC será "
              f"optimista por autocorrelación espacial).")
        test = np.zeros(len(y), dtype=bool)
        for cls in (0, 1):
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            test[idx[:max(1, int(round(len(idx) * test_frac)))]] = True
        return ~test, test

    bloques_test = set(_muestra_test(con_presencia)) | set(_muestra_test(solo_fondo))
    test = np.isin(bloque, list(bloques_test))
    return ~test, test


def climate_envelope(matriz, variables, *, clase_col: str = "clase") -> dict:
    """
    Envolvente climática observada en los sitios de PRESENCIA.

    Sin esto, el prompt solo recibe el NOMBRE de la variable limitante y el modelo
    no puede describir un clima a partir de datos: recita la etiqueta y rellena el
    resto con conocimiento previo. Devuelve, por variable, mediana y P10-P90 con
    unidades, listo para inyectarse en FUENTE 1.
    """
    pres = matriz[matriz[clase_col] == 1]
    out = {}
    for v in variables:
        if v not in pres.columns:
            continue
        serie = pres[v].dropna().astype(float)
        if serie.empty:
            continue
        out[v] = {
            "label":  LABELS.get(v, v),
            "unit":   UNITS.get(v, ""),
            "median": float(np.median(serie)),
            "p10":    float(np.percentile(serie, 10)),
            "p90":    float(np.percentile(serie, 90)),
            "n":      int(serie.size),
        }
    return out


def format_envelope(env: dict, max_vars: int = 6) -> str:
    """Renderiza la envolvente como texto para el prompt (FUENTE 1)."""
    if not env:
        return "No disponible"
    filas = []
    for v, d in list(env.items())[:max_vars]:
        u = f" {d['unit']}" if d["unit"] else ""
        filas.append(f"    - {d['label']}: mediana {d['median']:.0f}{u} "
                     f"(P10-P90: {d['p10']:.0f}-{d['p90']:.0f}{u})")
    n = next(iter(env.values()))["n"]
    return f"Envolvente climática en los {n} sitios de presencia:\n" + "\n".join(filas)
