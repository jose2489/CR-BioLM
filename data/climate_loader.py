import os
import shutil
import urllib.request
import zipfile

import rasterio
from rasterio.mask import mask

# WorldClim 2.1 bioclim archives. The old biogeo.ucdavis.edu host no longer
# resolves; geodata.ucdavis.edu is the current one and supports HTTP Range,
# so large downloads can resume.
_WORLDCLIM_BASE = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"

# Approximate download sizes, for the progress log and a sanity check.
_ZIP_SIZES_GB = {"10m": 0.05, "5m": 0.18, "2.5m": 0.66, "30s": 10.40}

# Costa Rica at each resolution, in usable (land) climate cells. 10m is far too
# coarse to train a national model on: 159 cells cannot support a meaningful
# background sample, and duplicate feature rows leak across a random train/test
# split. 2.5m is the practical minimum; 30s matches the DEM grid (420x380).
_CR_LAND_CELLS = {"10m": 159, "5m": 650, "2.5m": 2600, "30s": 50000}


class ClimateLoader:
    """
    Descarga, extrae y recorta las 19 variables bioclimáticas de WorldClim 2.1.

    El ZIP global se guarda en ``cache_dir`` — que por defecto NO vive dentro del
    repositorio, porque a 30 arc-seg pesa ~10.4 GB y el repo está en una carpeta
    sincronizada con Google Drive (espacio limitado y sincronización innecesaria).
    Solo los recortes por región, que pesan pocos MB, se escriben junto al proyecto.
    """

    def __init__(self, raw_data_dir="data_raw", cache_dir=None, resolution="10m"):
        self.raw_data_dir = raw_data_dir
        self.resolution = resolution
        self.climate_dir = os.path.join(self.raw_data_dir, "climate_rasters")
        self.clipped_dir = os.path.join(self.climate_dir, "clipped")

        # Los ZIP globales van fuera del repo salvo que se indique lo contrario.
        self.cache_dir = cache_dir or os.environ.get(
            "CRBIOLM_WORLDCLIM_DIR",
            os.path.join(os.path.expanduser("~"), "Documents", "Tesis",
                         "raw_data", "worldclim"),
        )

        os.makedirs(self.climate_dir, exist_ok=True)
        os.makedirs(self.clipped_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ descarga

    def _zip_path(self, resolution):
        legacy = os.path.join(self.climate_dir, f"wc2.1_{resolution}_bio.zip")
        if os.path.exists(legacy):
            return legacy                      # respeta descargas previas en el repo
        return os.path.join(self.cache_dir, f"wc2.1_{resolution}_bio.zip")

    def _descargar_worldclim(self, resolution):
        """Descarga el ZIP global si falta. Reanudable (HTTP Range) y con progreso."""
        zip_path = self._zip_path(resolution)
        url = f"{_WORLDCLIM_BASE}/wc2.1_{resolution}_bio.zip"
        esperado_gb = _ZIP_SIZES_GB.get(resolution)

        if os.path.exists(zip_path):
            try:
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    total = int(resp.headers.get("Content-Length", 0))
                if total and os.path.getsize(zip_path) == total:
                    return zip_path
                print(f"[WARN] {os.path.basename(zip_path)} incompleto "
                      f"({os.path.getsize(zip_path)/1e9:.2f} de {total/1e9:.2f} GB). Reanudando...")
            except Exception:
                return zip_path                # sin red pero hay caché: úsala

        if esperado_gb:
            print(f"[INFO] Descargando WorldClim {resolution} (~{esperado_gb:.2f} GB) → {zip_path}")
        libre_gb = shutil.disk_usage(self.cache_dir).free / 1e9
        if esperado_gb and libre_gb < esperado_gb * 1.2:
            print(f"[ERROR] Espacio insuficiente en {self.cache_dir}: "
                  f"{libre_gb:.1f} GB libres, se requieren ~{esperado_gb*1.2:.1f} GB.")
            return None

        try:
            desde = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
            req = urllib.request.Request(url)
            if desde:
                req.add_header("Range", f"bytes={desde}-")

            with urllib.request.urlopen(req, timeout=60) as resp, \
                    open(zip_path, "ab" if desde else "wb") as out:
                total = int(resp.headers.get("Content-Length", 0)) + desde
                hecho, ultimo = desde, -1
                while True:
                    bloque = resp.read(1 << 20)          # 1 MiB
                    if not bloque:
                        break
                    out.write(bloque)
                    hecho += len(bloque)
                    if total:
                        pct = int(hecho * 100 / total)
                        if pct != ultimo and pct % 5 == 0:
                            print(f"       {pct:3d}%  ({hecho/1e9:.2f}/{total/1e9:.2f} GB)")
                            ultimo = pct
            print("[INFO] Descarga completada.")
            return zip_path

        except Exception as e:
            print(f"[ERROR] Falló la descarga: {e}")
            print(f"[ACCION] Descarga manualmente {url} y colócalo en {self.cache_dir}. "
                  f"La descarga es reanudable: volver a ejecutar continúa donde quedó.")
            return None

    # ------------------------------------------------------------------- recorte

    def _extraer_y_recortar(self, zip_path, boundary, region_name="cr",
                            resolution="10m", variables=None):
        """
        Recorta cada variable al polígono de la región.

        Extrae un GeoTIFF global a la vez en ``cache_dir`` (fuera del repo), lo
        recorta y lo borra, de modo que el pico de disco es el de UN raster —
        relevante a 30 arc-seg, donde cada global pesa ~1.9 GB.
        """
        if boundary is None or boundary.empty:
            print("[ERROR] Se requiere un polígono de región para realizar el recorte.")
            return None

        geometria_region = [geom for geom in boundary.geometry]
        rutas_recortadas = {}
        pendientes = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            archivos_tif = [f for f in zf.namelist() if f.endswith(".tif") and "bio" in f]

            for archivo in archivos_tif:
                nombre_base = archivo.split("_")[-1].replace(".tif", "")
                variable_nombre = f"bio_{nombre_base}"
                if variables and variable_nombre not in variables:
                    continue
                # La resolución va en el nombre del caché. Sin ella "cr_bio_1.tif"
                # podía ser 10m o 30s indistintamente: un recorte previo a 10m hacía
                # que una corrida a 30s lo diera por hecho y entrenara en silencio
                # sobre la malla gruesa (21x19 en lugar de 409x360).
                ruta = os.path.join(
                    self.clipped_dir, f"{region_name}_{resolution}_{variable_nombre}.tif")
                rutas_recortadas[variable_nombre] = ruta
                if not os.path.exists(ruta):
                    pendientes.append((archivo, variable_nombre, ruta))

            if pendientes:
                print(f"[INFO] Recortando {len(pendientes)} variable(s) "
                      f"[{region_name} @ {resolution}]...")

            for archivo, variable_nombre, ruta in pendientes:
                print(f"       {variable_nombre} [{region_name}]...")
                zf.extract(archivo, self.cache_dir)
                ruta_global = os.path.join(self.cache_dir, archivo)
                try:
                    with rasterio.open(ruta_global) as src:
                        out_image, out_transform = mask(src, geometria_region, crop=True)
                        meta = src.meta.copy()
                        meta.update({"driver": "GTiff",
                                     "height": out_image.shape[1],
                                     "width": out_image.shape[2],
                                     "transform": out_transform})
                        with rasterio.open(ruta, "w", **meta) as dest:
                            dest.write(out_image)
                finally:
                    if os.path.exists(ruta_global):
                        os.remove(ruta_global)

        if rutas_recortadas:
            ejemplo = next(iter(rutas_recortadas.values()))
            with rasterio.open(ejemplo) as s:
                print(f"[INFO] Matriz climática [{region_name} @ {resolution}] lista: "
                      f"{len(rutas_recortadas)} variables, malla {s.width}x{s.height}.")
        return rutas_recortadas

    # ---------------------------------------------------------------------- API

    def get_climate_layers(self, boundary, region_name="cr", resolution=None,
                           variables=None):
        """
        Retorna {variable: ruta_recortada} para la región dada.

        - ``resolution``: "10m" | "5m" | "2.5m" | "30s". Por defecto la del
          constructor. Ver ``_CR_LAND_CELLS``: 10m deja solo 159 celdas útiles
          sobre Costa Rica, insuficiente para entrenar un modelo nacional.
        - ``variables``: subconjunto opcional, p. ej. {"bio_1", "bio_12"}. Si es
          None se recortan las 19.
        """
        resolution = resolution or self.resolution

        if region_name == "cr":
            celdas = _CR_LAND_CELLS.get(resolution)
            if celdas and celdas < 1000:
                print(f"[WARN] A {resolution} Costa Rica tiene ~{celdas} celdas de clima "
                      f"utilizables: insuficiente para un muestreo de fondo sin filas "
                      f"duplicadas (fuga entre train/test). Usa '2.5m' o '30s'.")

        zip_path = self._descargar_worldclim(resolution)
        if not zip_path:
            return None

        return self._extraer_y_recortar(zip_path, boundary, region_name=region_name,
                                        resolution=resolution, variables=variables)
