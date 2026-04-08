import os
import urllib.request
import zipfile
import rasterio
from rasterio.mask import mask

class ClimateLoader:
    """
    Clase para gestionar la descarga, extraccion y recorte espacial de las 
    19 variables bioclimaticas de WorldClim.
    """
    def __init__(self, raw_data_dir="data_raw"):
        self.raw_data_dir = raw_data_dir
        self.climate_dir = os.path.join(self.raw_data_dir, "climate_rasters")
        self.clipped_dir = os.path.join(self.climate_dir, "clipped")
        
        os.makedirs(self.climate_dir, exist_ok=True)
        os.makedirs(self.clipped_dir, exist_ok=True)

    def _descargar_worldclim(self):
        """
        Verifica la existencia del ZIP global de WorldClim. Si no existe, intenta descargarlo.
        """
        zip_path = os.path.join(self.climate_dir, "wc2.1_10m_bio.zip")
        url = "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_10m_bio.zip"
        
        if not os.path.exists(zip_path):
            print(f"[INFO] El archivo global no existe en cache. Iniciando descarga (aprox. 130MB)...")
            try:
                urllib.request.urlretrieve(url, zip_path)
                print("[INFO] Descarga completada exitosamente.")
            except Exception as e:
                print(f"[ERROR] Fallo la descarga automatica: {e}")
                print(f"[ACCION] Por favor descarga el archivo manualmente de {url} y colocalo en {self.climate_dir} con el nombre 'wc2.1_10m_bio.zip'.")
                return None
        return zip_path

    def _extraer_y_recortar(self, zip_path, boundary, region_name='cr'):
        """
        Extrae las 19 variables del ZIP y las recorta usando el poligono de la región.
        El parámetro region_name controla el prefijo del cache (ej. 'cr' o 'meso').
        """
        if boundary is None or boundary.empty:
            print("[ERROR] Se requiere un polígono de región para realizar el recorte.")
            return None

        geometria_region = [geom for geom in boundary.geometry]
        rutas_recortadas = {}

        print(f"[INFO] Verificando y procesando las 19 variables bioclimáticas (región: {region_name})...")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                archivos_tif = [f for f in zip_ref.namelist() if f.endswith('.tif') and 'bio' in f]

                for archivo in archivos_tif:
                    nombre_base = archivo.split('_')[-1].replace('.tif', '')
                    variable_nombre = f"bio_{nombre_base}"
                    ruta_recortada = os.path.join(self.clipped_dir, f"{region_name}_{variable_nombre}.tif")
                    rutas_recortadas[variable_nombre] = ruta_recortada

                    if os.path.exists(ruta_recortada):
                        continue

                    zip_ref.extract(archivo, self.climate_dir)
                    ruta_global = os.path.join(self.climate_dir, archivo)

                    print(f"       Recortando {variable_nombre} [{region_name}]...")
                    with rasterio.open(ruta_global) as src:
                        out_image, out_transform = mask(src, geometria_region, crop=True)
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform
                        })
                        with rasterio.open(ruta_recortada, "w", **out_meta) as dest:
                            dest.write(out_image)

                    os.remove(ruta_global)

            print(f"[INFO] Matriz climática [{region_name}] (19 variables) lista en caché.")
            return rutas_recortadas

        except Exception as e:
            print(f"[ERROR] Fallo al procesar los rasters: {e}")
            return None

    def get_climate_layers(self, boundary, region_name='cr'):
        """
        Método principal que orquesta la validación, descarga y recorte.
        Retorna un diccionario con las rutas de las 19 variables recortadas.

        Parámetros:
        - boundary: GeoDataFrame del polígono de recorte (CR o Mesoamérica).
        - region_name: Prefijo para el nombre del caché ('cr' o 'meso').
        """
        zip_path = self._descargar_worldclim()
        if not zip_path:
            return None

        return self._extraer_y_recortar(zip_path, boundary, region_name=region_name)