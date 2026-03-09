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

    def _extraer_y_recortar(self, zip_path, country_boundary):
        """
        Extrae las 19 variables del ZIP y las recorta inmediatamente usando el poligono del pais.
        """
        if country_boundary is None or country_boundary.empty:
            print("[ERROR] Se requiere el poligono del pais para realizar el recorte.")
            return None

        # Extraer geometria para rasterio
        geometria_pais = [geom for geom in country_boundary.geometry]
        rutas_recortadas = {}

        print("[INFO] Verificando y procesando las 19 variables bioclimaticas...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Filtrar solo los archivos .tif que corresponden a las 19 variables
                archivos_tif = [f for f in zip_ref.namelist() if f.endswith('.tif') and 'bio' in f]
                
                for archivo in archivos_tif:
                    # Ejemplo: extraer el numero de wc2.1_10m_bio_1.tif -> bio_1
                    nombre_base = archivo.split('_')[-1].replace('.tif', '')
                    variable_nombre = f"bio_{nombre_base}"
                    ruta_recortada = os.path.join(self.clipped_dir, f"cr_{variable_nombre}.tif")
                    rutas_recortadas[variable_nombre] = ruta_recortada
                    
                    # Logica de Cache: Si ya esta recortado, saltar procesamiento
                    if os.path.exists(ruta_recortada):
                        continue
                        
                    # Extraer el TIF global temporalmente
                    zip_ref.extract(archivo, self.climate_dir)
                    ruta_global = os.path.join(self.climate_dir, archivo)
                    
                    # Recortar
                    print(f"       Recortando {variable_nombre}...")
                    with rasterio.open(ruta_global) as src:
                        out_image, out_transform = mask(src, geometria_pais, crop=True)
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform
                        })
                        with rasterio.open(ruta_recortada, "w", **out_meta) as dest:
                            dest.write(out_image)
                            
                    # Eliminar el archivo global extraido para ahorrar espacio
                    os.remove(ruta_global)
                    
            print("[INFO] Matriz climatica local (19 variables) lista en cache.")
            return rutas_recortadas
            
        except Exception as e:
            print(f"[ERROR] Fallo al procesar los rasters: {e}")
            return None

    def get_climate_layers(self, country_boundary):
        """
        Metodo principal que orquesta la validacion, descarga y recorte.
        Retorna un diccionario con las rutas de las 19 variables recortadas.
        """
        zip_path = self._descargar_worldclim()
        if not zip_path:
            return None
            
        return self._extraer_y_recortar(zip_path, country_boundary)