import os
import urllib.request
import zipfile
import rasterio
from rasterio.mask import mask


class ClimateLoader:
    def __init__(self, raw_data_dir="data_raw"):
        self.raw_data_dir = raw_data_dir
        self.climate_dir = os.path.join(self.raw_data_dir, "climate_rasters")
        self.clipped_dir = os.path.join(self.climate_dir, "clipped")

        os.makedirs(self.climate_dir, exist_ok=True)
        os.makedirs(self.clipped_dir, exist_ok=True)

    def _descargar_worldclim(self):
        zip_path = os.path.join(self.climate_dir, "wc2.1_10m_bio.zip")
        url = "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_10m_bio.zip"

        if not os.path.exists(zip_path):
            print("[INFO] Descargando WorldClim...")
            try:
                urllib.request.urlretrieve(url, zip_path)
                print("[OK] Descarga completa")
            except Exception as e:
                print(f"[ERROR] Descarga falló: {e}")
                return None

        return zip_path

    def _extraer_y_recortar(self, zip_path, country_boundary):
        if country_boundary is None or country_boundary.empty:
            print("[ERROR] country_boundary inválido")
            return None

        geometria_pais = [geom for geom in country_boundary.geometry]
        rutas_recortadas = {}

        print("[INFO] Procesando variables climáticas...")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                archivos_tif = [
                    f for f in zip_ref.namelist()
                    if f.endswith('.tif') and "bio" in f
                ]

                if not archivos_tif:
                    print("[ERROR] No se encontraron TIFs en el ZIP")
                    return None

                for archivo in archivos_tif:
                    nombre_base = archivo.split('_')[-1].replace('.tif', '')
                    variable_nombre = f"bio_{nombre_base}"
                    ruta_recortada = os.path.join(
                        self.clipped_dir, f"cr_{variable_nombre}.tif"
                    )

                    rutas_recortadas[variable_nombre] = ruta_recortada

                    if os.path.exists(ruta_recortada):
                        continue

                    print(f"[INFO] Procesando {variable_nombre}")

                    # extraer
                    zip_ref.extract(archivo, self.climate_dir)
                    ruta_global = os.path.join(self.climate_dir, archivo)

                    if not os.path.exists(ruta_global):
                        print(f"[ERROR] No existe {ruta_global}")
                        continue

                    try:
                        with rasterio.open(ruta_global) as src:
                            out_image, out_transform = mask(
                                src, geometria_pais, crop=True
                            )

                            out_meta = src.meta.copy()
                            out_meta.update({
                                "driver": "GTiff",
                                "height": out_image.shape[1],
                                "width": out_image.shape[2],
                                "transform": out_transform
                            })

                            with rasterio.open(ruta_recortada, "w", **out_meta) as dest:
                                dest.write(out_image)

                    except Exception as e:
                        print(f"[ERROR] Fallo en {variable_nombre}: {e}")
                        continue

                    finally:
                        if os.path.exists(ruta_global):
                            os.remove(ruta_global)

            if not rutas_recortadas:
                print("[ERROR] No se generaron capas")
                return None

            print("[OK] Capas climáticas listas")
            return rutas_recortadas

        except Exception as e:
            print(f"[ERROR] General en rasters: {e}")
            return None

    def get_climate_layers(self, country_boundary):
        zip_path = self._descargar_worldclim()

        if not zip_path:
            print("[ERROR] ZIP no disponible")
            return None

        result = self._extraer_y_recortar(zip_path, country_boundary)

        if not result:
            print("[ERROR] Falló procesamiento climático")
            return None

        return result