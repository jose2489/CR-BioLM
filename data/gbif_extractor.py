import time

import pandas as pd
import geopandas as gpd
from pygbif import occurrences

class GBIFExtractor:
    """
    Clase encargada de la ingesta y limpieza espacial de presencias empiricas.
    """
    # Bounding box de Mesoamérica (incluye México sur hasta Yucatán y todo Centroamérica)
    MESOAMERICA_WKT = "POLYGON((-92.5 7.0,-77.0 7.0,-77.0 21.5,-92.5 21.5,-92.5 7.0))"

    def __init__(self, limit=800, country_code='CR'):
        self.limit = limit
        self.country_code = country_code

    def fetch_occurrences(self, species_name):
        """
        Descarga los registros de la API de GBIF para una especie especifica.
        """
        print(f"Consultando base de datos GBIF para: {species_name}...")
        
        try:
            gbif_data = occurrences.search(
                scientificName=species_name, 
                country=self.country_code, 
                hasCoordinate=True, 
                limit=self.limit
            )
            registros = gbif_data['results']
            
            presencias_coords = [
                (r['decimalLongitude'], r['decimalLatitude']) 
                for r in registros if 'decimalLongitude' in r
            ]
            
            df_presencias = pd.DataFrame(presencias_coords, columns=['lon', 'lat'])
            gdf_presencias = gpd.GeoDataFrame(
                df_presencias, 
                geometry=gpd.points_from_xy(df_presencias.lon, df_presencias.lat), 
                crs="EPSG:4326"
            )
            
            print(f"Total de presencias obtenidas de GBIF: {len(gdf_presencias)}")
            return gdf_presencias
            
        except Exception as e:
            print(f"Error al conectar con GBIF: {e}")
            return None

    def _search_con_reintentos(self, intentos=4, espera_inicial=5, **kwargs):
        """
        occurrences.search con reintentos y espera exponencial.

        La API de GBIF devuelve 503 ("Backend fetch failed") de forma intermitente.
        Sin reintentos, un solo fallo transitorio aborta toda la corrida de una
        especie — inaceptable en un barrido de cientos de especies, donde se
        perderían corridas al azar y de forma irreproducible.
        """
        for intento in range(1, intentos + 1):
            try:
                return occurrences.search(**kwargs)
            except Exception as e:
                transitorio = any(s in str(e) for s in ("503", "502", "504", "timeout",
                                                        "Timeout", "Connection"))
                if intento == intentos or not transitorio:
                    print(f"[ERROR] GBIF falló tras {intento} intento(s): {e}")
                    return None
                espera = espera_inicial * (2 ** (intento - 1))
                print(f"[WARN] GBIF transitorio ({str(e)[:60]}). "
                      f"Reintento {intento + 1}/{intentos} en {espera}s...")
                time.sleep(espera)
        return None

    def fetch_occurrences_mesoamerica(self, species_name, limit=3000):
        """
        Descarga registros de GBIF para toda Mesoamérica usando un polígono WKT.
        Proporciona una muestra representativa del nicho completo de la especie
        para entrenar el modelo más allá del ámbito territorial de Costa Rica.
        """
        print(f"[INFO] Consultando GBIF (Mesoamérica) para: {species_name} (límite: {limit} registros)...")

        try:
            gbif_data = self._search_con_reintentos(
                scientificName=species_name,
                geometry=self.MESOAMERICA_WKT,
                hasCoordinate=True,
                limit=limit
            )
            if gbif_data is None:
                return None
            registros = gbif_data['results']

            presencias_coords = [
                (r['decimalLongitude'], r['decimalLatitude'])
                for r in registros if 'decimalLongitude' in r
            ]

            df_presencias = pd.DataFrame(presencias_coords, columns=['lon', 'lat'])
            # Eliminar duplicados exactos (misma coordenada)
            df_presencias = df_presencias.drop_duplicates()

            gdf_presencias = gpd.GeoDataFrame(
                df_presencias,
                geometry=gpd.points_from_xy(df_presencias.lon, df_presencias.lat),
                crs="EPSG:4326"
            )

            print(f"[INFO] Presencias obtenidas de GBIF (Mesoamérica): {len(gdf_presencias)}")
            return gdf_presencias

        except Exception as e:
            print(f"[ERROR] Al consultar GBIF Mesoamérica: {e}")
            return None

    def clean_spatial_outliers(self, gdf_presencias, boundary_polygon):
        """
        Elimina los puntos de presencia que caen fuera de un poligono especifico 
        (ej. fronteras del pais o areas oceanicas).
        """
        print("Aplicando filtro espacial estricto (recorte por poligono)...")
        
        if gdf_presencias is None or gdf_presencias.empty:
            print("No hay datos para limpiar.")
            return None
            
        try:
            gdf_limpio = gpd.clip(gdf_presencias, boundary_polygon)
            print(f"Presencias retenidas post-filtro espacial: {len(gdf_limpio)}")
            return gdf_limpio
        except Exception as e:
            print(f"Error durante el recorte espacial: {e}")
            return None