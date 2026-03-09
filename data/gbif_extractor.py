import pandas as pd
import geopandas as gpd
from pygbif import occurrences

class GBIFExtractor:
    """
    Clase encargada de la ingesta y limpieza espacial de presencias empiricas.
    """
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