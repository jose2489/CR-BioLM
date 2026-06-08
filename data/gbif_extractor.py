import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import geopandas as gpd
from pygbif import occurrences, species as gbif_species
import config


class GBIFExtractor:
    """
    Clase encargada de la ingesta y limpieza espacial de presencias empíricas,
    y de la extracción de descripciones textuales desde la API de GBIF.

    Límites de descarga configurables en config.py:
        GBIF_LIMIT_CR  : registros máximos para Costa Rica
        GBIF_LIMIT_MESO: registros máximos para Mesoamérica
    """

    MESOAMERICA_WKT = "POLYGON((-92.5 7.0,-77.0 7.0,-77.0 21.5,-92.5 21.5,-92.5 7.0))"

    def __init__(self, country_code: str = "CR"):
        self.country_code = country_code

    def fetch_species_description(self, species_name: str) -> str | None:
        """
        Consulta la API de GBIF Species para obtener descripciones textuales
        de la especie (hábitat, ecología, distribución) desde fuentes como
        EOL, Catalogue of Life y otros proveedores integrados.

        Parámetros:
            species_name: nombre científico de la especie

        Retorna:
            String con las descripciones concatenadas, o None si no hay datos.
        """
        print(f"[GBIF] Buscando descripción textual para: {species_name}...")
        try:
            backbone  = gbif_species.name_backbone(name=species_name, rank="SPECIES")
            usage_key = backbone.get("usageKey")
            if not usage_key:
                print(f"[GBIF] No se encontró usageKey para: {species_name}")
                return None

            result       = gbif_species.name_usage(key=usage_key, data="descriptions")
            descripciones = result.get("results", [])
            if not descripciones:
                print(f"[GBIF] Sin descripciones disponibles para: {species_name}")
                return None

            tipos_relevantes = {"habitat", "ecology", "distribution",
                                "general", "biology", "conservation"}
            partes = [
                str(d.get("description", "") or "").strip()
                for d in descripciones
                if str(d.get("description", "") or "").strip()
                and (not d.get("type") or any(t in str(d.get("type","")).lower()
                                               for t in tipos_relevantes))
            ]
            if not partes:
                partes = [str(d.get("description","")).strip()
                          for d in descripciones if d.get("description")]

            descripcion_final = " ".join(partes[:5])
            print(f"[GBIF] Descripción obtenida ({len(partes)} fragmentos): "
                  f"{descripcion_final[:80]}...")
            return descripcion_final if descripcion_final else None

        except Exception as e:
            print(f"[GBIF] Error al obtener descripción: {e}")
            return None

    def fetch_occurrences(self, species_name: str):
        """
        Descarga registros de GBIF para Costa Rica.

        Parámetros:
            species_name: nombre científico de la especie

        Retorna:
            GeoDataFrame con coordenadas de presencia, o None si falla.
        """
        print(f"Consultando base de datos GBIF para: {species_name}...")
        try:
            gbif_data = occurrences.search(
                scientificName=species_name,
                country=self.country_code,
                hasCoordinate=True,
                limit=config.GBIF_LIMIT_CR
            )
            registros = gbif_data['results']
            presencias_coords = [
                (r['decimalLongitude'], r['decimalLatitude'])
                for r in registros if 'decimalLongitude' in r
            ]
            df = pd.DataFrame(presencias_coords, columns=['lon', 'lat'])
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.lon, df.lat),
                crs="EPSG:4326"
            )
            print(f"Total de presencias obtenidas de GBIF: {len(gdf)}")
            return gdf
        except Exception as e:
            print(f"Error al conectar con GBIF: {e}")
            return None

    def fetch_occurrences_mesoamerica(self, species_name: str):
        """
        Descarga registros de GBIF para toda Mesoamérica usando un polígono WKT.
        Proporciona una muestra representativa del nicho completo de la especie
        para entrenar el modelo más allá del ámbito territorial de Costa Rica.

        Parámetros:
            species_name: nombre científico de la especie

        Retorna:
            GeoDataFrame con coordenadas de presencia, o None si falla.
        """
        print(f"[INFO] Consultando GBIF (Mesoamérica) para: {species_name} "
              f"(límite: {config.GBIF_LIMIT_MESO} registros)...")
        try:
            gbif_data = occurrences.search(
                scientificName=species_name,
                geometry=self.MESOAMERICA_WKT,
                hasCoordinate=True,
                limit=config.GBIF_LIMIT_MESO
            )
            registros = gbif_data['results']
            presencias_coords = [
                (r['decimalLongitude'], r['decimalLatitude'])
                for r in registros if 'decimalLongitude' in r
            ]
            df = pd.DataFrame(presencias_coords, columns=['lon', 'lat'])
            df = df.drop_duplicates()
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.lon, df.lat),
                crs="EPSG:4326"
            )
            print(f"[INFO] Presencias obtenidas de GBIF (Mesoamérica): {len(gdf)}")
            return gdf
        except Exception as e:
            print(f"[ERROR] Al consultar GBIF Mesoamérica: {e}")
            return None

    def clean_spatial_outliers(self, gdf_presencias, boundary_polygon):
        """
        Elimina los puntos de presencia que caen fuera de un polígono específico.

        Parámetros:
            gdf_presencias   : GeoDataFrame con puntos de presencia
            boundary_polygon : GeoDataFrame con el polígono de recorte

        Retorna:
            GeoDataFrame recortado, o None si la entrada está vacía.
        """
        print("Aplicando filtro espacial estricto (recorte por polígono)...")
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