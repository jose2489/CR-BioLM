import os
import geopandas as gpd

class ExpertMapLoader:
    """Carga geometrias base de paises para el pipeline CR-BioLM."""
    def __init__(self, raw_data_dir="data_raw"):
        self.raw_data_dir = raw_data_dir

    def load_country_boundary(self, country_name="Costa Rica"):
        """Carga el poligono base del pais desde un repositorio GeoJSON publico."""
        print(f"[INFO] Cargando limites territoriales de {country_name}...")
        url_mapa = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"

        try:
            world = gpd.read_file(url_mapa)
            country = world[world.name == country_name]
            if country.empty:
                return None
            return country
        except Exception as e:
            print(f"[ERROR] Al cargar mapa base: {e}")
            return None

    def load_mesoamerica_boundary(self):
        """
        Carga y une los polígonos de todos los países de Mesoamérica.
        Usado para ampliar el alcance del entrenamiento del modelo más allá de CR.
        """
        MESO_COUNTRIES = [
            'Costa Rica', 'Panama', 'Nicaragua', 'Honduras',
            'Guatemala', 'Belize', 'El Salvador', 'Mexico'
        ]
        print(f"[INFO] Cargando límites territoriales de Mesoamérica ({len(MESO_COUNTRIES)} países)...")
        url_mapa = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"

        try:
            world = gpd.read_file(url_mapa)
            meso = world[world.name.isin(MESO_COUNTRIES)].copy()
            if meso.empty:
                print("[ERROR] No se encontraron países de Mesoamérica en el GeoJSON.")
                return None
            print(f"[INFO] Límites de Mesoamérica cargados ({len(meso)} países encontrados).")
            return meso
        except Exception as e:
            print(f"[ERROR] Al cargar mapa de Mesoamérica: {e}")
            return None

