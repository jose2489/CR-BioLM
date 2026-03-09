import os
import subprocess
import geopandas as gpd

class ExpertMapLoader:
    """
    Clase encargada de cargar geometrias base y mapas de rangos expertos (BIEN).
    Incluye un puente (bridge) hacia R para descargar los datos si no existen localmente.
    """
    def __init__(self, raw_data_dir="data_raw"):
        self.raw_data_dir = raw_data_dir
        self.expert_dir = os.path.join(self.raw_data_dir, "expert_maps")
        os.makedirs(self.expert_dir, exist_ok=True)
        
        # Ruta absoluta al script de R
        self.r_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_bien.R")

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

    def _descargar_via_r(self, especie_formateada):
        """
        Ejecuta el script de R en un subproceso para descargar el shapefile.
        """
        print("[INFO] El mapa no existe localmente. Invocando script de R para descarga automatica...")
        
        # Comando a ejecutar: Rscript download_bien.R Quercus_costaricensis ruta/de/salida
        comando = ["Rscript", self.r_script_path, especie_formateada, self.expert_dir]
        
        try:
            # Ejecutamos R y capturamos su salida para imprimirla en la consola de Python
            resultado = subprocess.run(comando, capture_output=True, text=True)
            
            # Imprimimos lo que R nos dijo (cat)
            if resultado.stdout:
                print(resultado.stdout.strip())
                
            # Si el script de R falla (ej. especie no existe), return False
            if resultado.returncode != 0:
                if resultado.stderr:
                    print(f"[ERROR R]: {resultado.stderr.strip()}")
                return False
                
            return True
            
        except FileNotFoundError:
            print("[ERROR] No se encontro 'Rscript' en el sistema. Asegurate de que R este instalado y agregado a las variables de entorno (PATH) de Windows.")
            return False

    def load_expert_range(self, species_name, country_boundary=None):
        """
        Carga el shapefile de la especie. Si no existe, invoca a R para descargarlo.
        """
        file_name = f"{species_name.replace(' ', '_')}.shp"
        file_path = os.path.join(self.expert_dir, file_name)
        
        # Logica de Cache Inteligente
        if not os.path.exists(file_path):
            exito_descarga = self._descargar_via_r(species_name.replace(' ', '_'))
            if not exito_descarga:
                print("[INFO] Se omitira la capa experta para esta especie.")
                return None
                
        try:
            print("[INFO] Cargando mapa experto espacial...")
            expert_map = gpd.read_file(file_path).to_crs("EPSG:4326")
            
            if country_boundary is not None and not country_boundary.empty:
                expert_map = gpd.clip(expert_map, country_boundary)
                
            return expert_map
            
        except Exception as e:
            print(f"[ERROR] Al procesar el archivo local {file_name}: {e}")
            return None