import geopandas as gpd
import os
import config

class EcoregionsLoader:
    def load_ecoregions(self, filename):
        """Carga el shapefile de zonas de vida o ecosistemas y estandariza su formato."""
        file_path = os.path.join("data_raw", "ecoregions", filename) # Ajusta la ruta a como la tengas
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[ERROR] No se encontró el shapefile en: {file_path}")
            
        print(f"[INFO] Cargando capa espacial: {filename}...")
        gdf = gpd.read_file(file_path)

        # ==========================================
        # --- ESTANDARIZADOR AUTOMÁTICO DE COLUMNAS ---
        # ==========================================
        # Si la columna 'nombre' no existe, buscamos los nombres más comunes del SINAC/FONAFIFO
        if 'nombre' not in gdf.columns:
            # Puedes agregar aquí la columna exacta si la descubres después
            posibles_columnas = ['cobertura', 'clase', 'Clase', 'descripcio', 'uso_tierra', 'tipo_bosq', 'categoria', 'ecosistema']
            
            columna_encontrada = False
            for col in posibles_columnas:
                if col in gdf.columns:
                    # Renombramos la columna del SINAC a 'nombre' para que Geoprocessor no se rompa
                    gdf = gdf.rename(columns={col: 'nombre'})
                    print(f"[INFO] Columna original '{col}' estandarizada a 'nombre'.")
                    columna_encontrada = True
                    break
            
            if not columna_encontrada:
                # Si no adivina la columna, imprime las opciones para que puedas agregarla a la lista
                print(f"[ERROR CRÍTICO] No se encontró una columna de texto en el shapefile.")
                print(f"Columnas disponibles en este archivo: {gdf.columns.tolist()}")
                raise KeyError("Debes abrir ecoregions_loader.py y agregar el nombre correcto a 'posibles_columnas'.")
        
        # Estandarizamos el CRS (Sistema de Coordenadas) al de WorldClim (EPSG:4326)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
            
        return gdf