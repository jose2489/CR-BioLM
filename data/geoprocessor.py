import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point
import rasterio
from rasterio.windows import Window
import glob
import os

class Geoprocessor:
    """
    Clase responsable de la manipulacion espacial: generacion de pseudo-ausencias
    y extraccion de valores ambientales de los rasters.
    """
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        # Fijar la semilla para garantizar pseudo-ausencias reproducibles
        np.random.seed(self.random_seed)

    def generate_pseudo_absences(self, presencias_gdf, country_boundary, buffer_min=0.1, buffer_max=0.5, num_puntos=500):
        """
        Genera pseudo-ausencias en un 'anillo' alrededor de las presencias,
        limitadas estrictamente a las fronteras del pais.
        """
        print("[INFO] Generando pseudo-ausencias (Background sampling)...")
        
        # Uso de union_all() para compatibilidad con GeoPandas 1.0+
        union_presencias = presencias_gdf.geometry.union_all()
        
        # Crear el anillo (zona de estudio menos zona de exclusion)
        zona_estudio = union_presencias.buffer(buffer_max)
        zona_exclusion = union_presencias.buffer(buffer_min)
        anillo = zona_estudio.difference(zona_exclusion)
        
        # Limitar el anillo al poligono del pais
        anillo_cr = anillo.intersection(country_boundary.union_all())
        
        ausencias = []
        minx, miny, maxx, maxy = anillo_cr.bounds
        
        while len(ausencias) < num_puntos:
            pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if anillo_cr.contains(pnt):
                ausencias.append(pnt)
                
        ausencias_gdf = gpd.GeoDataFrame(geometry=ausencias, crs="EPSG:4326")
        print(f"[INFO] Se generaron {len(ausencias_gdf)} pseudo-ausencias balanceadas.")
        return ausencias_gdf

    def build_environmental_matrix(self, presencias_gdf, country_bounds, raster_paths, ecoregions_gdf=None):
        """
        Construye la matriz final combinando presencias, pseudo-ausencias,
        clima (WorldClim) y datos categoricos (Zonas de Vida).
        """
        print("[INFO] Construyendo la matriz ambiental combinada...")
        
        # 1. Generar Pseudo-ausencias
        ausencias_gdf = self.generate_pseudo_absences(presencias_gdf, country_bounds)
        
        # 2. Combinar puntos (Presencias = 1, Ausencias = 0)
        presencias_gdf['clase'] = 1
        ausencias_gdf['clase'] = 0
        
        # ---> EL ESCUDO ANTI-NANS (Restaurado) <---
        columnas_base = ['clase', 'geometry']
        presencias_limpias = presencias_gdf[columnas_base]
        ausencias_limpias = ausencias_gdf[columnas_base]
        
        # Concatenar y FORZAR que siga siendo un objeto espacial
        df_concat = pd.concat([presencias_limpias, ausencias_limpias], ignore_index=True)
        todos_los_puntos = gpd.GeoDataFrame(df_concat, geometry='geometry', crs=presencias_gdf.crs)
        
        # 3. Integracion Categorica (Zonas de Vida)
        if ecoregions_gdf is not None:
            print("[INFO] Cruzando puntos espacialmente con Zonas de Vida (Spatial Join)...")
            crs_original = todos_los_puntos.crs 
            
            todos_los_puntos = gpd.sjoin(todos_los_puntos, ecoregions_gdf[['nombre', 'geometry']], how="left", predicate="intersects")
            todos_los_puntos = todos_los_puntos[~todos_los_puntos.index.duplicated(keep='first')]
            todos_los_puntos['nombre'] = todos_los_puntos['nombre'].fillna('Zona_Desconocida')
            
            print("[INFO] Aplicando One-Hot Encoding a categorias biologicas...")
            dummies = pd.get_dummies(todos_los_puntos['nombre'], prefix='Eco', dtype=int)
            
            todos_los_puntos = todos_los_puntos.drop(columns=['nombre', 'index_right'])
            
            df_unido = pd.concat([todos_los_puntos, dummies], axis=1)
            todos_los_puntos = gpd.GeoDataFrame(df_unido, geometry='geometry', crs=crs_original)
            
        # 4. Extraccion de Clima (Rasters)
        print("[INFO] Extrayendo variables climaticas para todos los puntos...")
        matriz_final = self.extract_raster_values(todos_los_puntos, raster_paths)
        
        # Limpiar filas que cayeron en el oceano (NaNs de WorldClim)
        matriz_final = matriz_final.dropna()
        
        print(f"[INFO] Matriz final construida. Dimensiones: {matriz_final.shape[0]} filas, {matriz_final.shape[1]} columnas.")
        return matriz_final


    def extract_raster_values(self, gdf, raster_paths):
        """
        Extrae los valores de los TIFs para un GeoDataFrame.
        """
        # Aseguramos que es un objeto espacial y usamos sus propiedades nativas .x y .y
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
        
        # Usamos zip para emparejar lon/lat de forma vectorizada (mucho mas seguro)
        coords = zip(gdf.geometry.x, gdf.geometry.y)
        
        datos = []
        for lon, lat in coords:
            fila = {'lon': lon, 'lat': lat}
            # Si por alguna razon la coordenada esta corrupta, saltamos
            if pd.isna(lon) or pd.isna(lat):
                for nombre in raster_paths.keys():
                    fila[nombre] = np.nan
                datos.append(fila)
                continue

            for nombre, ruta in raster_paths.items():
                try:
                    with rasterio.open(ruta) as src:
                        gen = src.sample([(lon, lat)])
                        valor = list(gen)[0][0]
                        
                        # Filtro critico de valores NoData
                        if valor < -9999:  
                            fila[nombre] = np.nan
                        else:
                            fila[nombre] = valor
                except Exception:
                    fila[nombre] = np.nan 
            datos.append(fila)
            
        # Convertimos el clima extraido y lo unimos al GeoDataFrame original
        df_clima = pd.DataFrame(datos, index=gdf.index)
        df_completo = pd.concat([gdf, df_clima], axis=1)
        
        # Limpieza para Scikit-Learn: Eliminamos la columna geometrica
        if 'geometry' in df_completo.columns:
            df_completo = df_completo.drop(columns=['geometry'])
            
        return df_completo

    def extract_image_patches(self, points_gdf, raster_paths, window_size=15):
        """
        Extrae 'parches' espaciales (matrices 2D) alrededor de cada punto de presencia/ausencia.
        Ideal para alimentar Redes Neuronales Convolucionales (CNNs).
        
        Parámetros:
        - points_gdf: GeoDataFrame con presencias (1) y pseudo-ausencias (0).
        - raster_paths: Lista de rutas a los archivos .tif (WorldClim).
        - window_size: Tamaño del lado del cuadrado en píxeles (debe ser impar, ej. 15x15).
        
        Retorna:
        - X_tensor: Numpy array de dimensiones (Num_muestras, Num_canales, Alto, Ancho)
        - y_vector: Etiquetas de presencia/ausencia.
        """
        print(f"[INFO] Extrayendo parches espaciales de {window_size}x{window_size} píxeles para Deep Learning...")
        
        # Calculamos el radio desde el centro
        offset = window_size // 2 
        
        X_patches = []
        y_labels = []
        puntos_validos = []

        # Convertir a un sistema proyectado si es necesario, o usar directamente
        # Asumimos que points_gdf ya tiene la columna 'clase_presencia' (1 o 0)
        
        # Abrimos todos los rasters a la vez para no abrir y cerrar en cada ciclo
        raster_envs = [rasterio.open(path) for path in raster_paths]
        
        try:
            for idx, row in points_gdf.iterrows():
                lon, lat = row.geometry.x, row.geometry.y
                etiqueta = row['clase_presencia']
                
                parche_multicanal = []
                punto_valido = True
                
                for src in raster_envs:
                    # Convertir lat/lon a fila/columna en la matriz del mapa
                    py, px = src.index(lon, lat)
                    
                    # Definir la ventana de recorte centrada en el punto
                    window = Window(px - offset, py - offset, window_size, window_size)
                    
                    # Leer la ventana
                    data = src.read(1, window=window)
                    
                    # Verificar que no nos salimos del mapa (bordes de Costa Rica)
                    if data.shape != (window_size, window_size):
                        punto_valido = False
                        break
                    
                    # Verificar que no haya valores nulos (NaN o NoData en el mar)
                    if np.any(data == src.nodata) or np.any(np.isnan(data)):
                        punto_valido = False
                        break
                        
                    parche_multicanal.append(data)
                
                if punto_valido:
                    # Apilamos los 19 canales para este punto
                    X_patches.append(np.stack(parche_multicanal, axis=0))
                    y_labels.append(etiqueta)
                    puntos_validos.append(row)
                    
        finally:
            # Siempre cerramos los archivos al terminar
            for src in raster_envs:
                src.close()
                
        # Convertimos a arreglos de NumPy (formato estándar para PyTorch)
        X_tensor = np.array(X_patches)  # Shape: (N, 19, 15, 15)
        y_vector = np.array(y_labels)   # Shape: (N,)
        
        print(f"[INFO] Extracción completa. Tensores generados: {X_tensor.shape}")
        return X_tensor, y_vector, pd.DataFrame(puntos_validos)    
    

    def extract_multimodal_data(self, matriz_final, raster_paths, col_prefix='nombre_', window_size=15):
        """
        Extrae datos multimodales para la Red Neuronal Híbrida.
        Rama A (Imágenes): Parches espaciales 2D del clima.
        Rama B (Tabular): Vectores 1D de cobertura forestal (SINAC).
        """
        import os # Asegurarnos de tener os para construir las rutas
        print(f"[INFO] Extrayendo datos multimodales (Imágenes {window_size}x{window_size} + Tabular)...")
        
        offset = window_size // 2 
        
        X_img_patches = []
        X_tab_vectors = []
        y_labels = []

        # 1. DETECCIÓN AUTOMÁTICA DE COLUMNAS DEL SINAC (Por exclusión)
        # Excluimos las columnas base y las de clima, lo que sobre ES el mapa del SINAC
        columnas_base = ['geometry', 'clase_presencia', 'species', 'lon', 'lat']
        columnas_clima = [f'bio_{i}' for i in range(1, 20)] + [f'bio{i}' for i in range(1, 20)]
        categorical_cols = [col for col in matriz_final.columns if col not in columnas_base and col not in columnas_clima]
        
        print(f"[INFO] Se detectaron {len(categorical_cols)} variables categóricas para la rama tabular.")
        
        # 2. AUTOCORRECCIÓN DE RUTAS DE RASTER
        # Si la lista solo tiene 'bio_1', la convertimos a 'data_raw/worldclim/bio_1.tif'
        rutas_validas = []
        for path in raster_paths:
            # Limpiamos el nombre base (ej. convertimos 'bio_1' o 'bio_1.tif' simplemente en 'bio_1')
            path_str = str(path).replace('.tif', '') 
            
            # El radar: busca cualquier archivo que termine en bio_1.tif en TODO el proyecto y subcarpetas
            busqueda = glob.glob(f"**/*{path_str}.tif", recursive=True)
            
            if busqueda:
                # Si encuentra el archivo, guarda la ruta real y absoluta que encontró
                rutas_validas.append(busqueda[0]) 
            else:
                print(f"[ERROR] El radar no pudo encontrar la imagen para: {path_str}")
                raise FileNotFoundError(f"Asegúrate de que los archivos .tif existan en alguna carpeta.")

        # Abrimos los archivos usando las rutas reales que el radar encontró
        raster_envs = [rasterio.open(path) for path in rutas_validas]
        
        # 3. DETECCIÓN DE LA COLUMNA DE PRESENCIA (1s y 0s)
        posibles_nombres = ['clase','presencia', 'clase_presencia', 'target', 'label', 'pa']
        col_etiqueta = None
        for col in posibles_nombres:
            if col in matriz_final.columns:
                col_etiqueta = col
                break
                
        if not col_etiqueta:
            print(f"Columnas en la matriz: {matriz_final.columns.tolist()}")
            raise KeyError("Por favor revisa arriba cómo se llama tu columna de presencia y agrégala a la lista.")
        
        try:
            for idx, row in matriz_final.iterrows():
                lon, lat = row['lon'], row['lat']
                etiqueta = row[col_etiqueta]
                
                # Extraer vector tabular (SINAC)
                vector_tabular = row[categorical_cols].values.astype(np.float32)
                
                parche_multicanal = []
                punto_valido = True
                
                for src in raster_envs:
                    py, px = src.index(lon, lat)
                    window = Window(px - offset, py - offset, window_size, window_size)
                    
                    # 1. boundless=True evita que el programa falle si el cuadrado toca el borde de Costa Rica
                    data = src.read(1, window=window, boundless=True, fill_value=src.nodata)
                    
                    # 2. Estandarizamos cualquier 'nodata' a np.nan para poder procesarlo
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)
                        
                    # 3. Si TODO el parche es mar o está vacío (100% NaN), lo descartamos
                    if np.all(np.isnan(data)):
                        punto_valido = False
                        break
                        
                    # 4. Si solo ALGUNOS píxeles son mar (ej. una esquina), los rellenamos con 
                    # el promedio del clima de ese mismo cuadrito (Imputación Espacial)
                    if np.any(np.isnan(data)):
                        mean_val = np.nanmean(data)
                        data = np.where(np.isnan(data), mean_val, data)
                        
                    parche_multicanal.append(data)
                
                if punto_valido:
                    X_img_patches.append(np.stack(parche_multicanal, axis=0))
                    X_tab_vectors.append(vector_tabular)
                    y_labels.append(etiqueta)
                    
        finally:
            for src in raster_envs:
                src.close()
                
        # Convertimos todo a tensores de NumPy listos para PyTorch
        X_img_tensor = np.array(X_img_patches, dtype=np.float32) # Shape: (N, 19, 15, 15)
        X_tab_tensor = np.array(X_tab_vectors, dtype=np.float32) # Shape: (N, num_categorias)
        y_tensor = np.array(y_labels, dtype=np.int64)            # Shape: (N,)
        
        print(f"[INFO] Extracción completada.")
        print(f"       -> Tensores de Imagen: {X_img_tensor.shape}")
        print(f"       -> Tensores Tabulares: {X_tab_tensor.shape}")
        
        return X_img_tensor, X_tab_tensor, y_tensor