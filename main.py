import os
import argparse
import config
import torch # Aseguramos tener torch a mano
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from utils.visualizer import Visualizer
from xai.lime_explainer import LIMEExplainer
from data.ecoregions_loader import EcoregionsLoader
from data.gbif_extractor import GBIFExtractor
from data.expert_maps import ExpertMapLoader
from data.climate_loader import ClimateLoader
from data.geoprocessor import Geoprocessor
from models.random_forest import RandomForestSDM
from xai.shap_explainer import SHAPExplainer
from llm.groq_client import GroqAnalyst
from models.cnn_model import CNNSDM
from sklearn.model_selection import train_test_split
from xai.grad_cam import MultimodalGradCAM
from utils.geoprocesamiento import extraer_altitud, generar_contexto_conservacion


def procesar_especie(especie_nombre, user_question=None):
    """
    Ejecuta el pipeline completo de CR-BioLM para una especie especifica.
    Retorna True si fue exitoso, False si fue omitida por falta de datos.
    """
    print("=" * 60)
    print(f"[INICIO] Ejecutando pipeline para: {especie_nombre.upper()}")
    print("=" * 60)
    
    # 1. Preparar directorio de salida
    out_dir = config.crear_directorio_ejecucion(especie_nombre)
    print(f"[INFO] Resultados se guardaran en: {out_dir}")

    # 2. Cargar geometrias base
    map_loader = ExpertMapLoader()
    country_bounds = map_loader.load_country_boundary(config.DEFAULT_COUNTRY)

    # 3. Cargar mapa experto (Validacion de IUCN/MOL)
    expert_map = map_loader.load_expert_range(especie_nombre, country_bounds)
    if expert_map is None:
        print("[ADVERTENCIA] Mapa experto (BIEN) no encontrado de forma automatica.")
        print("[ACCION REQUERIDA] Sugerencia para futuras ejecuciones:")
        print("   1. Descargue el rango experto manualmente desde IUCN (iucnredlist.org) o Map of Life (mol.org).")
        print(f"   2. Guarde el archivo Shapefile (.shp y dependencias) con el nombre: '{especie_nombre.replace(' ', '_')}.shp'")
        print(f"   3. Ubiqulo en el directorio: {map_loader.expert_dir}")
        print("[INFO] El pipeline continuara su ejecucion modelando unicamente sobre gradientes climaticos.")

    # 4. Ingesta de presencias GBIF (Validacion de datos vacios)
    extractor = GBIFExtractor()
    presencias = extractor.fetch_occurrences(especie_nombre)
    presencias_limpias = extractor.clean_spatial_outliers(presencias, country_bounds)

    if presencias_limpias is None or presencias_limpias.empty:
        print(f"[ERROR FATAL] No se obtuvieron registros de presencia limpios para {especie_nombre}.")
        print("[INFO] Terminando ejecucion para esta especie. Pasando a la siguiente...")
        return False
    # Generar y guardar el mapa multicapa
    vis = Visualizer()
    vis.plot_spatial_overlap(
        species_name=especie_nombre, 
        country_boundary=country_bounds, 
        expert_map=expert_map, 
        presencias_gdf=presencias_limpias, 
        output_dir=out_dir
    )
    #  Enriquecer dataframe con topografía
    ruta_altitud = os.path.join("data_raw", "topography", "altitud_cr.tif")
    presencias_limpias = extraer_altitud(presencias_limpias, ruta_altitud)

    # 5. Descarga y recorte de matrices climaticas (WorldClim)
    climate_loader = ClimateLoader()
    raster_paths = climate_loader.get_climate_layers(country_bounds)
    if not raster_paths:
        print("[ERROR FATAL] Fallo la carga de matrices climaticas.")
        return False
    eco_loader = EcoregionsLoader()
    ecoregions_gdf = eco_loader.load_ecoregions("cobertura_forestal_2023.shp")

    # 6. Geoprocesamiento y Matriz Ambiental
    geo = Geoprocessor()
    matriz_final = geo.build_environmental_matrix(presencias_limpias, country_bounds, raster_paths, ecoregions_gdf)

    # 7. Particion de datos para Machine Learning
    X = matriz_final.drop(columns=['clase', 'lon', 'lat'])
    y = matriz_final['clase']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED, stratify=y
    )

# ==========================================
    # 8. MODELADO PREDICTIVO ( RF vs CNN)
    # ==========================================
    EJECUTAR_CNN = False
    # --- 8.1 CAMINO A: Random Forest (Machine Learning Tradicional) ---
    print("\n" + "="*40)
    print("[FASE] Iniciando Modelado: Random Forest")
    rf_model = RandomForestSDM()
    rf_model.train(X_train, y_train) # Asume que tus datos 1D ya están divididos
    rf_metrics = rf_model.evaluate(X_test, y_test)
    
    if 'confusion_matrix' in rf_metrics:
        vis.plot_confusion_matrix(rf_metrics['confusion_matrix'], out_dir) # Opcional: renombrar a conf_matrix_rf.png en visualizer
        
    # --- 8.2 CAMINO B: Red Neuronal Multimodal (Deep Learning) ---
    if EJECUTAR_CNN:
        print("\n" + "="*40)
        print("[FASE] Iniciando Modelado: Red Neuronal Convolucional (CNN)")
    
        # 8.2.1. Extraer datos multimodales (Imágenes 15x15 + Vectores SINAC)
        # Asegúrate de pasar la matriz_final original sin dividir y la lista de raster_paths
        X_img, X_tab, y_tensor = geo.extract_multimodal_data(
            matriz_final, 
            raster_paths, 
            col_prefix='nombre_', # Este es el prefijo que tu One-Hot Encoding le puso a los ecosistemas
            window_size=15
        )
        
        # 8.2.2. Dividir los tensores en Entrenamiento (80%) y Prueba (20%)
        # Nota: train_test_split puede dividir múltiples arreglos en paralelo perfectamente
        X_img_train, X_img_test, X_tab_train, X_tab_test, y_cnn_train, y_cnn_test = train_test_split(
            X_img, X_tab, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
        )
        
        # 8.3. Inicializar, entrenar y evaluar la CNN
        cnn_model = CNNSDM(epochs=30, batch_size=32, learning_rate=0.001)
        cnn_model.train(X_img_train, X_tab_train, y_cnn_train)
        cnn_metrics = cnn_model.evaluate(X_img_test, X_tab_test, y_cnn_test)
        print("="*40 + "\n")

    else:
            print("[INFO] Fase de Red Neuronal (CNN) omitida por el usuario.")
            # Creamos métricas vacías para que el LLM no se estrelle al pedirlas
            cnn_metrics = {'roc_auc': 0.0, 'accuracy': 0.0}

    # 9. Explicabilidad Espacial
    # 9.1 SHAP (Global) - La brújula para el LLM
    shap_exp = SHAPExplainer()
    shap_data = shap_exp.explain_and_plot(rf_model, X_test, out_dir)

    # 9.2 LIME (Local) - El microscopio 
    lime_exp = LIMEExplainer()
    lime_exp.explain_and_plot(rf_model, X_train, X_test, out_dir)

    # ==========================================
     # 9.5 EXPLICABILIDAD DE DEEP LEARNING (Grad-CAM)
     # ==========================================
    if EJECUTAR_CNN:
        print("\n[INFO] Calculando mapas de calor espaciales (Grad-CAM) para la CNN...")

        # Buscar el primer punto del set de prueba donde realmente haya una planta (y=1)
        indices_presencia = np.where(y_cnn_test == 1)[0]

        if len(indices_presencia) > 0:
            idx_muestra = indices_presencia[0]

            # Extraer esa muestra específica y prepararla para PyTorch
            img_sample = torch.tensor(X_img_test[idx_muestra:idx_muestra+1], dtype=torch.float32).to(cnn_model.device)
            tab_sample = torch.tensor(X_tab_test[idx_muestra:idx_muestra+1], dtype=torch.float32).to(cnn_model.device)

            # Instanciar y generar
            cam_explainer = MultimodalGradCAM(cnn_model.model)
            heatmap = cam_explainer.generate_heatmap(img_sample, tab_sample, target_class=1)

            # Extraer el canal 0 (bio_1: Temperatura Anual) de la muestra para usarlo como fondo visual
            fondo_temperatura = X_img_test[idx_muestra, 0, :, :]

            # Dibujar y guardar
            cam_explainer.plot_cam(fondo_temperatura, heatmap, out_dir)
        else:
            print("[WARN] No hay presencias en el set de prueba para generar Grad-CAM.")
    
    else:
            print("[INFO] Fase de GRAD-CAM (CNN) omitida por el usuario.")
            # Creamos métricas vacías para que el LLM no se estrelle al pedirlas
            cnn_metrics = {'roc_auc': 0.0, 'accuracy': 0.0}

    # 9.6 Generar el contexto espacial
    rutas_vectores = {
        "Áreas Silvestres Protegidas": os.path.join("data_raw", "vectors", "areas_protegidas_cr.shp"),
        "Humedales": os.path.join("data_raw", "vectors", "humedales_cr.shp"),
        "Corredores Biológicos": os.path.join("data_raw", "vectors", "corredores_biologicos_cr.shp"),
        "Áreas de Conservación (Macro-regiones)": os.path.join("data_raw", "vectors", "areas_conservacion_cr.shp")
    }
    
    # Pasamos el diccionario completo a la función
    contexto_conservacion = generar_contexto_conservacion(presencias_limpias, rutas_vectores)

    # 10. IA Generativa (Comparativa de LLMs)
    llm_analyst = GroqAnalyst()
    modelos_a_evaluar = [
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
        "moonshotai/kimi-k2-instruct-0905"     
    ]
    print("[INFO] Iniciando bloque de generacion de perfiles (IA Generativa)...")

    area_km2 = None
    for modelo in modelos_a_evaluar:
        llm_analyst.generate_profile(
            species_name=especie_nombre,
            rf_metrics=rf_metrics,
            shap_dict=shap_data,
            output_dir=out_dir,
            area_km2=area_km2,
            user_question=user_question, # Pasamos la pregunta de la terminal
            model_override=modelo,
            contexto_conservacion=contexto_conservacion
        )


    # =================================================================
    # 11. Ejecución del Baseline Dual (Comparativa Visual)
    # =================================================================
    from llm.dual_baseline_client import DualBaselineAnalyst
    
    print("[INFO] Iniciando ejecución del Modelo Dual (Baseline Visión + Texto)...")
    
    # Buscamos la imagen que se generó en el paso de visualización
    # (Asegúrate de que la extensión sea .png o .jpg según lo que genere tu código)
    ruta_mapa = os.path.join(out_dir, "mapa_solapamiento_espacial.png") 
    
    try:
        baseline_analyst = DualBaselineAnalyst()
        baseline_analyst.generate_profile(
            species_name=especie_nombre,
            image_path=ruta_mapa,
            output_dir=out_dir,
            user_question=user_question  # Le pasamos la misma pregunta de la terminal
        )
    except Exception as e:
        print(f"[ERROR] No se pudo ejecutar el Baseline Dual: {e}")

    # =================================================================
    print(f"\n[EXITO] Pipeline completado para {especie_nombre}.")
    return True


def leer_lista_especies(ruta_archivo):
    """Lee un archivo TXT y retorna una lista de nombres limpios."""
    especies = []
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as file:
            for linea in file:
                nombre = linea.strip()
                if nombre and not nombre.startswith("#"):
                    especies.append(nombre)
        return especies
    except Exception as e:
        print(f"[ERROR] No se pudo leer el archivo de especies: {e}")
        return []

if __name__ == "__main__":
    # Configuracion de argumentos de terminal
    parser = argparse.ArgumentParser(description="Ejecutor del Pipeline CR-BioLM.")
    parser.add_argument("-s", "--species", type=str, help="Ejecutar para una sola especie. Ej: 'Quercus costaricensis'")
    parser.add_argument("-f", "--file", type=str, help="Ruta a un archivo .txt con una lista de especies (una por linea).")
    
    # NUEVO: Argumento opcional para la pregunta del usuario
    parser.add_argument("-q", "--question", type=str, help="Pregunta especifica para que el LLM responda basandose en el modelo.", default=None)
    
    args = parser.parse_args()
    
    if args.species:
        # Pasamos la especie y la pregunta
        procesar_especie(args.species, args.question)
        
    elif args.file:
        lista_especies = leer_lista_especies(args.file)
        print(f"[INFO] Modo Batch iniciado. Se detectaron {len(lista_especies)} especies en el archivo.")
        
        if args.question:
            print(f"[INFO] Pregunta global detectada para el lote: '{args.question}'")
        
        for i, especie in enumerate(lista_especies, 1):
            print(f"\n[LOTE {i}/{len(lista_especies)}]")
            procesar_especie(especie, args.question)
            
    else:
        print("[ERROR] Debes proporcionar un argumento. Usa -s para una especie o -f para una lista txt.")
        print("Ejemplo 1: python main.py -s \"Quercus costaricensis\"")
        print("Ejemplo 2: python main.py -s \"Quercus costaricensis\" -q \"¿Cómo le afecta el cambio climático?\"")