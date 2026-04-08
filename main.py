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
from llm.openrouter_client import OpenRouterClient


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

    # 2. Cargar geometrías base: CR (para outputs) y Mesoamérica (para entrenamiento)
    map_loader = ExpertMapLoader()
    cr_bounds = map_loader.load_country_boundary(config.DEFAULT_COUNTRY)
    meso_bounds = map_loader.load_mesoamerica_boundary()

    if meso_bounds is None:
        print("[ADVERTENCIA] No se pudo cargar el límite Mesoamericano. Se usará solo CR.")
        meso_bounds = cr_bounds

    # 3. Cargar mapa experto recortado a CR (solo para visualización)
    expert_map_cr = map_loader.load_expert_range(especie_nombre, cr_bounds)
    if expert_map_cr is None:
        print("[ADVERTENCIA] Mapa experto (BIEN) no encontrado de forma automatica.")
        print("[ACCION REQUERIDA] Sugerencia para futuras ejecuciones:")
        print("   1. Descargue el rango experto manualmente desde IUCN (iucnredlist.org) o Map of Life (mol.org).")
        print(f"   2. Guarde el archivo Shapefile (.shp y dependencias) con el nombre: '{especie_nombre.replace(' ', '_')}.shp'")
        print(f"   3. Ubiqulo en el directorio: {map_loader.expert_dir}")
        print("[INFO] El pipeline continuara su ejecucion modelando unicamente sobre gradientes climaticos.")

    # 4. Ingesta de presencias GBIF — Mesoamérica para entrenamiento, CR para outputs
    extractor = GBIFExtractor()
    presencias_meso = extractor.fetch_occurrences_mesoamerica(especie_nombre)
    presencias_meso = extractor.clean_spatial_outliers(presencias_meso, meso_bounds)

    # Subconjunto CR-only: para visualización, altitud y contexto de conservación
    presencias_cr = extractor.clean_spatial_outliers(presencias_meso, cr_bounds) if presencias_meso is not None else None

    if presencias_meso is None or presencias_meso.empty:
        print(f"[ERROR FATAL] No se obtuvieron registros de presencia para {especie_nombre} en Mesoamérica.")
        return False

    if presencias_cr is None or presencias_cr.empty:
        print(f"[ADVERTENCIA] No hay registros de {especie_nombre} dentro de Costa Rica.")
        print("[INFO] Se continuará con el modelo Mesoamericano, pero los outputs serán limitados.")
        presencias_cr = presencias_meso  # fallback: usar todo Mesoamérica para outputs

    print(f"[INFO] Presencias Mesoamérica: {len(presencias_meso)} | Solo CR: {len(presencias_cr)}")

    # Mapa multicapa con presencias y límite CR
    vis = Visualizer()
    vis.plot_spatial_overlap(
        species_name=especie_nombre,
        country_boundary=cr_bounds,
        expert_map=expert_map_cr,
        presencias_gdf=presencias_cr,
        output_dir=out_dir
    )

    # Mapa de distribución Mesoamericana (todos los puntos de entrenamiento)
    # Necesitamos el mapa experto sin recortar a CR para este mapa
    expert_map_meso = map_loader.load_expert_range(especie_nombre, meso_bounds) if expert_map_cr is not None else None
    vis.plot_mesoamerica_overview(
        species_name=especie_nombre,
        meso_boundary=meso_bounds,
        expert_map=expert_map_meso,
        presencias_meso=presencias_meso,
        presencias_cr=presencias_cr,
        output_dir=out_dir
    )

    # Enriquecer presencias CR con topografía (raster solo cubre CR)
    ruta_altitud = os.path.join("data_raw", "topography", "altitud_cr.tif")
    presencias_cr = extraer_altitud(presencias_cr, ruta_altitud)

    # 5. Rasters climáticos recortados a Mesoamérica (para entrenamiento del RF)
    climate_loader = ClimateLoader()
    raster_paths = climate_loader.get_climate_layers(meso_bounds, region_name='meso')
    if not raster_paths:
        print("[ERROR FATAL] Fallo la carga de matrices climaticas.")
        return False

    # Ecoregiones CR (solo útil para puntos dentro de CR; no se pasa al entrenamiento Meso)
    # Se omite para evitar sesgo geográfico: puntos no-CR quedarían todos como 'Zona_Desconocida'
    ecoregions_gdf = None

    # 5.1 Filtro de variables expertas (precipitación, que es el factor limitante en Mesoamérica)
    variables_expertas = ["bio_14", "bio_15", "bio_16", "bio_17", "bio_18", "bio_19"]
    raster_paths = {k: v for k, v in raster_paths.items() if any(var in k or var in v for var in variables_expertas)}

    print(f"[INFO] Filtro experto aplicado. Se entrenará el modelo con {len(raster_paths)} variables climáticas.")

    # 6. Geoprocesamiento y Matriz Ambiental — usando presencias y límites Mesoamericanos
    geo = Geoprocessor()
    matriz_final = geo.build_environmental_matrix(
        presencias_meso, meso_bounds, raster_paths,
        ecoregions_gdf=None,
        use_extent_background=True,  # Pseudo-ausencias distribuidas en toda Mesoamérica
        num_pseudoausencias=1500
    )

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
    
    # Contexto de conservación usando presencias CR-only (los vectores son capas de CR)
    contexto_conservacion = generar_contexto_conservacion(presencias_cr, rutas_vectores)

    # 10. IA Generativa (Comparativa de LLMs) Solo Texto y modelos matematicos 
    #Deprecated por modelos multimodales.
    """    llm_analyst = GroqAnalyst()
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
    """
    # ==========================================================
    # 10. IA Generativa (Agente Híbrido Multimodal - DEMO)
    # ==========================================================
    print("\n" + "="*40)
    print("[FASE] Iniciando Modelado Híbrido (Visión + Matemáticas)")
    
    ruta_del_mapa = os.path.join(out_dir, "mapa_solapamiento_espacial.png")

    # --- EXTRACCIÓN DE ALTITUD desde presencias CR (raster CR-only) ---
    info_altitud = "No disponible"
    try:
        col_altitud = [c for c in presencias_cr.columns if 'alt' in c.lower()]
        if col_altitud:
            nombre_col = col_altitud[0]
            alt_min = presencias_cr[nombre_col].min()
            alt_max = presencias_cr[nombre_col].max()
            alt_med = presencias_cr[nombre_col].mean()
            info_altitud = f"{alt_min:.0f} - {alt_max:.0f} msnm (Promedio: {alt_med:.0f} m)"
            print(f"[INFO] Altitud detectada (CR): {info_altitud}")
    except Exception as e:
        print(f"[WARN] Error al procesar estadística de altitud: {e}")
    # -------------------------------------

    # Lista de los mejores modelos Multimodales (VLM) en OpenRouter
    modelos_multimodales = [
        "openai/gpt-4o",                # El líder general
        #"google/gemini-1.5-pro",        # Excelente comprensión de contexto largo e imágenes
        "anthropic/claude-3.5-sonnet"  # El mejor razonamiento lógico actual
        # "qwen/qwen2-vl-72b-instruct"  # (Opcional) Si quieres mantener a Qwen en la comparativa
    ]

    try:
        # Instanciamos el cliente una sola vez
        or_client = OpenRouterClient(api_key=config.OPENROUTER_API_KEY)
        
        # Iteramos sobre cada modelo
        for modelo_vlm in modelos_multimodales:
            or_client.generate_profile(
                species_name=especie_nombre,
                rf_metrics=rf_metrics,
                shap_dict=shap_data,
                output_dir=out_dir,
                image_path=ruta_del_mapa,
                user_question=user_question,
                model_override=modelo_vlm,  # <- Aquí inyectamos el modelo en cada vuelta
                info_altitud=info_altitud
            )
            
    except Exception as e:
        print(f"[ERROR] Fallo en la inicialización del Agente Híbrido: {e}")
    # ==========================================================

    # =================================================================
    # 11. Ejecución del Baseline Dual (Comparativa Visual)
    # =================================================================
    #from llm.dual_baseline_client import DualBaselineAnalyst
    
    #print("[INFO] Iniciando ejecución del Modelo Dual (Baseline Visión + Texto)...")
    
    # Buscamos la imagen que se generó en el paso de visualización
    # (Asegúrate de que la extensión sea .png o .jpg según lo que genere tu código)
    #ruta_mapa = os.path.join(out_dir, "mapa_solapamiento_espacial.png") 
    
    #try:
    #    baseline_analyst = DualBaselineAnalyst()
    #    baseline_analyst.generate_profile(
    #        species_name=especie_nombre,
    #        image_path=ruta_mapa,
    #        output_dir=out_dir,
    #        user_question=user_question  # Le pasamos la misma pregunta de la terminal
    #    )
   # except Exception as e:
    #    print(f"[ERROR] No se pudo ejecutar el Baseline Dual: {e}")

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
            # Pasamos la especie y la pregunta a cada iteración
            procesar_especie(especie, args.question)
            
    else:
        print("[ERROR] Debes proporcionar un argumento. Usa -s para una especie o -f para una lista txt.")
        print("Ejemplo 1: python main.py -s \"Quercus costaricensis\"")
        print("Ejemplo 2: python main.py -s \"Quercus costaricensis\" -q \"¿Cómo le afecta el cambio climático?\"")