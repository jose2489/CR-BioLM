import os
import argparse
import config
import numpy as np

from sklearn.model_selection import train_test_split
from utils.visualizer import Visualizer
from xai.lime_explainer import LIMEExplainer
from data.gbif_extractor import GBIFExtractor
from data.expert_maps import ExpertMapLoader
from data.climate_loader import ClimateLoader
from data.geoprocessor import Geoprocessor
from models.random_forest import RandomForestSDM
from models.cnn_model import CNNSDM
from xai.shap_explainer import SHAPExplainer
from xai.grad_cam import MultimodalGradCAM
from utils.geoprocesamiento import extraer_altitud, generar_contexto_conservacion
from llm.openrouter_client import OpenRouterClient
from utils.map_gen.habitat_map import generate_habitat_map


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

    # 3. Mapa experto BIEN reemplazado por mapa de hábitat del Manual de Plantas CR
    expert_map_cr = None  # ya no se usa BIEN

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

    # Mapa Mesoamericano de entrenamiento (overview de todos los puntos)
    vis = Visualizer()
    vis.plot_mesoamerica_overview(
        species_name=especie_nombre,
        meso_boundary=meso_bounds,
        expert_map=None,
        presencias_meso=presencias_meso,
        presencias_cr=presencias_cr,
        output_dir=out_dir
    )

    # Enriquecer presencias CR con topografía (raster solo cubre CR)
    ruta_altitud = os.path.join("data_raw", "topography", "altitud_cr.tif")
    presencias_cr = extraer_altitud(presencias_cr, ruta_altitud)

    # --- Mapa de hábitat principal: Manual de Plantas CR + Unidades Fitogeográficas + GBIF ---
    import pandas as pd
    _catalog_path = os.path.join("outputs", "picked_species_enhanced_clean.csv")
    ruta_mapa_manual = None
    texto_manual = ""
    try:
        _catalog = pd.read_csv(_catalog_path)
        _row = _catalog[_catalog['species'] == especie_nombre]
        if not _row.empty:
            r = _row.iloc[0]
            ruta_mapa_manual = generate_habitat_map(
                species_name=especie_nombre,
                geographic_notes=r.get('geographic_notes'),
                elevation_min=r.get('elevation_min_m'),
                elevation_max=r.get('elevation_max_m'),
                presencias_gdf=presencias_cr,
                output_path=os.path.join(out_dir, "mapa_habitat_manual.png"),
            )
            # Build summary text for the LLM prompt
            parts = []
            geo = str(r.get('geographic_notes', '') or '').strip()
            if geo and geo.lower() != 'nan':
                parts.append(f"Distribución geográfica (Manual): {geo}")
            emin, emax = r.get('elevation_min_m'), r.get('elevation_max_m')
            if emin and emax:
                parts.append(f"Rango altitudinal: {int(emin)}–{int(emax)} m s.n.m.")
            hab = str(r.get('habitat_type', '') or '').strip()
            if hab and hab.lower() != 'nan':
                parts.append(f"Tipo de hábitat: {hab}")
            texto_manual = "── Manual de Plantas de Costa Rica ──\n" + "\n".join(parts) if parts else ""

            # Guardar ficha de referencia del Manual (para evaluación posterior)
            nombre_limpio = especie_nombre.replace(" ", "_")
            ruta_ficha = os.path.join(out_dir, f"{nombre_limpio}_ficha_MdP.txt")
            with open(ruta_ficha, "w", encoding="utf-8") as f:
                f.write(f"FICHA DE REFERENCIA — Manual de Plantas de Costa Rica\n")
                f.write(f"{'='*60}\n")
                f.write(f"Especie          : {especie_nombre}\n")
                f.write(f"Familia          : {r.get('family', 'N/D')}\n")
                f.write(f"Volumen          : {r.get('volume_title', 'N/D')}\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"Distribución geográfica:\n  {r.get('geographic_notes', 'N/D')}\n\n")
                f.write(f"Rango altitudinal:\n  {r.get('elevation_min_m', '?')}–{r.get('elevation_max_m', '?')} m s.n.m.\n\n")
                f.write(f"Tipo de hábitat:\n  {r.get('habitat_type', 'N/D')}\n\n")
                f.write(f"Descripción original (habitat_raw):\n  {r.get('habitat_raw', 'N/D')}\n\n")
                f.write(f"Ocurrencias GBIF en catálogo:\n  {r.get('occurrences', 'N/D')}\n")
            print(f"[INFO] Ficha Manual guardada: {ruta_ficha}")
            print(f"[INFO] Mapa hábitat Manual generado: {ruta_mapa_manual}")
        else:
            print(f"[INFO] '{especie_nombre}' no encontrada en el catálogo del Manual — mapa de hábitat omitido.")
    except Exception as e:
        print(f"[WARN] No se pudo generar el mapa del Manual: {e}")

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

    # ==========================================================
    # 10. IA Generativa (Agente Híbrido Multimodal)
    # ==========================================================
    print("\n" + "="*40)
    print("[FASE] Iniciando Modelado Híbrido (Visión + Matemáticas)")
    
    # Imagen 1 → mapa hábitat Manual (con GBIF points) — fuente botánica
    # Imagen 2 → mapa solapamiento espacial RF (si existe) — fuente predictiva
    ruta_del_mapa = ruta_mapa_manual or os.path.join(out_dir, "mapa_solapamiento_espacial.png")
    ruta_mapa_rf  = os.path.join(out_dir, "mapa_solapamiento_espacial.png")

    # --- EXTRACCIÓN DE ALTITUD desde presencias CR (raster CR-only) ---
    info_altitud = "No disponible"
    try:
        col_altitud = [c for c in presencias_cr.columns if 'alt' in c.lower()]
        if col_altitud:
            nombre_col = col_altitud[0]
            alt_series = presencias_cr[nombre_col].dropna()
            if not alt_series.empty:
                alt_min = alt_series.min()
                alt_max = alt_series.max()
                alt_med = alt_series.mean()
                info_altitud = f"{alt_min:.0f} - {alt_max:.0f} msnm (Promedio: {alt_med:.0f} m)"
            print(f"[INFO] Altitud detectada (CR): {info_altitud}")
    except Exception as e:
        print(f"[WARN] Error al procesar estadística de altitud: {e}")
    # -------------------------------------

    # Lista de los mejores modelos Multimodales (VLM) en OpenRouter
    modelos_multimodales = [
        "openai/gpt-4o",                # El líder general
        #"google/gemini-1.5-pro",        # Excelente comprensión de contexto largo e imágenes
        "anthropic/claude-opus-4-5"    # El mejor razonamiento lógico actual
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
                image_path=ruta_del_mapa,          # imagen 1: hábitat Manual + GBIF
                user_question=user_question,
                model_override=modelo_vlm,
                info_altitud=info_altitud,
                manual_image_path=ruta_mapa_rf,    # imagen 2: mapa predictivo RF
                texto_manual=texto_manual
            )
            
    except Exception as e:
        print(f"[ERROR] Fallo en la inicialización del Agente Híbrido: {e}")
    # ==========================================================

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