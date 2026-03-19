import os
import sys
import glob
import random
import subprocess
import pandas as pd
import re

def main():
    print("="*60)
    print("[INFO] Iniciando Orquestador de Experimentos en Lote (Batch DoE)")
    print("="*60)
    
    # Aseguramos la ruta absoluta de la raíz del proyecto
    DIR_RAIZ = os.path.abspath(os.path.dirname(__file__))
    
    # 1. Cargar datos usando rutas absolutas
    ruta_gbif = os.path.join(DIR_RAIZ, "utils", "catalogo_ocurrencias_gbif.csv")
    ruta_bien = os.path.join(DIR_RAIZ, "utils", "catalogo_mapas_bien.csv")
    
    try:
        df_bien = pd.read_csv(ruta_bien)
        
        gbif_data = []
        with open(ruta_gbif, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('---') or line.startswith('Top:') or 'Species' in line:
                    continue
                partes = re.split(r'\s{2,}|\t', line)
                if len(partes) >= 2:
                    nombre_especie = partes[0].strip()
                    conteo_puntos = partes[-1].strip()
                    gbif_data.append([nombre_especie, conteo_puntos])
                    
        df_gbif = pd.DataFrame(gbif_data, columns=['especie', 'ocurrencias'])

    except Exception as e:
        print(f"[ERROR] No se pudieron leer los catálogos: {e}")
        return

    col_esp_gbif = df_gbif.columns[0]
    col_num_gbif = df_gbif.columns[1]
    col_esp_bien = 'species' if 'species' in df_bien.columns else df_bien.columns[0]
    
    df_gbif['especie_limpia'] = df_gbif[col_esp_gbif].astype(str).str.replace('_', ' ').str.strip()
    df_bien['especie_limpia'] = df_bien[col_esp_bien].astype(str).str.replace('_', ' ').str.strip()
    df_gbif[col_num_gbif] = pd.to_numeric(df_gbif[col_num_gbif], errors='coerce')
    
    df_gbif_filtrado = df_gbif[df_gbif[col_num_gbif] > 150]
    especies_comunes = pd.merge(df_gbif_filtrado, df_bien, on='especie_limpia')
    lista_final = especies_comunes['especie_limpia'].unique().tolist()
    
    if len(lista_final) == 0:
        print("[ERROR] Ninguna especie cumple con >150 puntos y presencia en ambos catálogos.")
        return
        
    if len(lista_final) < 15:
        print(f"[ADVERTENCIA] Solo hay {len(lista_final)} especies válidas. Se usarán todas.")
        especies_seleccionadas = lista_final
    else:
        especies_seleccionadas = random.sample(lista_final, 15)
        
    print(f"[INFO] Se seleccionaron {len(especies_seleccionadas)} especies aleatorias:")
    for esp in especies_seleccionadas:
        print(f"  - {esp}")
        
    banco_preguntas = [
        "¿Cuál es el Clima y Zona Biológica más importante para esta especie?",
        "¿Qué variables bioclimáticas restringen más la distribución espacial de esta planta?",
        "Basado en el modelo espacial, ¿cuál ecosistema o zona de vida es el refugio óptimo para su supervivencia?",
        "¿Cómo afecta el estrés climático a la idoneidad del hábitat de esta especie en el mapa?",
        "Describe el nicho ecológico de la especie y menciona cuál es su principal limitante ambiental.",
        "¿En qué condiciones geográficas de temperatura o precipitación prospera mejor esta especie según los datos?",
        "Analiza el factor limitante principal descubierto por la IA y cómo impacta su distribución en Costa Rica."
    ]
    
    archivo_maestro = os.path.join(DIR_RAIZ, "master_resultados_anova.csv")
    
    if not os.path.exists(archivo_maestro):
        with open(archivo_maestro, 'w', encoding='utf-8') as f:
            f.write("Especie,Modelo,Pregunta,Precision_Biologica,Coherencia_Espacial,Causalidad_Matematica,Promedio_Total\n")
            
    # Rutas absolutas a los scripts
    script_main = os.path.join(DIR_RAIZ, "main.py")
    script_eval = os.path.join(DIR_RAIZ, "evaluator", "evaluator.py")
    
    for i, especie in enumerate(especies_seleccionadas, 1):
        pregunta_actual = random.choice(banco_preguntas)
        
        print("\n" + "="*60)
        print(f"[BATCH {i}/{len(especies_seleccionadas)}] Especie: {especie}")
        print(f"[PREGUNTA ASIGNADA]: {pregunta_actual}")
        print("="*60)
        
        # A) Ejecutar pipeline obligando al subproceso a ubicarse en DIR_RAIZ
        comando_main = [sys.executable, script_main, "-s", especie, "-q", pregunta_actual]
        subprocess.run(comando_main, cwd=DIR_RAIZ)
        
        # B) Buscar la carpeta generada usando ruta absoluta
        nombre_carpeta = especie.replace(" ", "_")
        patron_busqueda = os.path.join(DIR_RAIZ, "outputs", nombre_carpeta, "run_*")
        carpetas = glob.glob(patron_busqueda)
        
        if not carpetas:
            print(f"[ADVERTENCIA] El modelo falló o no se generó carpeta para {especie}. Saltando.")
            continue
            
        carpetas.sort(key=os.path.getmtime, reverse=True)
        carpeta_reciente = carpetas[0]
        
        # C) Ejecutar el Evaluador con ruta absoluta
        print(f"\n[INFO] Evaluando resultados en: {carpeta_reciente}")
        comando_eval = [sys.executable, script_eval, "-d", carpeta_reciente]
        subprocess.run(comando_eval, cwd=DIR_RAIZ)
        
        # D) Inyectar pregunta y consolidar
        csv_eval = os.path.join(carpeta_reciente, "resultados_anova.csv")
        if os.path.exists(csv_eval):
            try:
                df_eval = pd.read_csv(csv_eval)
                df_eval.insert(2, 'Pregunta', pregunta_actual)
                df_eval.to_csv(archivo_maestro, mode='a', header=False, index=False)
                print(f"[EXITO] Calificaciones de {especie} añadidas al master_resultados_anova.csv")
            except Exception as e:
                print(f"[ERROR] Al consolidar CSV de {especie}: {e}")
        else:
            print(f"[ERROR] El evaluador no generó el CSV para {especie}.")

    print("\n" + "="*60)
    print(f"[FIN DEL EXPERIMENTO] Todas las especies procesadas.")
    print(f"Dataset final en: {archivo_maestro}")
    print("="*60)

if __name__ == "__main__":
    main()