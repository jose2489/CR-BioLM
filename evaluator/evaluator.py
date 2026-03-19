import os
import sys
import glob
import json
import csv
import argparse
from openai import OpenAI

# Configuración de Rutas
DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIR_RAIZ = os.path.dirname(DIR_ACTUAL)
sys.path.append(DIR_RAIZ)
import config

def extract_metadata_and_text(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines or len(lines) < 3:
            return None, None, None
        
        modelo_str = lines[0].strip().replace("Modelo LLM: ", "").strip()
        especie_str = lines[1].strip().replace("Especie: ", "").strip()
        texto_perfil = "".join(lines[2:]).strip()
        
        return modelo_str, especie_str, texto_perfil
    except Exception as e:
        print(f"[ERROR] Leyendo archivo {filepath}: {e}")
        return None, None, None

def evaluate_profile(client, texto_perfil):
    prompt = f"""Eres un ecólogo experto evaluando respuestas de modelos de IA para una tesis de maestría.
Lee este perfil ecológico:
{texto_perfil}

Evalúa los siguientes 3 criterios asignando un puntaje entero del 1 al 5 (donde 1 es deficiente y 5 es excelente):
1. Precision_Biologica: ¿La explicación del nicho ecológico tiene sentido fisiológico y no contradice las leyes de la biología tropical?
2. Coherencia_Espacial: ¿La distribución descrita se alinea lógicamente con la geografía y zonas de vida mencionadas?
3. Causalidad_Matematica: ¿El texto basa sus conclusiones en datos o métricas específicas (ej. variables climáticas, impacto negativo/positivo) en lugar de hacer suposiciones geográficas genéricas?

Devuelve ÚNICAMENTE un objeto JSON válido con este formato exacto:
{{"Precision_Biologica": INT, "Coherencia_Espacial": INT, "Causalidad_Matematica": INT}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Eres un juez evaluador estricto y objetivo que solo responde en JSON puro."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, 
            response_format={"type": "json_object"} 
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[ERROR] Evaluando con Groq: {e}")
        return None

def obtener_ejecucion_mas_reciente(base_outputs):
    """Busca la carpeta run_... con la fecha de modificación más reciente."""
    search_pattern = os.path.join(base_outputs, "*", "run_*")
    carpetas = glob.glob(search_pattern)
    if not carpetas:
        return None
    carpetas.sort(key=os.path.getmtime, reverse=True)
    return carpetas[0]

def main():
    # 1. Configurar Argumentos de CLI
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge para evaluar perfiles ecológicos.")
    parser.add_argument("-d", "--dir", type=str, help="Ruta a la carpeta de ejecución específica (ej. outputs/Quercus/run_123)")
    args = parser.parse_args()

    print("="*60)
    print("[INFO] Iniciando el LLM-as-a-Judge (Evaluador Automático)")
    print("="*60)
    
    if not hasattr(config, 'GROQ_API_KEY') or not config.GROQ_API_KEY:
        print("[ERROR] GROQ_API_KEY no encontrada en config.py")
        return

    client = OpenAI(
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    
    # 2. Determinar qué carpeta evaluar
    base_outputs = os.path.join(DIR_RAIZ, "outputs")
    
    if args.dir:
        target_dir = os.path.abspath(args.dir)
        print(f"[INFO] Carpeta especificada por parámetro:\n       {target_dir}\n")
    else:
        target_dir = obtener_ejecucion_mas_reciente(base_outputs)
        if target_dir:
            print(f"[INFO] Detectando automáticamente la ejecución MÁS RECIENTE:\n       {target_dir}\n")
        else:
            print(f"[ERROR] No se encontraron carpetas de ejecución en {base_outputs}")
            return
    
    # 3. Buscar solo en esa carpeta
    search_pattern = os.path.join(target_dir, "llm_profile_*.txt")
    archivos = glob.glob(search_pattern)
    
    if not archivos:
        print(f"[ADVERTENCIA] No hay archivos de texto en esa carpeta para evaluar.")
        return
        
    resultados_totales = []
    
    for idx, filepath in enumerate(archivos, 1):
        nombre_archivo = os.path.basename(filepath)
        print(f"[{idx}/{len(archivos)}] Analizando: {nombre_archivo}")
        
        modelo, especie, texto = extract_metadata_and_text(filepath)
        if not texto: continue
            
        puntajes = evaluate_profile(client, texto)
        if puntajes:
            pb = puntajes.get("Precision_Biologica", 0)
            ce = puntajes.get("Coherencia_Espacial", 0)
            cm = puntajes.get("Causalidad_Matematica", 0)
            promedio = round((pb + ce + cm) / 3.0, 2)
            
            resultados_totales.append({
                "Especie": especie, "Modelo": modelo,
                "Precision_Biologica": pb, "Coherencia_Espacial": ce,
                "Causalidad_Matematica": cm, "Promedio_Total": promedio
            })
            print(f"  -> Calificación: PB({pb}) CE({ce}) CM({cm}) | Promedio: {promedio}")
            
    # 4. Guardar el CSV DENTRO de la carpeta evaluada
    csv_filename = os.path.join(target_dir, "resultados_anova.csv")
    if resultados_totales:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=resultados_totales[0].keys())
            writer.writeheader()
            writer.writerows(resultados_totales)
        print(f"\n[EXITO] CSV guardado junto a los perfiles en:\n{csv_filename}")

if __name__ == "__main__":
    main()