import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def main():
    print("="*60)
    print("[INFO] Iniciando Análisis Estadístico (ANOVA y Gráficos)")
    print("="*60)

    archivo_maestro = "master_resultados_anova.csv"
    carpeta_salida = "analisis_resultados"

    if not os.path.exists(archivo_maestro):
        print(f"[ERROR] No se encontró el archivo {archivo_maestro}. Corre el run_batch.py primero.")
        return

    # Crear carpeta para guardar los gráficos
    os.makedirs(carpeta_salida, exist_ok=True)

    # 1. Cargar datos
    df = pd.read_csv(archivo_maestro)
    print(f"[INFO] Se cargaron {len(df)} evaluaciones del archivo maestro.")

    # 2. Limpieza de nombres de modelos para que los gráficos se vean bien
    mapeo_nombres = {
        'Baseline Dual (Visión OpenRouter + Texto Groq)': 'Baseline (Visión)',
        'llama-3.3-70b-versatile': 'Llama 3.3 (70B)',
        'qwen/qwen3-32b': 'Qwen3 (32B)',
        'moonshotai/kimi-k2-instruct-0905': 'Kimi K2'
    }
    # Si hay modelos nuevos, los deja igual; si están en el diccionario, los acorta
    df['Modelo_Corto'] = df['Modelo'].replace(mapeo_nombres)

    # =================================================================
    # FASE ESTADÍSTICA: ANOVA de 1 vía
    # =================================================================
    print("\n[ESTADÍSTICA] Ejecutando One-Way ANOVA sobre Promedio_Total...")
    modelos_unicos = df['Modelo_Corto'].unique()
    
    # Agrupamos los puntajes por modelo
    grupos_puntajes = [df[df['Modelo_Corto'] == modelo]['Promedio_Total'].dropna() for modelo in modelos_unicos]
    
    f_stat, p_value = stats.f_oneway(*grupos_puntajes)
    
    print(f" -> Estadístico F: {f_stat:.4f}")
    print(f" -> Valor p (p-value): {p_value:.4e}")

    if p_value < 0.05:
        print("\n[CONCLUSION] Existen diferencias significativas (p < 0.05) entre los modelos.")
        print("[ESTADÍSTICA] Ejecutando Prueba Post-Hoc de Tukey HSD...")
        
        tukey = pairwise_tukeyhsd(endog=df['Promedio_Total'], groups=df['Modelo_Corto'], alpha=0.05)
        print("\n" + str(tukey))
        
        # Guardar resultados Tukey en TXT
        with open(os.path.join(carpeta_salida, "tukey_resultados.txt"), "w") as f:
            f.write(str(tukey))
    else:
        print("\n[CONCLUSION] No se encontraron diferencias estadísticamente significativas (p >= 0.05).")

    # =================================================================
    # FASE DE VISUALIZACIÓN
    # =================================================================
    print("\n[INFO] Generando gráficos de alta calidad...")
    sns.set_theme(style="whitegrid")

    # --- Gráfico 1: Boxplot General (Promedio Total) ---
    plt.figure(figsize=(10, 6))
    # Ordenamos por mediana para que el mejor quede a la derecha/arriba
    orden_modelos = df.groupby('Modelo_Corto')['Promedio_Total'].median().sort_values(ascending=False).index
    
    ax1 = sns.boxplot(x='Modelo_Corto', y='Promedio_Total', data=df, order=orden_modelos, palette="Set2")
    # Agregamos los puntos reales de fondo para ver la dispersión
    sns.stripplot(x='Modelo_Corto', y='Promedio_Total', data=df, order=orden_modelos, color=".25", alpha=0.5, jitter=True)
    
    plt.title('Distribución de Calificaciones Globales por Modelo', fontsize=14, pad=15)
    plt.ylabel('Puntaje Promedio (1-5)', fontsize=12)
    plt.xlabel('Arquitectura Evaluada', fontsize=12)
    plt.ylim(0.5, 5.5)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, "01_boxplot_promedio_total.png"), dpi=300)
    plt.close()

    # --- Gráfico 2: Desempeño desglosado por los 3 Criterios ---
    # Convertimos las 3 columnas de criterios en 2 columnas (Variable y Valor) para poder graficarlas juntas
    df_melt = pd.melt(df, id_vars=['Modelo_Corto'], 
                      value_vars=['Precision_Biologica', 'Coherencia_Espacial', 'Causalidad_Matematica'],
                      var_name='Criterio', value_name='Puntaje')
    
    # Limpiamos los nombres de los criterios
    df_melt['Criterio'] = df_melt['Criterio'].str.replace('_', ' ')

    plt.figure(figsize=(12, 7))
    ax2 = sns.barplot(x='Modelo_Corto', y='Puntaje', hue='Criterio', data=df_melt, 
                      order=orden_modelos, palette="muted", errorbar='sd', capsize=0.1)
    
    plt.title('Desempeño de los Modelos Desglosado por Criterio de Evaluación', fontsize=14, pad=15)
    plt.ylabel('Puntaje (Media ± Desviación Estándar)', fontsize=12)
    plt.xlabel('Arquitectura Evaluada', fontsize=12)
    plt.ylim(0, 5.5)
    plt.legend(title="Criterio Ecológico", loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, "02_barplot_criterios.png"), dpi=300)
    plt.close()

    print(f"[EXITO] Gráficos y reportes guardados en la carpeta: {carpeta_salida}/")
    print("="*60)

if __name__ == "__main__":
    main()