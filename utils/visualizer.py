import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

class Visualizer:
    """
    Clase dedicada a la generacion y exportacion de mapas cartograficos 
    y otras utilidades visuales de alta resolucion.
    """
    
    def plot_spatial_overlap(self, species_name, country_boundary, expert_map, presencias_gdf, output_dir):
        """
        Genera un mapa estatico superponiendo la capa base del pais, 
        el poligono de rango experto (si existe) y los puntos de presencia empiricos.
        """
        print("[INFO] Generando mapa multicapa de distribucion espacial...")
        
        # Configuracion del lienzo (lienzo cuadrado ideal para Costa Rica)
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 1. Capa Base: Fronteras del pais
        if country_boundary is not None and not country_boundary.empty:
            country_boundary.plot(ax=ax, color='#e5e7eb', edgecolor='#374151', alpha=0.8)
            
        # 2. Capa Experta: Rango de BIEN/IUCN
        if expert_map is not None and not expert_map.empty:
            expert_map.plot(
                ax=ax, 
                color='#10b981', # Verde esmeralda
                alpha=0.4, 
                edgecolor='#047857',
                label='Rango Experto (BIEN)'
            )
            
        # 3. Capa Empirica: Coordenadas limpias de GBIF
        if presencias_gdf is not None and not presencias_gdf.empty:
            presencias_gdf.plot(
                ax=ax, 
                color='#ef4444', # Rojo vibrante
                markersize=25, 
                alpha=0.8, 
                edgecolor='black',
                linewidth=0.5,
                label='Presencias (GBIF)'
            )
            
        # Estetica academica del grafico
        plt.title(f"Distribucion de la Especie\n{species_name}", fontsize=14, fontweight='bold', fontstyle='italic')
        plt.xlabel("Longitud", fontsize=11)
        plt.ylabel("Latitud", fontsize=11)
        
        # Agregar cuadrilla (grid) muy sutil
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Configurar la leyenda de forma segura
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='lower right', frameon=True, shadow=True)
       
        # ==========================================
        # --- LEYENDA PERSONALIZADA ---
        # ==========================================
        # 3.1. Creamos el cuadro para el mapa experto
            expert_patch = mpatches.Patch(
            facecolor='#73C6A3', 
            edgecolor='green', 
            alpha=0.6, 
            label='Mapa Experto (BIEN / IUCN)'
        )
        
        # 3.2. Creamos el punto rojo para GBIF
        gbif_marker = mlines.Line2D(
            [], [], color='white', marker='o', 
            markerfacecolor='#ff4d4d', markeredgecolor='black', 
            markersize=7, label='Presencias (GBIF)'
        )
        
        # 3.3. Forzamos a la gráfica a usar nuestra leyenda
        plt.legend(handles=[expert_patch, gbif_marker], loc='lower right', shadow=True, framealpha=0.9)
            
        # Exportacion en alta resolucion (300 DPI requerido para papers)
        file_path = os.path.join(output_dir, "mapa_solapamiento_espacial.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Mapa guardado exitosamente en: {file_path}")
        return file_path
    
    def plot_confusion_matrix(self, conf_matrix, output_dir):
        """
        Genera y guarda un mapa de calor visual para la Matriz de Confusión.
        """
        print("[INFO] Generando gráfico de Matriz de Confusión...")
        plt.figure(figsize=(6, 5))
        
        # Mapa de calor con estilo científico
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Ausencia (0)', 'Presencia (1)'],
                    yticklabels=['Ausencia (0)', 'Presencia (1)'])
        
        plt.xlabel('Predicción del Modelo', fontsize=11, fontweight='bold')
        plt.ylabel('Realidad (GBIF / Pseudo-ausencias)', fontsize=11, fontweight='bold')
        plt.title('Matriz de Confusión del Modelo', fontsize=13)
        
        plot_path = os.path.join(output_dir, "matriz_confusion.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"[INFO] Matriz de Confusión guardada en: {plot_path}")