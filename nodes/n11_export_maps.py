"""
N11: ExportMaps
Genera y exporta visualizaciones espaciales: overview mesoamericano,
matriz de confusión, mapa de hábitat manual y solapamiento RF.

INPUT  (state):
    - species_name    : nombre científico de la especie
    - meso_boundary   : GeoDataFrame con el límite de Mesoamérica
    - presencias_meso : GeoDataFrame con ocurrencias en Mesoamérica
    - presencias_cr   : GeoDataFrame con ocurrencias en CR
    - rf_metrics      : dict con confusion_matrix
    - output_dir      : directorio de salida de la ejecución
    - habitat_map_path: ruta al mapa manual (opcional, desde N04)

OUTPUT (state):
    - exported_maps      : dict {nombre_mapa: ruta_png} con los mapas generados
    - habitat_rf_map_path: ruta al mapa de solapamiento RF, si existe en disco

ARCHIVO: n11_export_maps.json
"""
import os
from graph_state import GraphState
from utils.visualizer import Visualizer
from node_utils import save_node_json


def export_maps_node(state: GraphState) -> GraphState:
    """
    Nodo N11: genera mapas de distribución, confusión y hábitat.
    Cada mapa se intenta de forma independiente; los fallos no detienen
    el pipeline.

    Parámetros:
        state (GraphState): estado del pipeline.

    Retorna:
        GraphState: estado actualizado con exported_maps
                    y habitat_rf_map_path.
    """
    print("\n[N11:ExportMaps] Exportando visualizaciones...")
    species_name    = state['species_name']
    meso_boundary   = state['meso_boundary']
    presencias_meso = state['presencias_meso']
    presencias_cr   = state['presencias_cr']
    rf_metrics      = state['rf_metrics']
    output_dir      = state['output_dir']

    vis = Visualizer()
    exported_maps = {}

    # Mapa 1: Overview Mesoamericano
    try:
        overview_path = vis.plot_mesoamerica_overview(
            species_name=species_name,
            meso_boundary=meso_boundary,
            expert_map=None,
            presencias_meso=presencias_meso,
            presencias_cr=presencias_cr,
            output_dir=output_dir
        )
        exported_maps['mesoamerica_overview'] = overview_path
    except Exception as e:
        print(f"[N11:WARN] Mapa Mesoamérica falló: {e}")

    # Mapa 2: Matriz de confusión
    try:
        if 'confusion_matrix' in rf_metrics:
            confusion_path = vis.plot_confusion_matrix(
                rf_metrics['confusion_matrix'],
                output_dir
            )
            exported_maps['confusion_matrix'] = confusion_path
    except Exception as e:
        print(f"[N11:WARN] Matriz de confusión falló: {e}")

    # Mapa 3: Hábitat Manual (si existe desde N04)
    if state.get('habitat_map_path'):
        exported_maps['habitat_manual'] = state.get('habitat_map_path')

    # Mapa 4: Solapamiento RF (si fue generado en disco por otro proceso)
    habitat_rf_path = os.path.join(output_dir, "mapa_solapamiento_espacial.png")
    if os.path.exists(habitat_rf_path):
        exported_maps['habitat_rf'] = habitat_rf_path
        state['habitat_rf_map_path'] = habitat_rf_path

    state['exported_maps'] = exported_maps

    save_node_json({
        "node": "N11_ExportMaps",
        "mapas_generados": len([v for v in exported_maps.values() if v]),
        "mapas": exported_maps,
        "status": "ok"
    }, output_dir, "n11_export_maps.json")

    print(f"[N11:✓] Mapas exportados: {len([v for v in exported_maps.values() if v])}")
    return state