"""
N11: ExportMaps
Genera y exporta visualizaciones espaciales.
"""
import os
import json
from graph_state import GraphState
from utils.visualizer import Visualizer


def export_maps_node(state: GraphState) -> GraphState:
    print("\n[N11:ExportMaps] Exportando visualizaciones...")

    species_name = state['species_name']
    meso_boundary = state['meso_boundary']
    presencias_meso = state['presencias_meso']
    presencias_cr = state['presencias_cr']
    rf_metrics = state['rf_metrics']
    output_dir = state['output_dir']

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

    # Mapa 3: Hábitat Manual (si existe)
    if state.get('habitat_map_path'):
        exported_maps['habitat_manual'] = state.get('habitat_map_path')

    # Mapa 4: Solapamiento RF (si existe en disco)
    habitat_rf_path = os.path.join(output_dir, "mapa_solapamiento_espacial.png")
    if os.path.exists(habitat_rf_path):
        exported_maps['habitat_rf'] = habitat_rf_path
        state['habitat_rf_map_path'] = habitat_rf_path

    state['exported_maps'] = exported_maps

    _save_json({
        "node": "N1_ExportMaps",
        "mapas_generados": len([v for v in exported_maps.values() if v]),
        "mapas": exported_maps,
        "status": "ok"
    }, output_dir, "n11_export_maps.json")

    print(f"[N11:✓] Mapas exportados: {len([v for v in exported_maps.values() if v])}")
    return state


def _save_json(data: dict, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
