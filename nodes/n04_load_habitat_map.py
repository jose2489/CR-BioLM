"""
N04: LoadHabitatMap
Genera el mapa de hábitat desde el Manual de Plantas de Costa Rica.
Nodo completamente opcional: nunca detiene el pipeline.

INPUT  (state):
    - species_name  : nombre científico de la especie
    - presencias_cr : GeoDataFrame con ocurrencias en CR
    - output_dir    : directorio de salida de la ejecución

OUTPUT (state):
    - habitat_map_path : ruta al PNG generado, o None si no aplica
    - texto_manual     : texto descriptivo del manual para el reporte LLM,
                         o string vacío si no hay datos

ARCHIVO: n04_load_habitat_map.json
"""
import os
import pandas as pd
from graph_state import GraphState
from utils.map_gen.habitat_map import generate_habitat_map
from node_utils import save_node_json

CATALOG_PATH = os.path.join("outputs", "picked_species_enhanced_clean.csv")


def load_habitat_map_node(state: GraphState) -> GraphState:
    """
    Nodo N04: genera mapa de hábitat y ficha de referencia desde el
    catálogo del Manual de Plantas de Costa Rica.

    Parámetros:
        state (GraphState): estado del pipeline.

    Retorna:
        GraphState: estado actualizado con habitat_map_path y texto_manual.
                    Si la especie no está en el catálogo o falla la carga,
                    continúa sin interrumpir el pipeline.
    """
    print("\n[N04:LoadHabitatMap] Generando mapa de hábitat desde Manual de Plantas...")
    species_name = state['species_name']
    presencias_cr = state['presencias_cr']
    output_dir = state['output_dir']

    ruta_mapa_manual = None
    texto_manual = ""
    ficha_data = {}

    try:
        catalog = pd.read_csv(CATALOG_PATH)
        row = catalog[catalog['species'] == species_name]

        if not row.empty:
            r = row.iloc[0]
            ruta_mapa_manual = generate_habitat_map(
                species_name=species_name,
                geographic_notes=r.get('geographic_notes'),
                elevation_min=r.get('elevation_min_m'),
                elevation_max=r.get('elevation_max_m'),
                presencias_gdf=presencias_cr,
                output_path=os.path.join(output_dir, "mapa_habitat_manual.png"),
            )

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

            ficha_data = {
                "especie": species_name,
                "familia": r.get('family', 'N/D'),
                "volumen": r.get('volume_title', 'N/D'),
                "distribucion_geografica": r.get('geographic_notes', 'N/D'),
                "altitud_min_m": r.get('elevation_min_m'),
                "altitud_max_m": r.get('elevation_max_m'),
                "habitat_type": r.get('habitat_type', 'N/D'),
                "habitat_raw": r.get('habitat_raw', 'N/D'),
                "ocurrencias_gbif_catalogo": r.get('occurrences', 'N/D'),
            }

            # Ficha legible en TXT
            nombre_limpio = species_name.replace(" ", "_")
            ruta_ficha = os.path.join(output_dir, f"{nombre_limpio}_ficha_MdP.txt")
            with open(ruta_ficha, "w", encoding="utf-8") as f:
                f.write("FICHA DE REFERENCIA — Manual de Plantas de Costa Rica\n")
                f.write(f"{'='*60}\n")
                for k, v in ficha_data.items():
                    f.write(f"{k:<30}: {v}\n")

            print(f"[N04:✓] Mapa: {ruta_mapa_manual}")
        else:
            print(f"[N04:INFO] '{species_name}' no encontrada en catálogo Manual.")

    except Exception as e:
        print(f"[N04:INFO] Catálogo no disponible - continuando sin él. (Detalle: {e})")

    state['habitat_map_path'] = ruta_mapa_manual
    state['texto_manual'] = texto_manual

    save_node_json({
        "node": "N04_LoadHabitatMap",
        "species_name": species_name,
        "habitat_map_generated": ruta_mapa_manual is not None,
        "habitat_map_path": ruta_mapa_manual,
        "texto_manual_disponible": bool(texto_manual),
        "ficha": ficha_data,
        "status": "ok"
    }, output_dir, "n04_load_habitat_map.json")

    return state