"""
Definición del grafo LangGraph para CR-BioLM.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Literal
from langgraph.graph import StateGraph, END
from nodes.graph_state import GraphState

# Grupo 1: Ingesta
from nodes.n01_load_geometry import load_geometry_node
from nodes.n02_fetch_gbif import fetch_gbif_node
from nodes.n03_load_climate import load_climate_node
from nodes.n04_load_habitat_map import load_habitat_map_node

# Grupo 2: Transformación
from nodes.n05_enrich_altitude import enrich_altitude_node
from nodes.n06_build_matrix import build_matrix_node

# Grupo 3: Modelado
from nodes.n07_train_random_forest import train_random_forest_node
from nodes.n08_predict import predict_node

# Grupo 4: Explicabilidad
from nodes.n09_explain_shap import explain_shap_node
from nodes.n10_explain_lime import explain_lime_node

# Grupo 5: Salidas
from nodes.n11_export_maps import export_maps_node
from nodes.n12_conservation_context import generate_conservation_context_node
from nodes.n13_generate_report import generate_report_node


def should_continue_after_gbif(state: GraphState) -> Literal["continue", "end"]:
    if not state.get('should_continue', True):
        print("\n[ROUTER] Pipeline detenido: sin datos de presencias válidos.")
        return "end"
    return "continue"


def create_workflow() -> StateGraph:
    workflow = StateGraph(GraphState)

    # Registrar nodos
    workflow.add_node("load_geometry", load_geometry_node)
    workflow.add_node("fetch_gbif", fetch_gbif_node)
    workflow.add_node("load_climate", load_climate_node)
    workflow.add_node("load_habitat_map", load_habitat_map_node)
    workflow.add_node("enrich_altitude", enrich_altitude_node)
    workflow.add_node("build_matrix", build_matrix_node)
    workflow.add_node("train_random_forest", train_random_forest_node)
    workflow.add_node("predict", predict_node)
    workflow.add_node("explain_shap", explain_shap_node)
    workflow.add_node("explain_lime", explain_lime_node)
    workflow.add_node("export_maps", export_maps_node)
    workflow.add_node("generate_conservation_context", generate_conservation_context_node)
    workflow.add_node("generate_report", generate_report_node)

    # Flujo
    workflow.set_entry_point("load_geometry")
    workflow.add_edge("load_geometry", "fetch_gbif")

    workflow.add_conditional_edges(
        "fetch_gbif",
        should_continue_after_gbif,
        {"continue": "load_climate", "end": END}
    )

    workflow.add_edge("load_climate", "load_habitat_map")
    workflow.add_edge("load_habitat_map", "enrich_altitude")
    workflow.add_edge("enrich_altitude", "build_matrix")
    workflow.add_edge("build_matrix", "train_random_forest")
    workflow.add_edge("train_random_forest", "predict")
    workflow.add_edge("predict", "explain_shap")
    workflow.add_edge("explain_shap", "explain_lime")
    workflow.add_edge("explain_lime", "export_maps")
    workflow.add_edge("export_maps", "generate_conservation_context")
    workflow.add_edge("generate_conservation_context", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()


def ejecutar_pipeline(
    species_name: str,
    output_dir: str,
    user_question: str = None,
) -> GraphState:

    initial_state: GraphState = {
        'species_name': species_name,
        'output_dir': output_dir,
        'user_question': user_question,
        'error_messages': [],
        'should_continue': True
    }

    app = create_workflow()

    print("=" * 60)
    print(f"[INICIO] Ejecutando pipeline LangGraph para: {species_name.upper()}")
    print("=" * 60)

    final_state = app.invoke(initial_state)

    if final_state.get('error_messages'):
        print("\n[ADVERTENCIAS/ERRORES DURANTE LA EJECUCIÓN]:")
        for i, error in enumerate(final_state['error_messages'], 1):
            print(f"  {i}. {error}")

    print(f"\n[ÉXITO] Pipeline completado para {species_name}.")
    print(f"[INFO] Resultados guardados en: {output_dir}")

    return final_state


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
    import argparse
    import config
    from utils.question_bank import get_random_question

    parser = argparse.ArgumentParser(description="Ejecutor del Pipeline CR-BioLM con LangGraph.")
    parser.add_argument("-s", "--species", type=str, help="Nombre de la especie. Ej: 'Quercus costaricensis'")
    parser.add_argument("-f", "--file", type=str, help="Ruta a un archivo .txt con una lista de especies (una por línea).")
    parser.add_argument("-q", "--question", type=str, default=None, help="Pregunta libre para el LLM.")
    parser.add_argument("--persona", type=str, default=None, choices=["turista", "botanico", "municipalidad"],
                        help="Selecciona una pregunta aleatoria del banco según el perfil de usuario.")
    parser.add_argument("--canton", type=str, default=None,
                        help="Cantón para preguntas de municipalidad (ej: 'Nicoya').")
    parser.add_argument("--proyecto", type=str, default=None,
                        help="Tipo de proyecto para municipalidad (ej: 'proyecto hidroeléctrico').")

    args = parser.parse_args()

    def resolver_pregunta():
        if args.question:
            return args.question
        if args.persona:
            q = get_random_question(args.persona, canton=args.canton, proyecto=args.proyecto)
            print(f"[INFO] Pregunta seleccionada [{args.persona}]: {q}")
            return q
        return None

    if args.species:
        output_dir = config.crear_directorio_ejecucion(args.species)
        ejecutar_pipeline(
            species_name=args.species,
            output_dir=output_dir,
            user_question=resolver_pregunta(),
        )

    elif args.file:
        lista_especies = leer_lista_especies(args.file)
        print(f"[INFO] Modo Batch iniciado. Se detectaron {len(lista_especies)} especies en el archivo.")
        for i, especie in enumerate(lista_especies, 1):
            print(f"\n[LOTE {i}/{len(lista_especies)}]")
            output_dir = config.crear_directorio_ejecucion(especie)
            ejecutar_pipeline(
                species_name=especie,
                output_dir=output_dir,
                user_question=resolver_pregunta(),  # cada especie obtiene pregunta independiente si --persona
            )

    else:
        print("[ERROR] Debes proporcionar un argumento. Usa -s para una especie o -f para una lista txt.")
        print("Ejemplo 1: python workflow.py -s \"Quercus costaricensis\"")
        print("Ejemplo 2: python workflow.py -s \"Quercus costaricensis\" -q \"¿Cómo le afecta el cambio climático?\"")
        print("Ejemplo 3: python workflow.py -s \"Quercus costaricensis\" --persona botanico")
        print("Ejemplo 4: python workflow.py -s \"Quercus costaricensis\" --persona municipalidad --canton Nicoya")
        print("Ejemplo 5: python workflow.py -f especies.txt")
        print("Ejemplo 6: python workflow.py -f especies.txt -q \"¿Cuál es el rango altitudinal óptimo?\"")
        print("Ejemplo 7: python workflow.py -f especies.txt --persona turista")