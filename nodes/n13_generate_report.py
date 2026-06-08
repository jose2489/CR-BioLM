"""
N13: GenerateReport
Genera el reporte final de perfil de especie usando un LLM vía OpenRouter.
El modelo se configura en nodes/config_llm.py.

INPUT  (state):
    - species_name       : nombre científico de la especie
    - rf_metrics         : dict con métricas del modelo Random Forest
    - shap_data          : dict con resultados SHAP
    - output_dir         : directorio de salida de la ejecución
    - user_question      : pregunta opcional del usuario (puede ser None)
    - info_altitud       : string con rango altitudinal (desde N05)
    - texto_manual       : texto del Manual de Plantas (desde N04)
    - exported_maps      : dict con rutas de mapas generados (desde N11)
    - habitat_map_path   : ruta al mapa de hábitat manual (desde N04)
    - habitat_rf_map_path: ruta al mapa de solapamiento RF (desde N11)

OUTPUT (state):
    - final_report_path : ruta al archivo .txt del reporte generado,
                          o None si el LLM falló
    - error_messages    : append si el LLM falla

ARCHIVO: n13_generate_report.json
"""
import os
from graph_state import GraphState
from llm.openrouter_client import OpenRouterClient
from config_llm import LLM_MODEL, is_multimodal
from node_utils import save_node_json
import config


def generate_report_node(state: GraphState) -> GraphState:
    """
    Nodo N13: llama al LLM para generar el perfil ecológico de la especie.
    Si el modelo es multimodal, adjunta imágenes disponibles.
    Si generate_profile retorna False o lanza excepción, registra el error
    con status 'error' y no asigna final_report_path.

    Parámetros:
        state (GraphState): estado del pipeline.

    Retorna:
        GraphState: estado actualizado con final_report_path (o None si falló).
    """
    print("\n[N13:GenerateReport] Generando reporte con LLM...")
    species_name  = state['species_name']
    rf_metrics    = state['rf_metrics']
    shap_data     = state['shap_data']
    output_dir    = state['output_dir']
    user_question = state.get('user_question')
    info_altitud  = state.get('info_altitud', 'No disponible')
    texto_manual  = state.get('texto_manual', '')

    # Resolver imágenes solo si el modelo es multimodal
    imagen_principal = None
    habitat_rf_path  = None
    if is_multimodal(LLM_MODEL):
        exported_maps = state.get('exported_maps', {})
        imagen_principal = (
            state.get('habitat_map_path') or
            exported_maps.get('mesoamerica_overview') or
            exported_maps.get('confusion_matrix')
        )
        habitat_rf_path = state.get('habitat_rf_map_path')

        if imagen_principal and not os.path.exists(imagen_principal):
            print(f"[N13:WARN] Imagen principal no existe: {imagen_principal}")
            imagen_principal = None
        if habitat_rf_path and not os.path.exists(habitat_rf_path):
            print(f"[N13:WARN] Mapa RF no existe: {habitat_rf_path}")
            habitat_rf_path = None

    try:
        or_client = OpenRouterClient(api_key=config.OPENROUTER_API_KEY)
        exito = or_client.generate_profile(
            species_name=species_name,
            rf_metrics=rf_metrics,
            shap_dict=shap_data,
            output_dir=output_dir,
            image_path=imagen_principal,
            user_question=user_question,
            model_override=LLM_MODEL,
            info_altitud=info_altitud,
            manual_image_path=habitat_rf_path,
            texto_manual=texto_manual
        )

        # generate_profile retorna False ante cualquier fallo interno
        if not exito:
            raise RuntimeError(
                "generate_profile retornó False. "
                "Revisar llm_api_error.txt en el directorio de salida."
            )

        modelo_limpio = LLM_MODEL.replace('/', '_').replace('-', '_').replace(':', '_')
        final_report_path = os.path.join(
            output_dir, f"llm_profile_BIMODAL_{modelo_limpio}.txt"
        )

        # Verificar que el archivo fue realmente creado
        if not os.path.exists(final_report_path):
            raise RuntimeError(
                f"generate_profile reportó éxito pero el archivo no existe: {final_report_path}"
            )

        state['final_report_path'] = final_report_path

        save_node_json({
            "node": "N13_GenerateReport",
            "species_name": species_name,
            "modelo_usado": LLM_MODEL,
            "es_multimodal": is_multimodal(LLM_MODEL),
            "imagen_principal": imagen_principal,
            "imagen_rf": habitat_rf_path,
            "user_question": user_question,
            "report_path": final_report_path,
            "status": "ok"
        }, output_dir, "n13_generate_report.json")

        print(f"[N13:✓] Reporte generado: {final_report_path}")

    except Exception as e:
        print(f"[N13:ERROR] Fallo en generación de reporte: {e}")
        state['error_messages'].append(f"Reporte no disponible: {e}")
        state['final_report_path'] = None
        save_node_json({
            "node": "N13_GenerateReport",
            "modelo_usado": LLM_MODEL,
            "status": "error",
            "error": str(e)
        }, output_dir, "n13_generate_report.json")

    return state