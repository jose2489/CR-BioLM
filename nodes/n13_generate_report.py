"""
N13: GenerateReport
Genera reporte final usando LLM vía OpenRouter.
El modelo se configura en nodes/config_llm.py.
"""
import os
import json
from graph_state import GraphState
from llm.openrouter_client import OpenRouterClient
from config_llm import LLM_MODEL, is_multimodal
import config


def generate_report_node(state: GraphState) -> GraphState:
    print("\n[N13:GenerateReport] Generando reporte con LLM...")

    species_name = state['species_name']
    rf_metrics = state['rf_metrics']
    shap_data = state['shap_data']
    output_dir = state['output_dir']
    user_question = state.get('user_question')
    info_altitud = state.get('info_altitud', 'No disponible')
    texto_manual = state.get('texto_manual', '')

    # Resolver imágenes solo si el modelo es multimodal
    imagen_principal = None
    habitat_rf_path = None

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

        or_client.generate_profile(
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

        modelo_limpio = LLM_MODEL.replace('/', '_').replace('-', '_').replace(':', '_')
        final_report_path = os.path.join(
            output_dir, f"llm_profile_BIMODAL_{modelo_limpio}.txt"
        )

        state['final_report_path'] = final_report_path

        _save_json({
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
        _save_json({
            "node": "N13_GenerateReport",
            "modelo_usado": LLM_MODEL,
            "status": "error",
            "error": str(e)
        }, output_dir, "n13_generate_report.json")

    return state


def _save_json(data: dict, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
