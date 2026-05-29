import sys
import os
import io
import json
import numpy as np
from job_store import add_log, set_done

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from workflow import ejecutar_pipeline
import config


class Logger(io.StringIO):
    def __init__(self, job_id):
        super().__init__()
        self.job_id = job_id

    def write(self, message):
        if message.strip():
            add_log(self.job_id, message.strip())
        super().write(message)


def make_serializable(obj):
    """Convierte recursivamente tipos numpy a tipos Python nativos."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def find_image(output_dir, filename):
    path = os.path.join(output_dir, filename)
    return path if os.path.exists(path) else None


def run_pipeline(job_id, species, question):
    logger = Logger(job_id)
    old_stdout = sys.stdout
    sys.stdout = logger

    try:
        add_log(job_id, "[INFO] Iniciando pipeline")
        output_dir = config.crear_directorio_ejecucion(species)

        final_state = ejecutar_pipeline(
            species_name=species,
            output_dir=output_dir,
            user_question=question
        )

        # Especie no encontrada en GBIF
        if not final_state.get('should_continue', True):
            error_msgs = final_state.get('error_messages', [])
            error_detail = error_msgs[0] if error_msgs else "Sin registros de presencia en Mesoamérica."
            add_log(job_id, f"[ERROR] {error_detail}")
            set_done(job_id, {
                "success": False,
                "species": species,
                "error": f"No se encontraron registros para '{species}' en GBIF. "
                         f"Verifique el nombre científico e intente nuevamente."
            })
            return

        add_log(job_id, "[INFO] Pipeline finalizado")

        # Leer modelo activo desde n13_generate_report.json
        llm_model = None
        n13_path = os.path.join(output_dir, "n13_generate_report.json")
        if os.path.exists(n13_path):
            with open(n13_path, encoding="utf-8") as f:
                llm_model = json.load(f).get("modelo_usado")

        result = {
            "success": True,
            "species": species,
            "output_dir": output_dir,
            "llm_model": llm_model,
            "images": {
                "mapa_distribucion": find_image(output_dir, "mapa_distribucion_mesoamerica.png"),
                "confusion_matrix":  find_image(output_dir, "matriz_confusion.png"),
                "shap":              find_image(output_dir, "shap_summary.png"),
                "lime":              find_image(output_dir, "lime_local_explanation.png"),
                "habitat_manual":    find_image(output_dir, "mapa_habitat_manual.png"),
            }
        }
        set_done(job_id, make_serializable(result))

    except Exception as e:
        error_msg = f"[ERROR FATAL] {str(e)}"
        add_log(job_id, error_msg)
        set_done(job_id, {"success": False, "error": error_msg})
    finally:
        sys.stdout = old_stdout