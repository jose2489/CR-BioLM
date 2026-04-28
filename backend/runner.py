import sys
import os
import io
import numpy as np
from job_store import add_log, set_done

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from main import procesar_especie

class Logger(io.StringIO):
    def __init__(self, job_id):
        super().__init__()
        self.job_id = job_id
    def write(self, message):
        if message.strip():
            add_log(self.job_id, message.strip())
        super().write(message)

def make_serializable(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj

def run_pipeline(job_id, species, question):
    logger = Logger(job_id)
    old_stdout = sys.stdout
    sys.stdout = logger
    try:
        add_log(job_id, "[INFO] Iniciando pipeline")
        result = procesar_especie(species, question)
        print("RESULT FROM PIPELINE:", result)
        add_log(job_id, "[INFO] Pipeline finalizado")

        if not result or not isinstance(result, dict):
            set_done(job_id, {"error": "Pipeline no retornó resultados"})
        elif result.get("success") is False:
            set_done(job_id, make_serializable(result))
        else:
            set_done(job_id, make_serializable(result))
    except Exception as e:
        error_msg = f"[ERROR FATAL] {str(e)}"
        add_log(job_id, error_msg)
        set_done(job_id, {"error": error_msg})
    finally:
        sys.stdout = old_stdout