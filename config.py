import os
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# ── 1. REPRODUCIBILIDAD ACADÉMICA ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── 2. RUTAS BASE DEL PROYECTO ────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data_raw")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ── 3. VARIABLES DE ENTORNO Y APIs ────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print(f"[CONFIG] OPENROUTER_API_KEY loaded: {bool(OPENROUTER_API_KEY)}")

# ── 4. PARÁMETROS DEL PIPELINE ────────────────────────────────────────────────

# GBIF
GBIF_LIMIT_CR   = 800    # máximo de registros para Costa Rica
GBIF_LIMIT_MESO = 3000   # máximo de registros para Mesoamérica

# Construcción de matriz ambiental
NUM_PSEUDOAUSENCIAS = 1500
TEST_SIZE           = 0.2  # fracción del dataset para prueba (80/20)

# Random Forest
RF_N_ESTIMATORS = 100

# ── 5. CONFIGURACIÓN GENERAL ──────────────────────────────────────────────────
DEFAULT_COUNTRY = "Costa Rica"
DEFAULT_LLM     = "openrouter/free"

# ── 6. GESTOR DE DIRECTORIOS DE EJECUCIÓN ────────────────────────────────────
def crear_directorio_ejecucion(especie_nombre: str) -> str:
    """
    Crea y retorna la ruta de una carpeta única para la ejecución actual,
    basada en la especie y la fecha/hora exacta.

    Parámetros:
        especie_nombre: nombre científico de la especie

    Retorna:
        Ruta al directorio outputs/{Especie}/run_{timestamp}/
    """
    especie_formateada = especie_nombre.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_ejecucion = os.path.join(
        OUTPUT_BASE_DIR, especie_formateada, f"run_{timestamp}"
    )
    os.makedirs(ruta_ejecucion, exist_ok=True)
    return ruta_ejecucion