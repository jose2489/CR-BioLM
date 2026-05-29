"""
Configuración central de modelos LLM para CR-BioLM.
Modificar LLM_MODEL y LLM_IS_MULTIMODAL para cambiar el modelo usado en todo el pipeline.
"""

# ==========================================
# MODELO ACTIVO
# ==========================================
LLM_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

# ==========================================
# ¿EL MODELO SOPORTA IMÁGENES?
# True  → adjunta imágenes al prompt (multimodal)
# False → solo texto (monomodal)
# ==========================================
LLM_IS_MULTIMODAL = True


def is_multimodal(model_name: str = None) -> bool:
    """Retorna True si el modelo activo soporta imágenes."""
    return LLM_IS_MULTIMODAL