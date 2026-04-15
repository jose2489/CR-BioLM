# utils/translator.py

NOMBRES_VARIABLES = {
    "bio_1": "bio_1: Temp. Media Anual",
    "bio_2": "bio_2: Rango Diurno Medio",
    "bio_3": "bio_3: Isotermalidad",
    "bio_4": "bio_4: Estacionalidad Temp.",
    "bio_5": "bio_5: Temp. Max. Mes Cálido",
    "bio_6": "bio_6: Temp. Min. Mes Frío",
    "bio_7": "bio_7: Rango Anual Temp.",
    "bio_8": "bio_8: Temp. Media Trim. Húmedo",
    "bio_9": "bio_9: Temp. Media Trim. Seco",
    "bio_10": "bio_10: Temp. Media Trim. Cálido",
    "bio_11": "bio_11: Temp. Media Trim. Frío",
    "bio_12": "bio_12: Precipitación Anual",
    "bio_13": "bio_13: Precip. Mes Húmedo",
    "bio_14": "bio_14: Precip. Mes Seco",
    "bio_15": "bio_15: Estacionalidad Precip.",
    "bio_16": "bio_16: Precip. Trim. Húmedo",
    "bio_17": "bio_17: Precip. Trim. Seco",
    "bio_18": "bio_18: Precip. Trim. Cálido",
    "bio_19": "bio_19: Precip. Trim. Frío"
}

def traducir_variable(var_tecnica):
    """Convierte un código técnico a un nombre legible."""
    if not var_tecnica:
        return "variables ambientales"
        
    if var_tecnica in NOMBRES_VARIABLES:
        return NOMBRES_VARIABLES[var_tecnica]
    
    if var_tecnica.startswith("Eco_"):
        # Mantiene el indicador de que es una zona espacial
        return var_tecnica.replace("Eco_", "Zona: ").replace("_", " ").title()
    
    return var_tecnica

def traducir_lista_variables(lista_tecnica):
    """Traduce una lista completa de variables (ideal para Pandas y LIME)."""
    return [traducir_variable(var) for var in lista_tecnica]