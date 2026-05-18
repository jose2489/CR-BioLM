"""
Estado compartido del grafo LangGraph para CR-BioLM.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import TypedDict, Optional, Dict, Any
import pandas as pd
import geopandas as gpd


class GraphState(TypedDict, total=False):

    # Inputs del usuario
    species_name: str
    user_question: Optional[str]
    output_dir: str

    # Grupo 1: Geometrías
    cr_boundary: Optional[gpd.GeoDataFrame]
    meso_boundary: Optional[gpd.GeoDataFrame]

    # Grupo 2: Ocurrencias
    presencias_meso: Optional[gpd.GeoDataFrame]
    presencias_cr: Optional[gpd.GeoDataFrame]

    # Grupo 3: Clima
    climate_rasters: Optional[Dict[str, str]]
    ecoregions_gdf: Optional[gpd.GeoDataFrame]

    # Grupo 4: Hábitat
    habitat_map_path: Optional[str]
    habitat_rf_map_path: Optional[str]
    texto_manual: Optional[str]

    # Grupo 5: Matriz de Modelado
    environmental_matrix: Optional[pd.DataFrame]
    X_train: Optional[pd.DataFrame]
    X_test: Optional[pd.DataFrame]
    y_train: Optional[pd.Series]
    y_test: Optional[pd.Series]

    # Grupo 6: Modelos y Predicciones
    rf_model: Optional[Any]
    predictions: Optional[Any]
    probabilities: Optional[Any]

    # Grupo 7: Métricas
    rf_metrics: Optional[Dict[str, Any]]

    # Grupo 8: Explicabilidad
    shap_data: Optional[Dict[str, Any]]
    lime_data: Optional[Dict[str, Any]]

    # Grupo 9: Contexto de Conservación
    info_altitud: Optional[str]
    contexto_conservacion: Optional[Dict[str, Any]]

    # Grupo 10: Reportes
    final_report_path: Optional[str]
    exported_maps: Optional[Dict[str, str]]

    # Control de flujo
    error_messages: list
    should_continue: bool
