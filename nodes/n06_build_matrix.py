"""
N06: BuildMatrix
Construye la matriz ambiental (features + labels) para Machine Learning.
"""
import os
import json
from graph_state import GraphState
from data.geoprocessor import Geoprocessor
from sklearn.model_selection import train_test_split
import config

NUM_PSEUDOAUSENCIAS = 1500
TEST_SIZE = 0.2


def build_matrix_node(state: GraphState) -> GraphState:
    print("\n[N06:BuildMatrix] Construyendo matriz ambiental...")

    presencias_meso = state['presencias_meso']
    meso_bounds = state['meso_boundary']
    raster_paths = state['climate_rasters']

    geo = Geoprocessor()
    matriz_final = geo.build_environmental_matrix(
        presencias_meso,
        meso_bounds,
        raster_paths,
        ecoregions_gdf=None,
        use_extent_background=True,
        num_pseudoausencias=NUM_PSEUDOAUSENCIAS
    )

    X = matriz_final.drop(columns=['clase', 'lon', 'lat'])
    y = matriz_final['clase']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=config.SEED,
        stratify=y
    )

    state['environmental_matrix'] = matriz_final
    state['X_train'] = X_train
    state['X_test'] = X_test
    state['y_train'] = y_train
    state['y_test'] = y_test

    _save_json({
        "node": "N06_BuildMatrix",
        "total_filas": len(matriz_final),
        "features": list(X.columns),
        "num_features": X.shape[1],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "presencias_train": int(y_train.sum()),
        "ausencias_train": int((y_train == 0).sum()),
        "presencias_test": int(y_test.sum()),
        "ausencias_test": int((y_test == 0).sum()),
        "num_pseudoausencias": NUM_PSEUDOAUSENCIAS,
        "test_split": TEST_SIZE,
        "seed": config.SEED,
        "status": "ok"
    }, state['output_dir'], "n06_build_matrix.json")

    print(f"[N06:✓] Matriz creada: {len(X_train)} train | {len(X_test)} test")
    print(f"[N06:✓] Features: {X.shape[1]} variables")
    return state


def _save_json(data: dict, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
