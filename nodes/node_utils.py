"""
Utilidades compartidas para nodos del pipeline.
"""
import os
import json


def save_node_json(data: dict, output_dir: str, filename: str):
    """Guarda el output JSON de un nodo en output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)