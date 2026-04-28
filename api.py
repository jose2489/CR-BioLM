from fastapi import FastAPI
from pydantic import BaseModel
from main import procesar_especie
import os

from fastapi.staticfiles import StaticFiles

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

app = FastAPI()

class Request(BaseModel):
    species: str
    question: str | None = None

@app.post("/run")
def run_pipeline(req: Request):
    result = procesar_especie(req.species, req.question)

    if not result["success"]:
        return result

    # Convertir rutas a URLs simples
    base_path = result["output_dir"]

    def rel(path):
        return path.replace(base_path, "")

    return {
        "success": True,
        "species": result["species"],
        "rf": result["rf_metrics"],
        "cnn": result["cnn_metrics"],
        "images": {
            "map": rel(result["images"]["map"]),
            "confusion": rel(result["images"]["confusion_matrix"]),
            "shap": rel(result["images"]["shap"]),
            "lime": rel(result["images"]["lime"]),
            "gradcam": rel(result["images"]["gradcam"]),
        }
    }