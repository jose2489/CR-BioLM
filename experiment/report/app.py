#!/usr/bin/env python
# experiment/report/app.py
#
# Visor web del experimento CR-BioLM.
# Ejecutar: python experiment/report/app.py
# Abrir en: http://localhost:8000

import os
import sys
import json
import glob
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import uvicorn

RUNS_DIR   = os.path.join("experiment", "runs")
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES  = Jinja2Templates(directory=os.path.join(REPORT_DIR, "templates"))

app = FastAPI(title="CR-BioLM Experiment Viewer")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def image_to_base64(path):
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def read_txt(path):
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""


def extraer_respuesta(txt):
    marker = "[ANÁLISIS HÍBRIDO GENERADO POR IA]"
    if marker in txt:
        return txt.split(marker)[1].strip()
    return txt.strip()


def get_all_experiments():
    exps = []
    for exp_dir in sorted(glob.glob(os.path.join(RUNS_DIR, "EXP-*")), reverse=True):
        meta      = load_json(os.path.join(exp_dir, "experiment_meta.json"))
        eval_log  = load_json(os.path.join(exp_dir, "evaluation_log.json"))
        results   = None
        csv_path  = os.path.join(exp_dir, "results.csv")
        if os.path.exists(csv_path):
            results = pd.read_csv(csv_path)

        # Calcular scores medios por tier si hay resultados
        tier_scores = {}
        if results is not None and "score_compuesto" in results.columns:
            for tier in ["T1", "T2", "T3"]:
                subset = results[results["tier"] == tier]["score_compuesto"]
                tier_scores[tier] = round(subset.mean(), 3) if len(subset) > 0 else None

        # Contar especies completadas
        n_done = sum(1 for k, v in eval_log.items()
                     if k.endswith("|eval") and v.get("status") == "done")

        exps.append({
            "exp_id":      meta.get("exp_id", os.path.basename(exp_dir)),
            "persona":     meta.get("persona", "—"),
            "n_species":   meta.get("n_species", "?"),
            "started_at":  meta.get("started_at", "")[:10],
            "status":      meta.get("status", "unknown"),
            "notes":       meta.get("notes", ""),
            "tier_scores": tier_scores,
            "n_evaluated": n_done,
        })
    return exps


def get_experiment_detail(exp_id):
    exp_dir  = os.path.join(RUNS_DIR, exp_id)
    meta     = load_json(os.path.join(exp_dir, "experiment_meta.json"))
    exp_log  = load_json(os.path.join(exp_dir, "experiment_log.json"))
    csv_path = os.path.join(exp_dir, "results.csv")

    species_list = []
    results_df   = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

    for especie_dir in sorted(glob.glob(os.path.join(exp_dir, "*"))):
        if not os.path.isdir(especie_dir):
            continue
        especie = os.path.basename(especie_dir).replace("_", " ")
        perfil  = meta.get("persona", "botanico")

        pregunta_key = f"{especie}|pregunta|{perfil}"
        pregunta     = exp_log.get(pregunta_key, {}).get("pregunta", "—")

        # Scores por tier
        tier_scores = {}
        if results_df is not None:
            subset = results_df[results_df["especie"] == especie]
            for tier in ["T1", "T2", "T3"]:
                t = subset[subset["tier"] == tier]["score_compuesto"]
                tier_scores[tier] = round(t.mean(), 3) if len(t) > 0 else None

        # Tier winner
        valid = {k: v for k, v in tier_scores.items() if v is not None}
        winner = max(valid, key=valid.get) if valid else None

        species_list.append({
            "especie":     especie,
            "especie_id":  os.path.basename(especie_dir),
            "pregunta":    pregunta,
            "tier_scores": tier_scores,
            "winner":      winner,
        })

    # Resumen estadístico
    summary = {}
    if results_df is not None and "score_compuesto" in results_df.columns:
        for tier in ["T1", "T2", "T3"]:
            t = results_df[results_df["tier"] == tier]["score_compuesto"]
            summary[tier] = {
                "mean": round(t.mean(), 3) if len(t) > 0 else None,
                "std":  round(t.std(), 3)  if len(t) > 0 else None,
                "n":    len(t),
            }

    return {"meta": meta, "species": species_list, "summary": summary}


def get_species_detail(exp_id, especie_id):
    exp_dir     = os.path.join(RUNS_DIR, exp_id)
    especie_dir = os.path.join(exp_dir, especie_id)
    meta        = load_json(os.path.join(exp_dir, "experiment_meta.json"))
    exp_log     = load_json(os.path.join(exp_dir, "experiment_log.json"))
    perfil      = meta.get("persona", "botanico")
    especie     = especie_id.replace("_", " ")

    pregunta_key = f"{especie}|pregunta|{perfil}"
    pregunta     = exp_log.get(pregunta_key, {}).get("pregunta", "—")

    # Leer ficha MdP
    ficha = ""
    ficha_paths = glob.glob(os.path.join(especie_dir, "*_ficha_MdP.txt"))
    if not ficha_paths:
        ficha_paths = glob.glob(os.path.join(especie_dir, "*", "*_ficha_MdP.txt"))
    if ficha_paths:
        ficha = read_txt(ficha_paths[0])

    # Construir datos por tier
    tiers_data = {}
    for tier in ["T1", "T2", "T3"]:
        tier_dir = os.path.join(especie_dir, tier)
        if not os.path.isdir(tier_dir):
            continue

        # Mapas según tier
        maps = {}
        if tier == "T1":
            maps["meso"] = image_to_base64(
                os.path.join(tier_dir, "mapa_distribucion_mesoamerica.png"))
        else:
            maps["meso"] = image_to_base64(
                os.path.join(tier_dir, "mapa_distribucion_mesoamerica.png"))
            maps["habitat"] = image_to_base64(
                os.path.join(tier_dir, "mapa_habitat_manual.png"))
            if tier == "T3":
                # Try spatial overlap first, fall back to RF confusion map
                rf_path = os.path.join(tier_dir, "mapa_solapamiento_espacial.png")
                if not os.path.isfile(rf_path):
                    rf_path = os.path.join(tier_dir, "matriz_confusion.png")
                maps["rf"] = image_to_base64(rf_path)

        # Respuestas por modelo
        modelos = {}
        for modelo_file in glob.glob(os.path.join(tier_dir, "llm_profile_BIMODAL_*.txt")):
            modelo_key = os.path.basename(modelo_file).replace(
                "llm_profile_BIMODAL_", "").replace(".txt", "")
            modelo_label = modelo_key.replace("openai_gpt_4o", "GPT-4o").replace(
                "anthropic_claude_sonnet_4_5", "Claude Sonnet")
            raw      = read_txt(modelo_file)
            respuesta = extraer_respuesta(raw)
            modelos[modelo_label] = respuesta

        # Scores del juez
        scores = {}
        for eval_file in glob.glob(os.path.join(tier_dir, f"eval_{tier}_*.json")):
            eval_data   = load_json(eval_file)
            modelo_key  = os.path.basename(eval_file).split(f"eval_{tier}_")[1]
            modelo_key  = modelo_key.replace(f"_{perfil}.json", "")
            modelo_label = modelo_key.replace("openai_gpt_4o", "GPT-4o").replace(
                "anthropic_claude_sonnet_4_5", "Claude Sonnet")
            scores[modelo_label] = {
                "M1": eval_data.get("M1_precision_geografica"),
                "M2": eval_data.get("M2_precision_altitudinal"),
                "M3": eval_data.get("M3_relevancia_respuesta"),
                "M4": eval_data.get("M4_variable_climatica"),
                "score": eval_data.get("score_compuesto"),
                "r_M1":  eval_data.get("razonamiento_M1", ""),
                "r_M2":  eval_data.get("razonamiento_M2", ""),
                "r_M3":  eval_data.get("razonamiento_M3", ""),
                "r_M4":  eval_data.get("razonamiento_M4", ""),
            }

        tiers_data[tier] = {"maps": maps, "modelos": modelos, "scores": scores}

    return {
        "exp_id":   exp_id,
        "especie":  especie,
        "pregunta": pregunta,
        "perfil":   perfil,
        "ficha":    ficha,
        "tiers":    tiers_data,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    experiments = get_all_experiments()
    return TEMPLATES.TemplateResponse(request, "index.html", {
        "experiments": experiments,
    })


@app.get("/exp/{exp_id}", response_class=HTMLResponse)
async def experiment_detail(request: Request, exp_id: str):
    exp_dir = os.path.join(RUNS_DIR, exp_id)
    if not os.path.isdir(exp_dir):
        raise HTTPException(status_code=404, detail=f"Experimento no encontrado: {exp_id}")
    data = get_experiment_detail(exp_id)
    return TEMPLATES.TemplateResponse(request, "experiment.html", data)


@app.get("/exp/{exp_id}/species/{especie_id}", response_class=HTMLResponse)
async def species_detail(request: Request, exp_id: str, especie_id: str):
    data = get_species_detail(exp_id, especie_id)
    return TEMPLATES.TemplateResponse(request, "species.html", data)


@app.get("/api/results/{exp_id}")
async def api_results(exp_id: str):
    csv_path = os.path.join(RUNS_DIR, exp_id, "results.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404)
    df = pd.read_csv(csv_path)
    return JSONResponse(df.to_dict(orient="records"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    print(f"\nCR-BioLM Experiment Viewer")
    print(f"Abriendo en: http://localhost:{args.port}\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
