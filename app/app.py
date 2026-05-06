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
import secrets
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException, Depends, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import uvicorn

import psycopg2.extras
from experiment.db import (init_db, get_conn, get_expert_progress, upsert_expert_session,
                           save_human_evaluation, get_first_unsubmitted,
                           get_raw_scores_for_kappa)

RUNS_DIR   = os.getenv("RUNS_DIR", os.path.join("experiment", "runs"))
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES  = Jinja2Templates(directory=os.path.join(REPORT_DIR, "templates"))

app = FastAPI(title="CR-BioLM Experiment Viewer")

# ── Auth ──────────────────────────────────────────────────────────────────────

security = HTTPBasic()

_CREDENTIALS = {
    os.getenv("EXPERT_1_USER", "expert1"): os.getenv("EXPERT_1_PASS", ""),
    os.getenv("EXPERT_2_USER", "expert2"): os.getenv("EXPERT_2_PASS", ""),
    os.getenv("ADMIN_USER",    "admin"):   os.getenv("ADMIN_PASS",    ""),
}
_ADMIN_USER = os.getenv("ADMIN_USER", "admin")

# Model A/B randomization seed (same for both experts — required for IRR calc)
_MODEL_SEED = {
    os.getenv("EXPERT_1_USER", "expert1"): ("openai/gpt-4o", "anthropic/claude-sonnet-4-5"),
    os.getenv("EXPERT_2_USER", "expert2"): ("openai/gpt-4o", "anthropic/claude-sonnet-4-5"),
}


def require_eval_auth(credentials: HTTPBasicCredentials = Depends(security)):
    user = credentials.username
    pwd  = credentials.password
    stored = _CREDENTIALS.get(user, "")
    ok = stored and secrets.compare_digest(pwd.encode(), stored.encode())
    if not ok:
        raise HTTPException(status_code=401, detail="Credenciales incorrectas",
                            headers={"WWW-Authenticate": "Basic"})
    return user


def require_admin(credentials: HTTPBasicCredentials = Depends(security)):
    user = require_eval_auth(credentials)
    if user != _ADMIN_USER:
        raise HTTPException(status_code=403, detail="Solo el administrador puede acceder aquí")
    return user


@app.on_event("startup")
def startup():
    init_db()


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
            for tier in ["T0", "T1", "T3"]:
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
        if perfil == "random":
            perfil = exp_log.get(f"{especie}|persona", perfil)

        pregunta_key = f"{especie}|pregunta|{perfil}"
        pregunta     = exp_log.get(pregunta_key, {}).get("pregunta", "—")

        # Scores por tier
        tier_scores = {}
        if results_df is not None:
            subset = results_df[results_df["especie"] == especie]
            for tier in ["T0", "T1", "T3"]:
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

    # Resumen estadístico por tier
    summary = {}
    if results_df is not None and "score_compuesto" in results_df.columns:
        for tier in ["T0", "T1", "T3"]:
            t = results_df[results_df["tier"] == tier]["score_compuesto"]
            summary[tier] = {
                "mean": round(t.mean(), 3) if len(t) > 0 else None,
                "std":  round(t.std(), 3)  if len(t) > 0 else None,
                "n":    len(t),
            }

    # Resumen por métrica (media sobre todos los tiers donde aplica)
    METRICS = [
        {"key": "M5_profundidad_analitica", "label": "M5 — Profundidad Analítica",
         "max": 3, "tiers": "T0/T1/T3",
         "desc": "Calidad del razonamiento, independiente de la exactitud factual. "
                 "Mide si el sistema integra fuentes, reconoce limitaciones y formula afirmaciones con cobertura epistémica apropiada. "
                 "En T0, una negativa bien calibrada puede obtener 3/3."},
        {"key": "M1_precision_geografica", "label": "M1 — Precisión Geográfica",
         "max": 3, "tiers": "T0/T1/T3",
         "desc": "¿Las zonas geográficas mencionadas (vertiente, cordillera, región, cantón) son consistentes con el Manual de Plantas? "
                 "En T0, evalúa si la negativa identifica correctamente que no puede determinar la distribución sin datos."},
        {"key": "M3_relevancia_respuesta", "label": "M3 — Relevancia de Respuesta",
         "max": 3, "tiers": "T0/T1/T3",
         "desc": "¿La respuesta contesta directamente la pregunta con información específica de la especie? "
                 "En T0, una negativa informativa que explica qué datos resolverían la pregunta puede obtener 2–3."},
        {"key": "M2_precision_altitudinal", "label": "M2 — Uso del Contexto Altitudinal",
         "max": 2, "tiers": "T3 únicamente (bonus)",
         "desc": "BONUS para T3. ¿El sistema usó responsablemente el rango altitudinal disponible "
                 "(GBIF detectado + Manual)? Penaliza solo si hay alucinación o ausencia total del dato. "
                 "N/A para T0/T1 — excluido del denominador."},
        {"key": "M4_variable_climatica", "label": "M4 — Variable Climática (SHAP)",
         "max": 2, "tiers": "T3 únicamente (bonus)",
         "desc": "BONUS para T3. ¿El sistema identificó correctamente la variable climática más limitante "
                 "según los valores SHAP del modelo RF? Incluye explicación mecanística. "
                 "N/A para T0/T1 — excluido del denominador."},
    ]
    metric_summary = []
    if results_df is not None:
        for m in METRICS:
            col = m["key"]
            if col in results_df.columns:
                numeric = pd.to_numeric(results_df[col], errors="coerce").dropna()
                m["mean"] = round(numeric.mean(), 3) if len(numeric) > 0 else None
                m["n"]    = len(numeric)
            else:
                m["mean"] = None
                m["n"]    = 0
            metric_summary.append(m)

    # Jueces usados — ensemble eval stores modelo_juez_A / modelo_juez_B
    judges = []
    if results_df is not None:
        for col in ("modelo_juez_A", "modelo_juez_B", "modelo_juez"):
            if col in results_df.columns:
                judges += results_df[col].dropna().unique().tolist()
        judges = sorted(set(judges))

    # Fórmula de score compuesto
    score_formula = (
        "T0/T1: (M1 + M3 + M5) / 9   |   "
        "T3 (sin bonus): (M1 + M3 + M5) / 9   |   "
        "T3 + M2: / 11   |   T3 + M4: / 11   |   T3 + M2 + M4: / 13   |   "
        "Caps: D1 taxonomy → ≤0.1 · M3≤1 → ≤0.2"
    )

    return {
        "meta": meta, "species": species_list, "summary": summary,
        "metric_summary": metric_summary, "judges": judges,
        "score_formula": score_formula,
    }


def get_species_detail(exp_id, especie_id):
    exp_dir     = os.path.join(RUNS_DIR, exp_id)
    especie_dir = os.path.join(exp_dir, especie_id)
    meta        = load_json(os.path.join(exp_dir, "experiment_meta.json"))
    exp_log     = load_json(os.path.join(exp_dir, "experiment_log.json"))
    perfil      = meta.get("persona", "botanico")
    especie     = especie_id.replace("_", " ")
    if perfil == "random":
        perfil = exp_log.get(f"{especie}|persona", perfil)

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
    for tier in ["T0", "T1", "T3"]:
        tier_dir = os.path.join(especie_dir, tier)
        if not os.path.isdir(tier_dir):
            continue

        # Mapas según tier (T0 no tiene imágenes relevantes al LLM, pero mostramos meso para contexto)
        maps = {}
        if tier != "T0":
            maps["meso"] = image_to_base64(
                os.path.join(tier_dir, "mapa_distribucion_mesoamerica.png"))
        if tier == "T3":
            maps["habitat"] = image_to_base64(
                os.path.join(tier_dir, "mapa_habitat_manual.png"))
        if tier == "T3":
            rf_path = os.path.join(tier_dir, "mapa_solapamiento_espacial.png")
            if not os.path.isfile(rf_path):
                rf_path = os.path.join(tier_dir, "matriz_confusion.png")
            maps["rf"]   = image_to_base64(rf_path)
            maps["shap"] = image_to_base64(os.path.join(tier_dir, "shap_summary.png"))
            maps["lime"] = image_to_base64(os.path.join(tier_dir, "lime_local_explanation.png"))

        # RF metrics (extracted from LLM profile header for T3)
        rf_metrics = {}
        if tier == "T3":
            # Try dedicated rf_metrics.json first (written by newer pipeline runs)
            rf_json = load_json(os.path.join(tier_dir, "rf_metrics.json"))
            if rf_json:
                rf_metrics = rf_json
            else:
                # Fall back: parse LLM profile header for AUC
                for llm_file in glob.glob(os.path.join(tier_dir, "llm_profile_BIMODAL_*.txt")):
                    raw = read_txt(llm_file)
                    for line in raw.splitlines():
                        if "AUC:" in line:
                            try:
                                rf_metrics["auc"] = float(
                                    line.split("AUC:")[1].strip().split(")")[0])
                            except Exception:
                                pass
                        if "Factor limitante" in line and ":" in line:
                            rf_metrics["factor_limitante"] = line.split(":")[-1].strip()
                        if "Altitud" in line and "detectada" in line and ":" in line:
                            rf_metrics["altitud_rango"] = line.split(":", 1)[-1].strip()
                    if rf_metrics:
                        break

        # Respuestas por modelo
        modelos = {}
        for modelo_file in glob.glob(os.path.join(tier_dir, "llm_profile_BIMODAL_*.txt")):
            modelo_key = os.path.basename(modelo_file).replace(
                "llm_profile_BIMODAL_", "").replace(".txt", "")
            modelo_label = modelo_key.replace("openai_gpt_4o", "GPT-4o").replace(
                "anthropic_claude_sonnet_4_5", "Claude Sonnet")
            raw       = read_txt(modelo_file)
            respuesta = extraer_respuesta(raw)
            modelos[modelo_label] = respuesta

        # Scores del juez — supports both legacy (single-judge) and ensemble format
        scores = {}
        for eval_file in glob.glob(os.path.join(tier_dir, f"eval_{tier}_*.json")):
            eval_data    = load_json(eval_file)
            modelo_key   = os.path.basename(eval_file).split(f"eval_{tier}_")[1]
            modelo_key   = modelo_key.replace(f"_{perfil}.json", "")
            modelo_label = modelo_key.replace("openai_gpt_4o", "GPT-4o").replace(
                "anthropic_claude_sonnet_4_5", "Claude Sonnet")
            # Ensemble: use aggregate scores; legacy: use top-level scores directly
            agg = eval_data.get("judge_aggregate") or eval_data
            scores[modelo_label] = {
                "M5": agg.get("M5_profundidad_analitica"),
                "M1": agg.get("M1_precision_geografica"),
                "M3": agg.get("M3_relevancia_respuesta"),
                "M2": agg.get("M2_precision_altitudinal"),
                "M4": agg.get("M4_variable_climatica"),
                "score":        eval_data.get("score_compuesto"),
                "disagree_flag": eval_data.get("disagree_flag", False),
                "taxonomy_valid": eval_data.get("taxonomy_valid", True),
                # Rationale from judge A (ensemble) or direct (legacy)
                "cita_M5": (eval_data.get("judge_A_scores") or agg).get("cita_M5", ""),
                "r_M5":    (eval_data.get("judge_A_scores") or agg).get("razonamiento_M5", ""),
                "cita_M1": (eval_data.get("judge_A_scores") or agg).get("cita_M1", ""),
                "r_M1":    (eval_data.get("judge_A_scores") or agg).get("razonamiento_M1", ""),
                "cita_M3": (eval_data.get("judge_A_scores") or agg).get("cita_M3", ""),
                "r_M3":    (eval_data.get("judge_A_scores") or agg).get("razonamiento_M3", ""),
                "r_M2":    (eval_data.get("judge_A_scores") or agg).get("razonamiento_M2", ""),
                "r_M4":    (eval_data.get("judge_A_scores") or agg).get("razonamiento_M4", ""),
            }

        tiers_data[tier] = {
            "maps": maps, "modelos": modelos,
            "scores": scores, "rf_metrics": rf_metrics,
        }

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


PROMPTS_OVERRIDE_PATH = os.getenv("PROMPTS_OVERRIDE_PATH", os.path.join("llm", "prompts_override.json"))

PROMPT_META = {
    "T0": {
        "label": "T0 — Baseline Paramétrico",
        "color": "gray",
        "desc": "Sin contexto adicional. El LLM responde desde conocimiento de entrenamiento únicamente. Sin imágenes.",
        "placeholders": ["{species_name}", "{instruccion_pregunta}"],
        "images": [],
    },
    "T1": {
        "label": "T1 — Baseline GBIF",
        "color": "slate",
        "desc": "Solo mapa Mesoamérica con puntos de presencia GBIF (Imagen 1). Sin métricas RF ni altitud.",
        "placeholders": ["{species_name}", "{instruccion_pregunta}", "{_regla}"],
        "images": ["Imagen 1: Mapa GBIF Mesoamérica"],
    },
    "T3": {
        "label": "T3 — Sistema Completo",
        "color": "emerald",
        "desc": "Mapa de hábitat predicho (Imagen 1) + mapa RF predictivo (Imagen 2) + métricas SHAP/AUC + rango altitudinal.",
        "placeholders": ["{species_name}", "{rf_auc:.4f}", "{info_altitud}", "{var_humana}",
                         "{direccion}", "{zona_humana}", "{secundaria_1}", "{secundaria_2}",
                         "{instruccion_pregunta}", "{_regla}"],
        "images": ["Imagen 1: Mapa hábitat predicho (Manual + Hammel + DEM)", "Imagen 2: Mapa RF predictivo"],
    },
    "_REGLA_STRICTA": {
        "label": "Regla Estricta Compartida",
        "color": "amber",
        "desc": "Bloque de reglas inyectado en T1 y T3 vía {_regla}. Controla el comportamiento del LLM.",
        "placeholders": [],
        "images": [],
    },
}


def _load_prompts():
    from llm.prompt_templates import get_effective_prompts
    return get_effective_prompts()


def _load_overrides():
    if os.path.isfile(PROMPTS_OVERRIDE_PATH):
        with open(PROMPTS_OVERRIDE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_overrides(data: dict):
    with open(PROMPTS_OVERRIDE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.get("/prompts", response_class=HTMLResponse)
async def prompts_page(request: Request):
    effective = _load_prompts()
    overrides = _load_overrides()
    from llm.prompt_templates import _DEFAULTS
    tiers = []
    for key in ["T0", "T1", "T3", "_REGLA_STRICTA"]:
        tiers.append({
            "key": key,
            "meta": PROMPT_META[key],
            "text": effective.get(key, ""),
            "default": _DEFAULTS.get(key, ""),
            "is_overridden": key in overrides,
        })
    return TEMPLATES.TemplateResponse(request, "prompts.html", {"tiers": tiers})


class PromptSaveRequest(BaseModel):
    key: str
    text: str


@app.post("/api/prompts/save")
async def save_prompt(body: PromptSaveRequest):
    from llm.prompt_templates import _DEFAULTS
    if body.key not in _DEFAULTS:
        raise HTTPException(status_code=400, detail=f"Unknown prompt key: {body.key}")
    overrides = _load_overrides()
    overrides[body.key] = body.text
    _save_overrides(overrides)
    return {"ok": True, "key": body.key, "overridden": True}


@app.post("/api/prompts/reset/{key}")
async def reset_prompt(key: str):
    from llm.prompt_templates import _DEFAULTS
    if key not in _DEFAULTS:
        raise HTTPException(status_code=400, detail=f"Unknown prompt key: {key}")
    overrides = _load_overrides()
    overrides.pop(key, None)
    _save_overrides(overrides)
    return {"ok": True, "key": key, "text": _DEFAULTS[key]}


@app.get("/questions", response_class=HTMLResponse)
async def questions(request: Request):
    from utils.question_bank import QUESTION_BANK
    personas = []
    for persona, entries in QUESTION_BANK.items():
        n_a = sum(1 for e in entries if e.get("stratum") == "A")
        n_c = sum(1 for e in entries if e.get("stratum") == "C")
        personas.append({
            "name": persona,
            "label": {"botanico": "Botánico / Ecólogo", "turista": "Turista / Naturalista"}.get(persona, persona),
            "entries": entries,
            "n_a": n_a,
            "n_c": n_c,
        })
    total = sum(len(p["entries"]) for p in personas)
    return TEMPLATES.TemplateResponse(request, "questions.html", {
        "personas": personas,
        "total": total,
    })


# ── Expert evaluation routes (Phase A — GET only) ─────────────────────────────

@app.get("/eval", response_class=HTMLResponse)
@app.get("/eval/", response_class=HTMLResponse)
async def eval_index(request: Request, user: str = Depends(require_eval_auth)):
    experiments = get_all_experiments()
    progress    = {}
    for exp in experiments:
        progress[exp["exp_id"]] = get_expert_progress(exp["exp_id"]).get(user, {})

    # Ensure expert_session exists (creates model A/B mapping on first visit)
    model_A, model_B = _MODEL_SEED.get(user, ("openai/gpt-4o", "anthropic/claude-sonnet-4-5"))
    upsert_expert_session(user, model_A, model_B)

    # Attach species list to each experiment for the index view
    for exp in experiments:
        exp["species_list"] = _get_species_list(exp["exp_id"])

    # B9 — auto-advance: if expert has started, redirect to first unsubmitted species
    if not request.query_params.get("index") and experiments:
        for exp in experiments:
            all_sp = exp["species_list"]
            user_progress = progress.get(exp["exp_id"], {})
            if user_progress:  # has started this experiment
                nxt = get_first_unsubmitted(exp["exp_id"], user, all_sp)
                if nxt:
                    return RedirectResponse(
                        f"/eval/{exp['exp_id']}/{nxt['especie_id']}", status_code=302)

    return TEMPLATES.TemplateResponse(request, "eval_index.html", {
        "user":        user,
        "is_admin":    user == _ADMIN_USER,
        "experiments": experiments,
        "progress":    progress,
    })


@app.get("/eval/results/{exp_id}", response_class=HTMLResponse)
async def eval_results(request: Request, exp_id: str,
                       user: str = Depends(require_admin)):
    from experiment.db import human_vs_llm_agreement, flag_review_candidates
    from sklearn.metrics import cohen_kappa_score

    data       = get_experiment_detail(exp_id)
    agreement  = human_vs_llm_agreement(exp_id)
    candidates = flag_review_candidates(exp_id)
    raw        = get_raw_scores_for_kappa(exp_id)

    kappa = {}
    METRIC_LABELS = {"M1": "Precisión Geográfica", "M3": "Relevancia",
                     "M5": "Profundidad Analítica", "M2": "Altitudinal (T3)", "M4": "Climática (T3)"}
    for m, vecs in raw.items():
        h, l = vecs["human"], vecs["llm"]
        if len(h) >= 2:
            try:
                kappa[m] = {
                    "label": METRIC_LABELS[m],
                    "kappa": round(cohen_kappa_score(h, l, weights="linear"), 3),
                    "n":     len(h),
                }
            except Exception:
                kappa[m] = {"label": METRIC_LABELS[m], "kappa": None, "n": len(h)}
        else:
            kappa[m] = {"label": METRIC_LABELS[m], "kappa": None, "n": len(h)}

    return TEMPLATES.TemplateResponse(request, "eval_results.html", {
        **data,
        "exp_id":      exp_id,
        "user":        user,
        "agreement":   agreement,
        "candidates":  candidates,
        "kappa":       kappa,
    })


@app.get("/eval/export/{exp_id}")
async def eval_export(exp_id: str, user: str = Depends(require_admin)):
    from experiment.db import export_for_thesis
    rows = export_for_thesis(exp_id)
    if not rows:
        raise HTTPException(status_code=404, detail="No hay evaluaciones humanas para exportar")

    import io, csv
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue().encode("utf-8-sig")]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=eval_{exp_id}.csv"},
    )


@app.get("/eval/{exp_id}/{especie_id}", response_class=HTMLResponse)
async def eval_form(request: Request, exp_id: str, especie_id: str,
                    user: str = Depends(require_eval_auth)):
    exp_dir = os.path.join(RUNS_DIR, exp_id)
    if not os.path.isdir(exp_dir):
        raise HTTPException(status_code=404, detail=f"Experimento no encontrado: {exp_id}")

    data = get_species_detail(exp_id, especie_id)

    # Determine Modelo A / Modelo B mapping for this user
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT model_A, model_B FROM expert_sessions WHERE username = %s", (user,))
        session = cur.fetchone()
    model_A = session["model_A"] if session else "openai/gpt-4o"
    model_B = session["model_B"] if session else "anthropic/claude-sonnet-4-5"

    # Build ordered species list for prev/next navigation
    all_species = _get_species_list(exp_id)
    idx = next((i for i, s in enumerate(all_species) if s["especie_id"] == especie_id), 0)
    prev_s = all_species[idx - 1] if idx > 0 else None
    next_s = all_species[idx + 1] if idx < len(all_species) - 1 else None

    # Already submitted tiers for this user
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT tier FROM human_evaluations
            WHERE exp_id=%s AND especie=%s AND evaluator=%s
        """, (exp_id, data["especie"], user))
        submitted = cur.fetchall()
    submitted_tiers = {r["tier"] for r in submitted}

    return TEMPLATES.TemplateResponse(request, "eval_form.html", {
        **data,
        "user":            user,
        "is_admin":        user == _ADMIN_USER,
        "exp_id":          exp_id,
        "especie_id":      especie_id,
        "model_A":         model_A,
        "model_B":         model_B,
        "submitted_tiers": submitted_tiers,
        "prev_species":    prev_s,
        "next_species":    next_s,
        "species_index":   idx + 1,
        "species_total":   len(all_species),
    })


@app.post("/eval/{exp_id}/{especie_id}")
async def eval_submit(request: Request, exp_id: str, especie_id: str,
                      user: str = Depends(require_eval_auth)):
    form = await request.form()
    tier = form.get("submit_tier")
    if tier not in ("T0", "T1", "T3"):
        raise HTTPException(status_code=400, detail="Tier inválido")

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT model_A, model_B FROM expert_sessions WHERE username = %s", (user,))
        session = cur.fetchone()
    model_A = session["model_A"] if session else "openai/gpt-4o"
    model_B = session["model_B"] if session else "anthropic/claude-sonnet-4-5"

    especie = especie_id.replace("_", " ")
    comment = form.get(f"{tier}_comment") or None

    def _likert(name):
        v = form.get(name)
        if v is None or v == "NA" or v == "":
            return None
        try:
            return int(v)
        except ValueError:
            return None

    for model_key, real_model in [("A", model_A), ("B", model_B)]:
        M1 = _likert(f"{tier}_{model_key}_M1")
        M3 = _likert(f"{tier}_{model_key}_M3")
        M5 = _likert(f"{tier}_{model_key}_M5")
        M2 = _likert(f"{tier}_{model_key}_M2") if tier == "T3" else None
        M4 = _likert(f"{tier}_{model_key}_M4") if tier == "T3" else None

        save_human_evaluation(
            exp_id=exp_id, especie=especie, tier=tier,
            modelo_generador=real_model, evaluator=user,
            M1=M1, M2=M2, M3=M3, M4=M4, M5=M5, comment=comment,
        )

    all_sp = _get_species_list(exp_id)
    nxt = get_first_unsubmitted(exp_id, user, all_sp)
    if nxt:
        return RedirectResponse(f"/eval/{exp_id}/{nxt['especie_id']}", status_code=302)
    return RedirectResponse("/eval?index=1", status_code=302)


def _get_species_list(exp_id):
    """Returns sorted list of {especie, especie_id} for the given experiment."""
    exp_dir = os.path.join(RUNS_DIR, exp_id)
    species = []
    for d in sorted(glob.glob(os.path.join(exp_dir, "*"))):
        if os.path.isdir(d):
            especie_id = os.path.basename(d)
            species.append({"especie": especie_id.replace("_", " "),
                            "especie_id": especie_id})
    return species


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
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    print(f"\nCR-BioLM Experiment Viewer")
    print(f"Abriendo en: http://localhost:{args.port}\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
