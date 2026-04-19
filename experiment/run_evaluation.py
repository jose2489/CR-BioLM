#!/usr/bin/env python
# experiment/run_evaluation.py
#
# Evaluador automático CR-BioLM con Gemini-as-Judge.
# Opera sobre un experimento específico (exp_id) o sobre todos los existentes.
# Guarda resultados en experiment/runs/{exp_id}/results.csv y evaluation_log.json
#
# Uso:
#   python experiment/run_evaluation.py --exp-id EXP-20260416-001-botanico
#   python experiment/run_evaluation.py --exp-id EXP-20260416-001-botanico --resume
#   python experiment/run_evaluation.py --all        # evalúa todos los experimentos
#   python experiment/run_evaluation.py --dry-run --exp-id EXP-20260416-001-botanico

import os
import sys
import json
import argparse
import datetime
import glob

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from llm.judge_client import JudgeClient, ficha_summary

RUNS_DIR = os.path.join("experiment", "runs")
TIERS    = ["T0", "T1", "T2", "T3"]
MODELOS  = ["openai_gpt_4o", "anthropic_claude_sonnet_4_5"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def eval_key(especie: str, tier: str, modelo: str, perfil: str) -> str:
    return f"{especie}|{tier}|{modelo}|{perfil}|eval"


def extraer_respuesta(ruta_txt: str) -> str:
    with open(ruta_txt, encoding="utf-8") as f:
        contenido = f.read()
    marker = "[ANÁLISIS HÍBRIDO GENERADO POR IA]"
    if marker in contenido:
        return contenido.split(marker)[1].strip()
    return contenido.strip()


def extraer_pregunta_del_perfil(ruta_txt: str) -> str:
    with open(ruta_txt, encoding="utf-8") as f:
        for line in f:
            if line.startswith("Pregunta Usuario"):
                return line.split(":", 1)[1].strip()
    return ""


def encontrar_archivos_por_tier(especie_dir: str) -> dict:
    resultado = {}
    for tier in TIERS:
        tier_dir = os.path.join(especie_dir, tier)
        if not os.path.isdir(tier_dir):
            continue
        resultado[tier] = {}
        for modelo in MODELOS:
            ruta = os.path.join(tier_dir, f"llm_profile_BIMODAL_{modelo}.txt")
            if os.path.isfile(ruta):
                resultado[tier][modelo] = ruta
    return resultado


# ── Evaluación de un experimento ──────────────────────────────────────────────

def evaluar_experimento(exp_dir: str, judge: JudgeClient,
                        resume: bool = False, dry_run: bool = False,
                        solo_especie: str = None) -> list[dict]:
    exp_id    = os.path.basename(exp_dir)
    eval_log  = load_json(os.path.join(exp_dir, "evaluation_log.json")) if resume else {}
    exp_log   = load_json(os.path.join(exp_dir, "experiment_log.json"))
    meta      = load_json(os.path.join(exp_dir, "experiment_meta.json"))
    perfil    = meta.get("persona", "botanico")

    all_rows      = []
    total_eval    = 0
    total_skip    = 0
    total_fail    = 0

    # Descubrir especies en este experimento
    if solo_especie:
        especie_dirs = [os.path.join(exp_dir, solo_especie.replace(" ", "_"))]
    else:
        especie_dirs = sorted([
            d for d in glob.glob(os.path.join(exp_dir, "*"))
            if os.path.isdir(d) and not os.path.basename(d).startswith(".")
        ])

    for especie_dir in especie_dirs:
        especie = os.path.basename(especie_dir).replace("_", " ")

        # Leer ficha MdP (ground truth)
        ficha_paths = glob.glob(os.path.join(especie_dir, "*_ficha_MdP.txt"))
        # También buscar dentro de T1/T2/T3
        if not ficha_paths:
            ficha_paths = glob.glob(os.path.join(especie_dir, "*", "*_ficha_MdP.txt"))
        if not ficha_paths:
            print(f"  [SKIP] Sin ficha MdP: {especie}")
            continue

        with open(ficha_paths[0], encoding="utf-8") as f:
            ficha_mdp = ficha_summary(f.read())  # truncated for judge (geo + alt + habitat only)

        # Recuperar pregunta del log del experimento
        pregunta_key = f"{especie}|pregunta|{perfil}"
        pregunta = exp_log.get(pregunta_key, {}).get("pregunta", "")

        archivos = encontrar_archivos_por_tier(especie_dir)
        if not archivos:
            print(f"  [SKIP] Sin archivos LLM: {especie}")
            continue

        print(f"\n  [ESPECIE] {especie}")

        for tier, modelos_dict in archivos.items():
            for modelo_limpio, ruta_txt in modelos_dict.items():
                key = eval_key(especie, tier, modelo_limpio, perfil)

                if resume and key in eval_log and eval_log[key].get("status") == "done":
                    print(f"    [SKIP] {tier} | {modelo_limpio.split('_')[1]}")
                    all_rows.append(eval_log[key]["scores"])
                    total_skip += 1
                    continue

                if dry_run:
                    print(f"    [DRY]  {tier} | {modelo_limpio} → evaluaría")
                    continue

                respuesta         = extraer_respuesta(ruta_txt)
                pregunta_efectiva = pregunta or extraer_pregunta_del_perfil(ruta_txt)

                print(f"    → {tier} | {modelo_limpio.split('_')[1]}...", end=" ", flush=True)

                scores = judge.evaluar(
                    pregunta=pregunta_efectiva,
                    respuesta=respuesta,
                    ficha_mdp=ficha_mdp,
                    perfil=perfil,
                    especie=especie,
                    tier=tier,
                    modelo_generador=modelo_limpio,
                    output_dir=os.path.join(especie_dir, tier),
                )

                if scores:
                    scores["exp_id"]  = exp_id
                    scores["perfil"]  = perfil
                    eval_log[key] = {
                        "status":    "done",
                        "scores":    scores,
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                    all_rows.append(scores)
                    total_eval += 1
                    print(f"score={scores['score_compuesto']:.3f} "
                          f"M5={scores.get('M5_profundidad_analitica')} "
                          f"M1={scores.get('M1_precision_geografica')} "
                          f"M3={scores.get('M3_relevancia_respuesta')} "
                          f"M2={scores.get('M2_precision_altitudinal')} "
                          f"M4={scores.get('M4_variable_climatica')}")
                else:
                    eval_log[key] = {
                        "status":    "failed",
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                    total_fail += 1
                    print("[FAIL]")

                save_json(eval_log, os.path.join(exp_dir, "evaluation_log.json"))

    # ── Guardar results.csv dentro del experimento ──
    if all_rows and not dry_run:
        df = pd.DataFrame(all_rows)
        cols = ["exp_id", "especie", "tier", "modelo_generador", "perfil", "modelo_juez",
                "M5_profundidad_analitica", "M1_precision_geografica", "M3_relevancia_respuesta",
                "M2_precision_altitudinal", "M4_variable_climatica", "score_compuesto",
                "cita_M5", "razonamiento_M5", "cita_M1", "razonamiento_M1",
                "cita_M3", "razonamiento_M3", "razonamiento_M2", "razonamiento_M4"]
        df = df[[c for c in cols if c in df.columns]]
        results_csv = os.path.join(exp_dir, "results.csv")
        df.to_csv(results_csv, index=False, encoding="utf-8")
        print(f"\n  [CSV] {results_csv}")

        # Resumen por tier × modelo
        if "score_compuesto" in df.columns:
            resumen = (df.groupby(["tier", "modelo_generador"])["score_compuesto"]
                       .agg(["mean", "std", "count"]).round(3))
            print(resumen.to_string())

    print(f"\n  [FIN] {exp_id}: eval={total_eval} skip={total_skip} fail={total_fail}")
    return all_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluador CR-BioLM con Gemini-as-Judge")
    parser.add_argument("--exp-id",  type=str, default=None,
                        help="ID del experimento a evaluar (ej: EXP-20260416-001-botanico)")
    parser.add_argument("--all",     action="store_true",
                        help="Evalúa todos los experimentos en experiment/runs/")
    parser.add_argument("--species", type=str, default=None,
                        help="Evalúa solo esta especie dentro del experimento")
    parser.add_argument("--resume",  action="store_true",
                        help="Salta evaluaciones ya completadas")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra qué evaluaría sin llamar al juez")
    args = parser.parse_args()

    if not args.exp_id and not args.all:
        parser.error("Debes especificar --exp-id o --all")

    judge = JudgeClient(api_key=config.OPENROUTER_API_KEY)

    if args.all:
        exp_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, "EXP-*")))
        print(f"[INFO] Evaluando {len(exp_dirs)} experimentos\n")
    else:
        exp_dirs = [os.path.join(RUNS_DIR, args.exp_id)]

    for exp_dir in exp_dirs:
        if not os.path.isdir(exp_dir):
            print(f"[ERROR] No existe: {exp_dir}")
            continue
        exp_id = os.path.basename(exp_dir)
        print(f"\n{'='*60}")
        print(f"[EXPERIMENTO] {exp_id}")
        print(f"{'='*60}")
        evaluar_experimento(
            exp_dir=exp_dir,
            judge=judge,
            resume=args.resume,
            dry_run=args.dry_run,
            solo_especie=args.species,
        )


if __name__ == "__main__":
    main()
