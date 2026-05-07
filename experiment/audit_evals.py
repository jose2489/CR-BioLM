#!/usr/bin/env python
# experiment/audit_evals.py
#
# Audit expected vs. actual evaluations for a CR-BioLM experiment run.
# Discovers which (especie, tier, modelo) combos are done/failed/missing,
# and optionally resubmits missing/failed ones.
#
# Usage:
#   python experiment/audit_evals.py --exp-id EXP-...
#   python experiment/audit_evals.py --exp-id EXP-... --resubmit
#   python experiment/audit_evals.py --exp-id EXP-... --resubmit --species "Sp name"
#   python experiment/audit_evals.py --exp-id EXP-... --dry-run

import os
import sys
import json
import glob
import argparse
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

RUNS_DIR = os.path.join("experiment", "runs")
TIERS    = ["T0", "T1", "T3"]
MODELOS  = ["openai_gpt_4o", "anthropic_claude_sonnet_4_5"]

try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_expected_combos(exp_dir):
    """
    Walk exp_dir looking for llm_profile_BIMODAL_*.txt files.
    Returns list of dicts: {especie, especie_dir, tier, modelo, perfil, txt_path}
    """
    meta     = _load_json(os.path.join(exp_dir, "experiment_meta.json"))
    exp_log  = _load_json(os.path.join(exp_dir, "experiment_log.json"))
    base_perfil = meta.get("persona", "botanico")

    combos = []
    for especie_dir in sorted(glob.glob(os.path.join(exp_dir, "*"))):
        if not os.path.isdir(especie_dir):
            continue
        basename = os.path.basename(especie_dir)
        if basename.startswith(".") or basename == "_archive_pre_cleanup":
            continue
        especie = basename.replace("_", " ")

        # perfil_key: matches what run_evaluation.py stores in eval_key (always meta.persona)
        # perfil_judge: per-species resolved perfil for calling judge.evaluar() during resubmit
        perfil_key = base_perfil
        if base_perfil == "random":
            perfil_judge = exp_log.get(f"{especie}|persona", "botanico")
        else:
            perfil_judge = base_perfil

        for tier in TIERS:
            tier_dir = os.path.join(especie_dir, tier)
            if not os.path.isdir(tier_dir):
                continue
            for modelo in MODELOS:
                txt_path = os.path.join(tier_dir, f"llm_profile_BIMODAL_{modelo}.txt")
                if os.path.isfile(txt_path):
                    combos.append({
                        "especie":      especie,
                        "especie_dir":  especie_dir,
                        "tier":         tier,
                        "modelo":       modelo,
                        "perfil":       perfil_key,    # used for key lookup
                        "perfil_judge": perfil_judge,  # used when calling judge
                        "txt_path":     txt_path,
                    })
    return combos


def classify_combos(exp_dir, combos):
    """
    Returns list of combos with 'status' field: done/failed/missing.
    """
    eval_log = _load_json(os.path.join(exp_dir, "evaluation_log.json"))
    result   = []
    for c in combos:
        key    = f"{c['especie']}|{c['tier']}|{c['modelo']}|{c['perfil']}|eval"
        entry  = eval_log.get(key)
        status = entry["status"] if entry else "missing"
        result.append({**c, "status": status})
    return result


# ── Display ───────────────────────────────────────────────────────────────────

def print_status_table(exp_id, classified):
    print(f"\nExperiment: {exp_id}")

    # Build rows: especie × tier → {modelo: status}
    rows = {}
    for c in classified:
        row_key = (c["especie"], c["tier"])
        rows.setdefault(row_key, {})[c["modelo"]] = c["status"]

    headers = ["Especie", "Tier"] + MODELOS
    table   = []
    for (especie, tier), model_status in sorted(rows.items()):
        table.append([especie, tier] + [model_status.get(m, "—") for m in MODELOS])

    if _HAS_TABULATE:
        print(tabulate(table, headers=headers, tablefmt="simple"))
    else:
        w = [max(len(h), max((len(str(r[i])) for r in table), default=0)) for i, h in enumerate(headers)]
        print("  ".join(h.ljust(w[i]) for i, h in enumerate(headers)))
        print("  ".join("-" * wi for wi in w))
        for row in table:
            print("  ".join(str(v).ljust(w[i]) for i, v in enumerate(row)))

    done    = sum(1 for c in classified if c["status"] == "done")
    failed  = sum(1 for c in classified if c["status"] == "failed")
    missing = sum(1 for c in classified if c["status"] == "missing")
    total   = len(classified)
    print(f"\nSummary: expected={total}  done={done}  failed={failed}  missing={missing}")


# ── Resubmit ──────────────────────────────────────────────────────────────────

def resubmit_missing(exp_dir, classified, dry_run=False, filter_species=None):
    """
    Inline-eval each missing/failed combo. Regenerates results.csv after all done.
    """
    to_resubmit = [c for c in classified if c["status"] in ("missing", "failed")]
    if filter_species:
        to_resubmit = [c for c in to_resubmit if c["especie"] == filter_species]

    if not to_resubmit:
        print("\nNada que reenviar.")
        return

    print(f"\n{len(to_resubmit)} combo(s) para reenviar:")

    if dry_run:
        for c in to_resubmit:
            print(f"  [DRY] {c['especie']} | {c['tier']} | {c['modelo']} (status={c['status']})")
        return

    import config
    from llm.judge_client import EnsembleJudge, ficha_summary

    judge    = EnsembleJudge(openrouter_api_key=config.OPENROUTER_API_KEY)
    eval_log = _load_json(os.path.join(exp_dir, "evaluation_log.json"))
    exp_log  = _load_json(os.path.join(exp_dir, "experiment_log.json"))

    for c in to_resubmit:
        especie      = c["especie"]
        tier         = c["tier"]
        modelo       = c["modelo"]
        perfil       = c["perfil"]          # for key storage
        perfil_judge = c.get("perfil_judge", perfil)  # for judge call
        especie_dir  = c["especie_dir"]
        key          = f"{especie}|{tier}|{modelo}|{perfil}|eval"

        print(f"\n  → {especie} | {tier} | {modelo}")

        # Load LLM response
        with open(c["txt_path"], encoding="utf-8") as f:
            contenido = f.read()
        marker = "[ANÁLISIS HÍBRIDO GENERADO POR IA]"
        respuesta = contenido.split(marker)[1].strip() if marker in contenido else contenido.strip()

        # Load ficha MdP
        ficha_paths = glob.glob(os.path.join(especie_dir, "*_ficha_MdP.txt"))
        if not ficha_paths:
            ficha_paths = glob.glob(os.path.join(especie_dir, "*", "*_ficha_MdP.txt"))
        if not ficha_paths:
            print(f"    [SKIP] Sin ficha MdP para {especie}")
            continue
        with open(ficha_paths[0], encoding="utf-8") as f:
            ficha_mdp = ficha_summary(f.read())

        # Load pregunta
        pregunta_key = f"{especie}|pregunta|{perfil}"
        pregunta = exp_log.get(pregunta_key, {}).get("pregunta", "")
        if not pregunta:
            # Fallback: extract from txt header
            with open(c["txt_path"], encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Pregunta Usuario"):
                        pregunta = line.split(":", 1)[1].strip()
                        break

        tier_dir = os.path.join(especie_dir, tier)
        scores = judge.evaluar(
            pregunta=pregunta,
            respuesta=respuesta,
            ficha_mdp=ficha_mdp,
            perfil=perfil_judge,
            especie=especie,
            tier=tier,
            modelo_generador=modelo,
            output_dir=tier_dir,
        )

        if scores:
            exp_id = os.path.basename(exp_dir)
            scores["exp_id"] = exp_id
            scores["perfil"] = perfil
            eval_log[key] = {
                "status":    "done",
                "scores":    scores,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            print(f"    [OK] score={scores['score_compuesto']:.3f}")
        else:
            eval_log[key] = {
                "status":    "failed",
                "timestamp": datetime.datetime.now().isoformat(),
            }
            print("    [FAIL]")

        _save_json(eval_log, os.path.join(exp_dir, "evaluation_log.json"))

    # Regenerate results.csv from all done entries
    _regenerate_results_csv(exp_dir, eval_log)


def _regenerate_results_csv(exp_dir, eval_log):
    rows = [
        entry["scores"]
        for key, entry in eval_log.items()
        if key.endswith("|eval") and entry.get("status") == "done" and "scores" in entry
    ]
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = ["exp_id", "especie", "tier", "modelo_generador", "perfil",
            "M5_profundidad_analitica", "M1_precision_geografica", "M3_relevancia_respuesta",
            "M2_precision_altitudinal", "M4_variable_climatica", "score_compuesto",
            "disagree_flag", "taxonomy_valid"]
    df = df[[c for c in cols if c in df.columns]]
    csv_path = os.path.join(exp_dir, "results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  [CSV] Regenerado: {csv_path} ({len(df)} filas)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audit & resubmit CR-BioLM evaluations")
    parser.add_argument("--exp-id",   required=True, help="Experiment ID (e.g. EXP-20260504-001-botanico)")
    parser.add_argument("--resubmit", action="store_true", help="Resubmit missing/failed combos")
    parser.add_argument("--dry-run",  action="store_true", help="Preview without calling judges")
    parser.add_argument("--species",  type=str, default=None, help="Limit resubmit to this species")
    args = parser.parse_args()

    exp_dir = os.path.join(RUNS_DIR, args.exp_id)
    if not os.path.isdir(exp_dir):
        print(f"[ERROR] No existe: {exp_dir}")
        sys.exit(1)

    combos     = discover_expected_combos(exp_dir)
    classified = classify_combos(exp_dir, combos)

    print_status_table(args.exp_id, classified)

    if args.resubmit or args.dry_run:
        resubmit_missing(exp_dir, classified, dry_run=args.dry_run, filter_species=args.species)


if __name__ == "__main__":
    main()
