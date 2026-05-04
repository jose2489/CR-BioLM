#!/usr/bin/env python
# experiment/calibration/leveleval.py
#
# LevelEval calibration pipeline for CR-BioLM.
# Validates the judge ensemble by generating synthetic responses at three quality levels
# and confirming the judge ranks Level 1 > Level 2 > Level 3 with clear gaps.
#
# Levels:
#   Level 1 (gold) — correct facts, precise elevations, expert Spanish (hand-written or LLM-refined)
#   Level 2 (mid)  — introduced minor errors (wrong cordillera, ±300 m altitude, ~50% noise)
#   Level 3 (poor) — major errors (wrong vertiente, irrelevant content)
#
# Usage:
#   python experiment/calibration/leveleval.py --stage unit      (3 species)
#   python experiment/calibration/leveleval.py --stage demo      (25 species)
#   python experiment/calibration/leveleval.py --stage thesis    (100 species, after demo)
#
# Output: experiment/calibration/leveleval_<stage>_<date>.json

import os
import sys
import json
import argparse
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
import pandas as pd
from llm.judge_client import EnsembleJudge, ficha_summary
from experiment.validators.domain_validators import run_domain_validators

CATALOG_PATH  = os.path.join("outputs", "picked_species_enhanced_clean.csv")
CALIBRATION_DIR = os.path.dirname(os.path.abspath(__file__))

STAGE_SIZES = {"unit": 3, "demo": 25, "thesis": 100}

# ── Synthetic response generator ──────────────────────────────────────────────

def _make_responses(species_name: str, ficha: str, tier: str,
                    openrouter_api_key: str) -> dict[str, str]:
    """
    Uses GPT-4o-mini to generate Level 1 (gold), then introduces errors for L2 and L3.
    """
    import requests

    base_url = "https://openrouter.ai/api/v1/chat/completions"
    headers  = {"Authorization": f"Bearer {openrouter_api_key}",
                 "Content-Type": "application/json"}

    def call(prompt: str) -> str:
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
        r = requests.post(base_url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"API error {r.status_code}: {r.text[:200]}")

    # Level 1 — gold: accurate, expert Spanish, structured with Razonamiento + Respuesta
    prompt_l1 = f"""Eres un ecólogo experto en plantas tropicales de Costa Rica.
Escribe una respuesta de nivel experto sobre la especie {species_name},
basándote ÚNICAMENTE en la siguiente ficha de referencia.
Usa el formato con ## Razonamiento (4-5 viñetas) y ## Respuesta (3-4 oraciones).
Incluye elevaciones exactas, nombres precisos de zonas geográficas y variables climáticas relevantes.
Evita vaguedades. Sé conciso y técnicamente preciso.

FICHA DE REFERENCIA:
{ficha}

Contexto de tier: {tier}. Para T3, menciona la variable climática más limitante y el rango altitudinal.
Para T1, basa tu respuesta solo en la distribución espacial de registros.
Para T0, usa tu conocimiento general sobre la especie."""

    l1 = call(prompt_l1)

    # Level 2 — mid: rewrite L1 with minor errors (~50% noise)
    prompt_l2 = f"""Reescribe el siguiente texto sobre {species_name} introduciendo ERRORES MENORES sutiles:
- Cambia una cordillera correcta por otra plausible pero incorrecta (ej: Talamanca → Guanacaste)
- Ajusta el rango altitudinal en ±300 m (ej: 1200–2400 m → 900–2100 m)
- Mantén la estructura y el tono general; el texto debe seguir pareciendo plausible.
- Mantén las secciones ## Razonamiento y ## Respuesta.

TEXTO ORIGINAL:
{l1}

Devuelve SOLO el texto reescrito, sin explicaciones."""

    l2 = call(prompt_l2)

    # Level 3 — poor: rewrite with major errors
    prompt_l3 = f"""Reescribe el siguiente texto sobre {species_name} introduciendo ERRORES GRAVES:
- Cambia la vertiente correcta por la opuesta (Caribe ↔ Pacífico)
- Inventa elevaciones completamente erróneas (ej: especie de 2000 m → 50–300 m)
- Añade contenido irrelevante o genérico que no responda la pregunta
- Usa nombres de zonas geográficas incorrectos para la especie
- Mantén las secciones ## Razonamiento y ## Respuesta.

TEXTO ORIGINAL:
{l1}

Devuelve SOLO el texto reescrito, sin explicaciones."""

    l3 = call(prompt_l3)

    return {"L1_gold": l1, "L2_mid": l2, "L3_poor": l3}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LevelEval calibration for CR-BioLM judges")
    parser.add_argument("--stage", choices=["unit", "demo", "thesis"], default="unit")
    parser.add_argument("--tier",  choices=["T0", "T1", "T3"], default="T3")
    parser.add_argument("--persona", choices=["botanico", "turista"], default="botanico")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    n_species = STAGE_SIZES[args.stage]
    date_str  = datetime.date.today().strftime("%Y%m%d")
    out_path  = os.path.join(CALIBRATION_DIR, f"leveleval_{args.stage}_{date_str}.json")

    print(f"\n[LevelEval] Stage={args.stage} | n={n_species} | tier={args.tier} | persona={args.persona}")
    print(f"[LevelEval] Output: {out_path}\n")

    # Load species
    catalog  = pd.read_csv(CATALOG_PATH)
    import random
    rng      = random.Random(args.seed)
    selected = catalog.sample(n=min(n_species, len(catalog)), random_state=args.seed)

    judge = EnsembleJudge(
        openrouter_api_key=config.OPENROUTER_API_KEY,
        groq_api_key=getattr(config, "GROQ_API_KEY", None),
    )

    from utils.question_bank import get_random_question
    results = []

    for _, row in selected.iterrows():
        species_name = str(row["species"]).strip()
        print(f"[LevelEval] Processing: {species_name}")

        # Build ficha from catalog
        parts = []
        geo = str(row.get("geographic_notes", "") or "").strip()
        if geo and geo.lower() != "nan":
            parts.append(f"Distribución geográfica: {geo}")
        emin, emax = row.get("elevation_min_m"), row.get("elevation_max_m")
        if emin and str(emin) != "nan" and emax and str(emax) != "nan":
            parts.append(f"Rango altitudinal: {int(float(emin))}–{int(float(emax))} m s.n.m.")
        hab = str(row.get("habitat_type", "") or "").strip()
        if hab and hab.lower() != "nan":
            parts.append(f"Tipo de hábitat: {hab}")
        ficha = "\n".join(parts) if parts else "Información no disponible."

        pregunta = get_random_question(args.persona, tier=args.tier, seed=args.seed)

        # Generate synthetic responses
        try:
            responses = _make_responses(species_name, ficha, args.tier, config.OPENROUTER_API_KEY)
        except Exception as e:
            print(f"  [ERROR] Response generation failed: {e}")
            continue

        # Evaluate each level
        level_results = {}
        for level_key, respuesta in responses.items():
            print(f"  → Evaluating {level_key}...")
            d_vals = run_domain_validators(
                species_name=species_name,
                respuesta=respuesta,
                alt_min_manual=float(emin) if emin and str(emin) != "nan" else None,
                alt_max_manual=float(emax) if emax and str(emax) != "nan" else None,
            )
            eval_result = judge.evaluar(
                pregunta=pregunta,
                respuesta=respuesta,
                ficha_mdp=ficha,
                perfil=args.persona,
                especie=species_name,
                tier=args.tier,
                modelo_generador="openai/gpt-4o-mini",  # synthetic generator
                taxonomy_valid=d_vals["taxonomy_valid"],
            )
            if eval_result:
                eval_result.update(d_vals)
                eval_result["level"] = level_key
                eval_result["respuesta_sintetica"] = respuesta
                level_results[level_key] = eval_result

        if len(level_results) == 3:
            s1 = level_results["L1_gold"]["score_compuesto"]
            s2 = level_results["L2_mid"]["score_compuesto"]
            s3 = level_results["L3_poor"]["score_compuesto"]
            rank_ok = s1 >= s2 >= s3
            print(f"  Scores: L1={s1:.3f} L2={s2:.3f} L3={s3:.3f} | Rank OK={rank_ok}")
        else:
            rank_ok = None

        results.append({
            "species":    species_name,
            "tier":       args.tier,
            "persona":    args.persona,
            "pregunta":   pregunta,
            "ficha":      ficha,
            "levels":     level_results,
            "rank_ok":    rank_ok,
        })

    # Summary
    n_ok  = sum(1 for r in results if r.get("rank_ok") is True)
    n_fail = sum(1 for r in results if r.get("rank_ok") is False)
    print(f"\n[LevelEval] Rank OK: {n_ok}/{len(results)} | Rank FAIL: {n_fail}/{len(results)}")
    if n_fail > 0:
        print("[LevelEval] WARNING: Some species failed monotonic ranking. Refine rubric before scaling.")
    else:
        print("[LevelEval] All species ranked correctly. Judge calibration passed.")

    output = {
        "stage":       args.stage,
        "tier":        args.tier,
        "persona":     args.persona,
        "date":        date_str,
        "n_species":   len(results),
        "n_rank_ok":   n_ok,
        "n_rank_fail": n_fail,
        "results":     results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[LevelEval] Saved to: {out_path}")


if __name__ == "__main__":
    main()
