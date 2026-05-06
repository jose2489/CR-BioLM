#!/usr/bin/env python
# experiment/run_experiment.py
#
# Orquestador del experimento CR-BioLM.
# Cada ejecución crea un directorio con código único: EXP-YYYYMMDD-NNN-{persona}
# Trackea estado en experiment/runs/{exp_id}/experiment_log.json — resumible.
#
# Uso:
#   python experiment/run_experiment.py --persona botanico
#   python experiment/run_experiment.py --persona random --n 15 --seed 7
#   python experiment/run_experiment.py --exp-id EXP-20260416-001-botanico --resume
#   python experiment/run_experiment.py --dry-run --persona turista --n 10

import os
import sys
import json
import argparse
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import config
import pandas as pd
from utils.question_bank import get_random_question, get_question_meta
from main import procesar_especie

RUNS_DIR     = os.path.join("experiment", "runs")
CATALOG_PATH = os.path.join("outputs", "picked_species_enhanced_clean.csv")
TIERS        = ["T0", "T1", "T3"]  # T2 dropped — ficha leakage fix
PERSONAS     = ["botanico", "turista"]


# ── Experiment ID ─────────────────────────────────────────────────────────────

def generar_exp_id(persona: str) -> str:
    """Genera un código único: EXP-YYYYMMDD-NNN-{persona}"""
    hoy = datetime.date.today().strftime("%Y%m%d")
    counter = 1
    while True:
        exp_id = f"EXP-{hoy}-{counter:03d}-{persona}"
        if not os.path.exists(os.path.join(RUNS_DIR, exp_id)):
            return exp_id
        counter += 1


# ── Log helpers ───────────────────────────────────────────────────────────────

def log_path(exp_dir: str) -> str:
    return os.path.join(exp_dir, "experiment_log.json")


def load_log(exp_dir: str) -> dict:
    p = log_path(exp_dir)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_log(log: dict, exp_dir: str):
    with open(log_path(exp_dir), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def save_meta(exp_dir: str, meta: dict):
    with open(os.path.join(exp_dir, "experiment_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def tier_key(especie: str, tier: str) -> str:
    return f"{especie}|{tier}"


def is_done(log: dict, key: str) -> bool:
    return log.get(key, {}).get("status") == "done"


# ── Catálogo ──────────────────────────────────────────────────────────────────

def cargar_catalogo(species_file: str = None) -> pd.DataFrame:
    df = pd.read_csv(CATALOG_PATH)
    if species_file:
        with open(species_file, encoding="utf-8") as f:
            nombres = [l.strip() for l in f if l.strip()]
        df = df[df["species"].isin(nombres)]
        print(f"[INFO] Filtrado a {len(df)} especies del archivo {species_file}")
    return df


# ── Question pre-assignment ────────────────────────────────────────────────────

def assign_questions(especies_list: list, persona_map: dict, log: dict, rng: random.Random) -> dict:
    """
    Assign questions without replacement per persona pool.
    persona_map: {especie: persona_str}
    Returns: {especie: {"q": ..., "stratum": ...}}
    """
    assignments = {}

    # Group species that still need assignment, per persona
    needs = {}
    for sp in especies_list:
        persona = persona_map[sp]
        key = f"{sp}|pregunta|{persona}"
        if key not in log:
            needs.setdefault(persona, []).append(sp)

    # Build pool per persona and assign
    for persona, species_needing in needs.items():
        entries = get_question_meta(persona)
        pool = []
        while len(pool) < len(species_needing):
            chunk = entries[:]
            rng.shuffle(chunk)
            pool.extend(chunk)
        for sp, entry in zip(species_needing, pool):
            assignments[sp] = entry

    return assignments


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Orquestador experimento CR-BioLM")
    parser.add_argument("--exp-id", type=str, default=None,
                        help="ID de experimento existente para resumir (ej: EXP-20260416-001-botanico)")
    parser.add_argument("--species-file", type=str, default=None,
                        help="TXT con una especie por línea (default: catálogo completo)")
    parser.add_argument("--persona", type=str, default="botanico",
                        choices=["turista", "botanico", "random"],
                        help="Persona fija o 'random' para asignar aleatoriamente por especie")
    parser.add_argument("--n", type=int, default=None,
                        help="Número de especies a muestrear del catálogo (default: todas)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria para reproducibilidad (default: 42)")
    parser.add_argument("--notes",   type=str, default="",
                        help="Notas opcionales para documentar el experimento")
    parser.add_argument("--resume",  action="store_true",
                        help="Salta tiers ya completados (requiere --exp-id)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra el plan sin ejecutar nada")
    args = parser.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)
    rng = random.Random(args.seed)

    # ── Resolver ID del experimento ──
    persona_label = "mixed" if args.persona == "random" else args.persona
    if args.exp_id:
        exp_id  = args.exp_id
        exp_dir = os.path.join(RUNS_DIR, exp_id)
        if not os.path.exists(exp_dir) and not args.dry_run:
            print(f"[ERROR] No existe el experimento: {exp_id}")
            sys.exit(1)
    else:
        exp_id  = generar_exp_id(persona_label)
        exp_dir = os.path.join(RUNS_DIR, exp_id)

    os.makedirs(exp_dir, exist_ok=True)

    log      = load_log(exp_dir) if args.resume else {}
    catalogo = cargar_catalogo(args.species_file)

    # ── Muestrear N especies si se especificó --n ──
    if args.n is not None:
        if args.n > len(catalogo):
            print(f"[WARN] --n {args.n} es mayor que el catálogo ({len(catalogo)}), usando todas.")
        else:
            catalogo = catalogo.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
            print(f"[INFO] Muestreadas {args.n} especies con seed={args.seed}")

    total = len(catalogo)
    especies_list = [str(r["species"]).strip() for _, r in catalogo.iterrows()]

    # ── Asignar persona por especie ──
    persona_map = {}
    if args.persona == "random":
        # Recover from log if resuming
        for sp in especies_list:
            log_key = f"{sp}|persona"
            if log_key in log:
                persona_map[sp] = log[log_key]
            else:
                persona_map[sp] = rng.choice(PERSONAS)
        # Persist new persona assignments immediately
        changed = False
        for sp in especies_list:
            log_key = f"{sp}|persona"
            if log_key not in log:
                log[log_key] = persona_map[sp]
                changed = True
        if changed and not args.dry_run:
            save_log(log, exp_dir)
    else:
        for sp in especies_list:
            persona_map[sp] = args.persona

    # ── Pre-asignar preguntas sin reemplazo por pool de persona ──
    _question_assignments = assign_questions(especies_list, persona_map, log, rng)

    # ── Guardar metadatos del experimento ──
    meta = {
        "exp_id":        exp_id,
        "persona":       args.persona,
        "n_species":     total,
        "tiers":         TIERS,
        "seed":          args.seed,
        "notes":         args.notes,
        "started_at":    datetime.datetime.now().isoformat(),
        "species_file":  args.species_file or "catalogo_completo",
        "models":        ["openai/gpt-4o", "anthropic/claude-sonnet-4-5"],
    }
    if not args.dry_run:
        save_meta(exp_dir, meta)

    print(f"\n{'='*60}")
    print(f"[EXPERIMENTO] {exp_id}")
    print(f"[EXPERIMENTO] {total} especies × {len(TIERS)} tiers (T0/T1/T3) × 2 modelos")
    print(f"[EXPERIMENTO] Persona: {args.persona} | Seed: {args.seed} | Resume: {args.resume}")
    print(f"[EXPERIMENTO] Directorio: {exp_dir}")
    if args.notes:
        print(f"[EXPERIMENTO] Notas: {args.notes}")
    print(f"{'='*60}\n")

    done_count = fail_count = skip_count = 0

    for idx, (_, row) in enumerate(catalogo.iterrows(), 1):
        especie = str(row["species"]).strip()
        persona = persona_map[especie]
        print(f"\n[{idx}/{total}] {especie}  (persona: {persona})")

        # ── Asignar pregunta fija (misma en T0/T1/T3 para esta especie) ──
        pregunta_key = f"{especie}|pregunta|{persona}"
        if pregunta_key in log and "pregunta" in log[pregunta_key]:
            pregunta = log[pregunta_key]["pregunta"]
            stratum  = log[pregunta_key].get("stratum", "A")
            print(f"  [Q] (recuperada) [{stratum}] {pregunta}")
        else:
            entry    = _question_assignments[especie]
            pregunta = entry["q"]
            stratum  = entry.get("stratum", "A")
            log[pregunta_key] = {"pregunta": pregunta, "stratum": stratum}
            if not args.dry_run:
                save_log(log, exp_dir)
            print(f"  [Q] (nueva) [{stratum}] {pregunta}")

        # ── Correr cada tier ──
        for tier in TIERS:
            key      = tier_key(especie, tier)
            tier_dir = os.path.join(exp_dir, especie.replace(" ", "_"), tier)

            if args.resume and is_done(log, key):
                print(f"  [SKIP] {tier} ya completado")
                skip_count += 1
                continue

            if args.dry_run:
                estado = "✓ done" if is_done(log, key) else "○ pendiente"
                print(f"  {estado} — {tier}")
                continue

            print(f"  → Corriendo {tier}...")
            os.makedirs(tier_dir, exist_ok=True)

            try:
                exito = procesar_especie(
                    especie_nombre=especie,
                    user_question=pregunta,
                    tier=tier,
                    output_dir_override=tier_dir,
                )
                if exito:
                    log[key] = {
                        "status":    "done",
                        "output":    tier_dir,
                        "persona":   persona,
                        "pregunta":  pregunta,
                        "stratum":   stratum,
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                    done_count += 1
                    print(f"  [OK] {tier} → {tier_dir}")
                else:
                    log[key] = {
                        "status":    "failed",
                        "error":     "procesar_especie retornó False",
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                    fail_count += 1
                    print(f"  [FAIL] {tier}")
            except Exception as e:
                log[key] = {
                    "status":    "failed",
                    "error":     str(e)[:300],
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                fail_count += 1
                print(f"  [ERROR] {tier}: {e}")

            save_log(log, exp_dir)

    # ── Actualizar metadatos con resultado final ──
    if not args.dry_run:
        meta["finished_at"] = datetime.datetime.now().isoformat()
        meta["done_count"]  = done_count
        meta["fail_count"]  = fail_count
        meta["skip_count"]  = skip_count
        meta["status"]      = "complete" if fail_count == 0 else "partial"
        save_meta(exp_dir, meta)

    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"[DRY-RUN] Plan para {exp_id}")
    else:
        print(f"[FIN] {exp_id}")
        print(f"      Completados: {done_count} | Saltados: {skip_count} | Fallidos: {fail_count}")
    print(f"[DIR] {exp_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
