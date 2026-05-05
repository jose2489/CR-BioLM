"""
experiment/db_sync.py

Syncs all EXP-* run data (experiment_meta.json + results.csv) into SQLite.
Safe to re-run — uses UPSERT so existing rows are updated, never duplicated.

Usage:
    python experiment/db_sync.py              # sync all runs
    python experiment/db_sync.py EXP-20260504-001-botanico  # sync one run
"""

import os
import sys
import json
import glob
import sqlite3
from datetime import datetime, timezone

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.db import get_conn, init_db

RUNS_DIR = os.path.join("experiment", "runs")


def _coerce(val):
    """Convert N/A / True / False strings to Python-native types."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    if s in ("N/A", "nan", "None", ""):
        return None
    if s == "True":
        return 1
    if s == "False":
        return 0
    try:
        return float(s)
    except ValueError:
        return s


def sync_experiment(exp_dir, conn):
    meta_path = os.path.join(exp_dir, "experiment_meta.json")
    csv_path  = os.path.join(exp_dir, "results.csv")

    if not os.path.isfile(meta_path):
        print(f"  [SKIP] No experiment_meta.json in {exp_dir}")
        return 0

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    exp_id = meta.get("exp_id", os.path.basename(exp_dir))

    # Upsert experiment row
    conn.execute("""
        INSERT INTO experiments (exp_id, persona, n_species, started_at, status, notes)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(exp_id) DO UPDATE SET
            status=excluded.status,
            n_species=excluded.n_species,
            notes=excluded.notes
    """, (
        exp_id,
        meta.get("persona"),
        meta.get("n_species"),
        meta.get("started_at"),
        meta.get("status"),
        meta.get("notes"),
    ))

    if not os.path.isfile(csv_path):
        print(f"  [WARN] No results.csv for {exp_id}")
        return 0

    df = pd.read_csv(csv_path)
    now = datetime.now(timezone.utc).isoformat()
    upserted = 0

    for _, row in df.iterrows():
        conn.execute("""
            INSERT INTO llm_evaluations
                (exp_id, especie, tier, perfil, modelo_generador, stratum,
                 M1, M2, M3, M4, M5, score_compuesto,
                 taxonomy_valid, disagree_flag, needs_human_review, synced_at)
            VALUES (?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?)
            ON CONFLICT(exp_id, especie, tier, modelo_generador) DO UPDATE SET
                M1=excluded.M1, M2=excluded.M2, M3=excluded.M3,
                M4=excluded.M4, M5=excluded.M5,
                score_compuesto=excluded.score_compuesto,
                taxonomy_valid=excluded.taxonomy_valid,
                disagree_flag=excluded.disagree_flag,
                synced_at=excluded.synced_at
        """, (
            exp_id,
            row.get("especie"),
            row.get("tier"),
            row.get("perfil", meta.get("persona")),
            row.get("modelo_generador"),
            _coerce(row.get("stratum")),
            _coerce(row.get("M1_precision_geografica")),
            _coerce(row.get("M2_precision_altitudinal")),
            _coerce(row.get("M3_relevancia_respuesta")),
            _coerce(row.get("M4_variable_climatica")),
            _coerce(row.get("M5_profundidad_analitica")),
            _coerce(row.get("score_compuesto")),
            _coerce(row.get("taxonomy_valid")),
            _coerce(row.get("disagree_flag")),
            0,  # needs_human_review — set manually or via flag_review_candidates()
            now,
        ))
        upserted += 1

    return upserted


def main(target_exp=None):
    init_db()
    pattern = os.path.join(RUNS_DIR, target_exp if target_exp else "EXP-*")
    dirs = sorted(glob.glob(pattern))

    if not dirs:
        print(f"No experiment dirs found matching: {pattern}")
        return

    total = 0
    with get_conn() as conn:
        for exp_dir in dirs:
            if not os.path.isdir(exp_dir):
                continue
            exp_id = os.path.basename(exp_dir)
            print(f"Syncing {exp_id}...")
            n = sync_experiment(exp_dir, conn)
            print(f"  → {n} rows upserted")
            total += n

    print(f"\nDone. {total} total rows synced.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    main(target)
