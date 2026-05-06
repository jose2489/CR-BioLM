import os
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone

DATABASE_URL = os.getenv("DATABASE_URL")

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    exp_id          TEXT PRIMARY KEY,
    persona         TEXT,
    n_species       INTEGER,
    started_at      TEXT,
    status          TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS llm_evaluations (
    id               SERIAL PRIMARY KEY,
    exp_id           TEXT NOT NULL,
    especie          TEXT NOT NULL,
    tier             TEXT NOT NULL,
    perfil           TEXT NOT NULL,
    modelo_generador TEXT NOT NULL,
    stratum          TEXT,
    M1               REAL,
    M2               REAL,
    M3               REAL,
    M4               REAL,
    M5               REAL,
    score_compuesto  REAL,
    taxonomy_valid   INTEGER,
    disagree_flag    INTEGER,
    needs_human_review INTEGER,
    synced_at        TEXT,
    UNIQUE(exp_id, especie, tier, modelo_generador)
);

CREATE TABLE IF NOT EXISTS expert_sessions (
    username         TEXT PRIMARY KEY,
    model_A          TEXT NOT NULL,
    model_B          TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS human_evaluations (
    id               SERIAL PRIMARY KEY,
    exp_id           TEXT NOT NULL,
    especie          TEXT NOT NULL,
    tier             TEXT NOT NULL,
    modelo_generador TEXT NOT NULL,
    evaluator        TEXT NOT NULL,
    M1               INTEGER,
    M2               INTEGER,
    M3               INTEGER,
    M4               INTEGER,
    M5               INTEGER,
    comment          TEXT,
    submitted_at     TEXT,
    UNIQUE(exp_id, especie, tier, modelo_generador, evaluator)
);
"""


def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL env var is not set")
    conn = psycopg2.connect(DATABASE_URL)
    return conn


def _cur(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    with get_conn() as conn:
        cur = conn.cursor()
        for stmt in [s.strip() for s in SCHEMA.split(";") if s.strip()]:
            cur.execute(stmt)
        conn.commit()


# ── Analytics queries ─────────────────────────────────────────────────────────

def score_summary(exp_id=None):
    """Mean/std/n per tier per modelo_generador, LLM and human side by side."""
    with get_conn() as conn:
        cur = _cur(conn)
        where = "WHERE exp_id = %s" if exp_id else ""
        params = (exp_id,) if exp_id else ()
        cur.execute(f"""
            SELECT tier, modelo_generador,
                   ROUND(AVG(score_compuesto)::numeric, 3) AS mean_score,
                   ROUND(AVG(M1)::numeric, 3) AS mean_M1,
                   ROUND(AVG(M3)::numeric, 3) AS mean_M3,
                   ROUND(AVG(M5)::numeric, 3) AS mean_M5,
                   COUNT(*) AS n
            FROM llm_evaluations
            {where}
            GROUP BY tier, modelo_generador
            ORDER BY tier, modelo_generador
        """, params)
        return [dict(r) for r in cur.fetchall()]


def score_trend_across_runs():
    """Score compuesto by tier over time (for thesis progression chart)."""
    with get_conn() as conn:
        cur = _cur(conn)
        cur.execute("""
            SELECT e.exp_id, e.started_at, lv.tier,
                   ROUND(AVG(lv.score_compuesto)::numeric, 3) AS mean_score,
                   COUNT(*) AS n
            FROM llm_evaluations lv
            JOIN experiments e ON lv.exp_id = e.exp_id
            GROUP BY e.exp_id, e.started_at, lv.tier
            ORDER BY e.started_at, lv.tier
        """)
        return [dict(r) for r in cur.fetchall()]


def human_vs_llm_agreement(exp_id):
    """
    Returns per-metric mean scores from both human and LLM evaluators for the
    given experiment. Cohen's kappa computation requires scikit-learn and is
    left to the caller (needs the full raw vectors, not aggregates).
    """
    with get_conn() as conn:
        cur = _cur(conn)
        cur.execute("""
            SELECT tier, modelo_generador,
                   ROUND(AVG(M1)::numeric, 3) AS M1,
                   ROUND(AVG(M3)::numeric, 3) AS M3,
                   ROUND(AVG(M5)::numeric, 3) AS M5,
                   COUNT(*) AS n
            FROM llm_evaluations WHERE exp_id = %s
            GROUP BY tier, modelo_generador
        """, (exp_id,))
        llm_rows = cur.fetchall()

        cur.execute("""
            SELECT tier, modelo_generador,
                   ROUND(AVG(M1)::numeric, 3) AS M1,
                   ROUND(AVG(M3)::numeric, 3) AS M3,
                   ROUND(AVG(M5)::numeric, 3) AS M5,
                   COUNT(*) AS n
            FROM human_evaluations WHERE exp_id = %s
            GROUP BY tier, modelo_generador
        """, (exp_id,))
        human_rows = cur.fetchall()

        return {
            "llm":   [dict(r) for r in llm_rows],
            "human": [dict(r) for r in human_rows],
        }


def flag_review_candidates(exp_id):
    """Species where human score differs from LLM judge by >= 2 Likert points."""
    with get_conn() as conn:
        cur = _cur(conn)
        cur.execute("""
            SELECT h.especie, h.tier, h.modelo_generador, h.evaluator,
                   h.M1 AS h_M1, l.M1 AS l_M1,
                   h.M3 AS h_M3, l.M3 AS l_M3,
                   h.M5 AS h_M5, l.M5 AS l_M5
            FROM human_evaluations h
            JOIN llm_evaluations l
              ON h.exp_id = l.exp_id
             AND h.especie = l.especie
             AND h.tier    = l.tier
             AND h.modelo_generador = l.modelo_generador
            WHERE h.exp_id = %s
              AND (ABS(h.M1 - l.M1) >= 2
                OR ABS(h.M3 - l.M3) >= 2
                OR ABS(h.M5 - l.M5) >= 2)
            ORDER BY h.especie, h.tier
        """, (exp_id,))
        return [dict(r) for r in cur.fetchall()]


def export_for_thesis(exp_id):
    """
    Returns a list of dicts with all human + LLM scores joined — ready for
    pandas DataFrame / CSV export for R or SPSS.
    """
    with get_conn() as conn:
        cur = _cur(conn)
        cur.execute("""
            SELECT
                h.exp_id, h.especie, h.tier, h.modelo_generador, h.evaluator,
                h.M1 AS h_M1, h.M2 AS h_M2, h.M3 AS h_M3,
                h.M4 AS h_M4, h.M5 AS h_M5,
                h.comment, h.submitted_at,
                l.M1 AS l_M1, l.M2 AS l_M2, l.M3 AS l_M3,
                l.M4 AS l_M4, l.M5 AS l_M5,
                l.score_compuesto, l.taxonomy_valid, l.disagree_flag
            FROM human_evaluations h
            LEFT JOIN llm_evaluations l
              ON h.exp_id = l.exp_id
             AND h.especie = l.especie
             AND h.tier    = l.tier
             AND h.modelo_generador = l.modelo_generador
            WHERE h.exp_id = %s
            ORDER BY h.especie, h.tier, h.evaluator
        """, (exp_id,))
        return [dict(r) for r in cur.fetchall()]


def get_expert_progress(exp_id):
    """Returns {username: {especie: [tiers_done]}} for the progress view."""
    with get_conn() as conn:
        cur = _cur(conn)
        cur.execute("""
            SELECT evaluator, especie, tier
            FROM human_evaluations
            WHERE exp_id = %s
            ORDER BY evaluator, especie, tier
        """, (exp_id,))
        rows = cur.fetchall()
    progress = {}
    for r in rows:
        progress.setdefault(r["evaluator"], {}).setdefault(
            r["especie"], []).append(r["tier"])
    return progress


def save_human_evaluation(exp_id, especie, tier, modelo_generador, evaluator,
                          M1, M2, M3, M4, M5, comment=None):
    """
    Upsert one human evaluation row. M values are Likert integers or None.
    Safe to call multiple times (ON CONFLICT replaces).
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO human_evaluations
                (exp_id, especie, tier, modelo_generador, evaluator,
                 M1, M2, M3, M4, M5, comment, submitted_at)
            VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(exp_id, especie, tier, modelo_generador, evaluator)
            DO UPDATE SET
                M1=EXCLUDED.M1, M2=EXCLUDED.M2, M3=EXCLUDED.M3,
                M4=EXCLUDED.M4, M5=EXCLUDED.M5,
                comment=EXCLUDED.comment, submitted_at=EXCLUDED.submitted_at
        """, (exp_id, especie, tier, modelo_generador, evaluator,
              M1, M2, M3, M4, M5, comment,
              datetime.now(timezone.utc).isoformat()))
        conn.commit()


def get_first_unsubmitted(exp_id, evaluator, all_species):
    """
    Returns the especie_id of the first species that still has unsubmitted tiers.
    all_species is a list of {'especie': str, 'especie_id': str} in display order.
    """
    with get_conn() as conn:
        cur = _cur(conn)
        cur.execute("""
            SELECT especie, tier FROM human_evaluations
            WHERE exp_id=%s AND evaluator=%s
        """, (exp_id, evaluator))
        rows = cur.fetchall()
    done = {}
    for r in rows:
        done.setdefault(r["especie"], set()).add(r["tier"])

    tiers_needed = {"T0", "T1", "T3"}
    for sp in all_species:
        submitted = done.get(sp["especie"], set())
        if submitted < tiers_needed:
            return sp
    return None  # all done


def get_raw_scores_for_kappa(exp_id):
    """
    Returns paired (human, llm) score vectors per metric, for Cohen's κ.
    Only includes rows where both human and LLM scores exist.
    Returns: dict of metric -> {"human": [...], "llm": [...]}
    """
    with get_conn() as conn:
        cur = _cur(conn)
        cur.execute("""
            SELECT h.especie, h.tier, h.modelo_generador,
                   h.M1 AS h_M1, h.M3 AS h_M3, h.M5 AS h_M5,
                   h.M2 AS h_M2, h.M4 AS h_M4,
                   l.M1 AS l_M1, l.M3 AS l_M3, l.M5 AS l_M5,
                   l.M2 AS l_M2, l.M4 AS l_M4
            FROM human_evaluations h
            JOIN llm_evaluations l
              ON h.exp_id = l.exp_id
             AND h.especie = l.especie
             AND h.tier    = l.tier
             AND h.modelo_generador = l.modelo_generador
            WHERE h.exp_id = %s
        """, (exp_id,))
        rows = cur.fetchall()

    result = {m: {"human": [], "llm": []} for m in ["M1", "M3", "M5", "M2", "M4"]}
    for r in rows:
        for m in ["M1", "M3", "M5"]:
            hv = r[f"h_{m}"]
            lv = r[f"l_{m}"]
            if hv is not None and lv is not None:
                result[m]["human"].append(int(hv))
                result[m]["llm"].append(int(round(lv)))
        for m in ["M2", "M4"]:
            hv = r[f"h_{m}"]
            lv = r[f"l_{m}"]
            if hv is not None and lv is not None:
                result[m]["human"].append(int(hv))
                result[m]["llm"].append(int(round(lv)))
    return result


def upsert_expert_session(username, model_A, model_B):
    """Create expert session if it doesn't exist; return existing if it does."""
    with get_conn() as conn:
        cur = _cur(conn)
        cur.execute(
            "SELECT * FROM expert_sessions WHERE username = %s", (username,)
        )
        existing = cur.fetchone()
        if existing:
            return dict(existing)
        cur2 = conn.cursor()
        cur2.execute(
            "INSERT INTO expert_sessions (username, model_A, model_B, created_at) VALUES (%s,%s,%s,%s)",
            (username, model_A, model_B, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        return {"username": username, "model_A": model_A, "model_B": model_B}


if __name__ == "__main__":
    init_db()
    print(f"DB initialised at {os.getenv('DATABASE_URL', '(no DATABASE_URL set)')}")
