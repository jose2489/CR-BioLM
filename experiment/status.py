#!/usr/bin/env python
# experiment/status.py
#
# CLI status viewer for CR-BioLM experiment runs.
#
# Usage:
#   python experiment/status.py --exp-id EXP-...         # full table
#   python experiment/status.py --all                     # one-line summary per run
#   python experiment/status.py --exp-id EXP-... --tail   # live-follow run.log
#   python experiment/status.py --exp-id EXP-... --json   # raw JSON output

import os
import sys
import json
import glob
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.audit_evals import discover_expected_combos, classify_combos

RUNS_DIR = os.path.join("experiment", "runs")
TIERS    = ["T0", "T1", "T3"]
MODELOS  = ["openai_gpt_4o", "anthropic_claude_sonnet_4_5"]

try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _read_log_tail(log_path, n=30):
    if not os.path.exists(log_path):
        return []
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:]]


# ── Status derivation ─────────────────────────────────────────────────────────

def get_exp_status(exp_dir):
    exp_id   = os.path.basename(exp_dir)
    meta     = _load_json(os.path.join(exp_dir, "experiment_meta.json"))
    exp_log  = _load_json(os.path.join(exp_dir, "experiment_log.json"))
    eval_log = _load_json(os.path.join(exp_dir, "evaluation_log.json"))
    log_path = os.path.join(exp_dir, "run.log")

    # Generation status from experiment_log.json
    gen_status = {}
    for key, val in exp_log.items():
        if "|" in key and not key.endswith("|eval"):
            parts = key.split("|")
            if len(parts) == 2 and parts[1] in TIERS:
                gen_status[key] = val.get("status", "unknown")

    # Evaluation status from evaluation_log.json
    eval_status = {}
    for key, val in eval_log.items():
        if key.endswith("|eval"):
            eval_status[key] = val.get("status", "unknown")

    # Counts
    gen_done    = sum(1 for s in gen_status.values() if s == "done")
    gen_failed  = sum(1 for s in gen_status.values() if s == "failed")
    eval_done   = sum(1 for s in eval_status.values() if s == "done")
    eval_failed = sum(1 for s in eval_status.values() if s == "failed")

    # Expected eval combos
    try:
        combos = discover_expected_combos(exp_dir)
        classified = classify_combos(exp_dir, combos)
        eval_missing = sum(1 for c in classified if c["status"] == "missing")
        eval_expected = len(classified)
    except Exception:
        eval_missing  = 0
        eval_expected = eval_done + eval_failed

    return {
        "exp_id":        exp_id,
        "meta":          meta,
        "generation":    gen_status,
        "evaluation":    eval_status,
        "log_tail":      _read_log_tail(log_path),
        "summary": {
            "gen_done":      gen_done,
            "gen_failed":    gen_failed,
            "gen_total":     len(gen_status),
            "eval_done":     eval_done,
            "eval_failed":   eval_failed,
            "eval_missing":  eval_missing,
            "eval_expected": eval_expected,
        },
    }


# ── Display ───────────────────────────────────────────────────────────────────

def _status_symbol(s):
    return {"done": "✓", "failed": "✗", "missing": "○", "pending": "…"}.get(s, s or "—")


def print_exp_full(data):
    meta    = data["meta"]
    summary = data["summary"]
    exp_id  = data["exp_id"]

    print(f"\n{'='*60}")
    print(f"Experiment : {exp_id}")
    print(f"Persona    : {meta.get('persona', '?')} | Species: {meta.get('n_species', '?')} | Status: {meta.get('status', '?')}")
    print(f"Started    : {str(meta.get('started_at', ''))[:19]}")
    print(f"{'='*60}")
    print(f"Generation : {summary['gen_done']}/{summary['gen_total']} done   "
          f"{summary['gen_failed']} failed")
    print(f"Evaluation : {summary['eval_done']}/{summary['eval_expected']} done   "
          f"{summary['eval_failed']} failed   {summary['eval_missing']} missing")

    if data["log_tail"]:
        print("\n--- run.log (last lines) ---")
        for line in data["log_tail"]:
            print(f"  {line}")


def print_all_summary(exp_dirs):
    rows = []
    for exp_dir in exp_dirs:
        data = get_exp_status(exp_dir)
        s    = data["summary"]
        m    = data["meta"]
        rows.append([
            data["exp_id"],
            m.get("persona", "?"),
            m.get("status", "?"),
            f"{s['gen_done']}/{s['gen_total']}",
            f"{s['eval_done']}/{s['eval_expected']}",
            f"{s['eval_missing']} missing",
        ])
    headers = ["exp_id", "persona", "status", "gen", "eval", "gaps"]
    if _HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="simple"))
    else:
        w = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
        print("  ".join(h.ljust(w[i]) for i, h in enumerate(headers)))
        print("  ".join("-" * wi for wi in w))
        for row in rows:
            print("  ".join(str(v).ljust(w[i]) for i, v in enumerate(row)))


# ── Tail mode ─────────────────────────────────────────────────────────────────

def tail_log(exp_dir, initial_lines=20):
    log_path = os.path.join(exp_dir, "run.log")
    if not os.path.exists(log_path):
        print(f"[WAIT] {log_path} not yet created...")

    seek_pos = 0
    try:
        # Print last N lines first
        tail = _read_log_tail(log_path, initial_lines)
        for line in tail:
            print(line)
        if os.path.exists(log_path):
            seek_pos = os.path.getsize(log_path)

        print(f"--- following {log_path} (Ctrl+C to stop) ---")
        while True:
            time.sleep(2)
            if not os.path.exists(log_path):
                continue
            size = os.path.getsize(log_path)
            if size > seek_pos:
                with open(log_path, encoding="utf-8") as f:
                    f.seek(seek_pos)
                    new_lines = f.read()
                seek_pos = size
                for line in new_lines.splitlines():
                    print(line)
    except KeyboardInterrupt:
        print("\n[tail stopped]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CR-BioLM experiment status viewer")
    parser.add_argument("--exp-id", type=str, default=None)
    parser.add_argument("--all",    action="store_true", help="One-line summary per run")
    parser.add_argument("--tail",   action="store_true", help="Live-follow run.log")
    parser.add_argument("--json",   action="store_true", help="Raw JSON output")
    args = parser.parse_args()

    if args.all:
        exp_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, "EXP-*")))
        if not exp_dirs:
            print("No experiment runs found.")
            return
        print_all_summary(exp_dirs)
        return

    if not args.exp_id:
        parser.error("Specify --exp-id or --all")

    exp_dir = os.path.join(RUNS_DIR, args.exp_id)
    if not os.path.isdir(exp_dir):
        print(f"[ERROR] Not found: {exp_dir}")
        sys.exit(1)

    if args.tail:
        tail_log(exp_dir)
        return

    data = get_exp_status(exp_dir)

    if args.json:
        # Remove log_tail from JSON (too verbose); caller can ask explicitly
        out = {k: v for k, v in data.items() if k != "log_tail"}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print_exp_full(data)


if __name__ == "__main__":
    main()
