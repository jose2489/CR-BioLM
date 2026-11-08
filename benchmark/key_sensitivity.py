"""Sensitivity of the confirmatory scores to the key corrections made after the run.

The answers were scored against questions_main.jsonl. The corrections of the catalogue
and of the keys the botanist found wrong were folded into questions_main_revised.jsonl
afterwards. This compares the two question sets and re-scores the stored answers against
the revised keys, reusing each answer's stored segmentation (stance, abstention, claims),
so no model is called: only the key changes.

Every answer whose key changed is scored twice, against the old key and the new one,
under the same (current) checker, so the difference is the key's alone. The old-key
verdict is also compared with the stored one: where they differ, the checker itself
changed after the run.

Run:  python -m benchmark.key_sensitivity
"""
from __future__ import annotations

import collections
import io
import json

from . import verify as V

NOT_ANSWERED = ("abstained", "not_scored", "beyond_record")


def _jsonl(name: str) -> list[dict]:
    from pathlib import Path
    return [json.loads(l) for l in io.open(Path(__file__).resolve().parent / "data" / name,
                                           encoding="utf-8") if l.strip()]


def _precision(rows: list[dict], field: str) -> dict[str, float]:
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r[field] in NOT_ANSWERED:
            continue
        by[r["arm"]][0] += r[field] == "correct"
        by[r["arm"]][1] += 1
    return {a: c / n for a, (c, n) in by.items() if n}


def main() -> None:
    from mpcr_rag.store import pg_store
    old = {q["qid"]: q for q in _jsonl("questions_main.jsonl")}
    new = {q["qid"]: q for q in _jsonl("questions_main_revised.jsonl")}
    added = sorted(set(new) - set(old))
    dropped = sorted(set(old) - set(new))
    changed = {qid for qid in set(old) & set(new) if old[qid]["key"] != new[qid]["key"]}
    print(f"questions: {len(old)} -> {len(new)}; added {len(added)}, dropped {len(dropped)}, "
          f"key changed {len(changed)}")
    print("  added:", ", ".join(added))
    print("  key changed, by type:",
          dict(collections.Counter(old[q]["qtype"] for q in changed)))
    answer_keys = collections.Counter()
    for q in changed:
        a, b = old[q]["key"].get("answer"), new[q]["key"].get("answer")
        answer_keys["answer changed" if a != b else "details only"] += 1
    print("  of which:", dict(answer_keys))

    rows = [v for v in _jsonl("verdicts_main.jsonl") if not v.get("error")]
    texts = {(a["qid"], a["arm"], str(a.get("replicate", 0))): a["answer_text"]
             for a in _jsonl("answers_main.jsonl") if a.get("answer_text")}
    conn = pg_store.connect()
    flips, mismatch = collections.Counter(), 0
    out = []
    with conn.cursor() as cur:
        records = {}
        for v in rows:
            r = {"arm": v["arm"], "qid": v["qid"], "old": v["direct"], "new": v["direct"]}
            if v["qid"] in changed:
                if v["species"] not in records:
                    records[v["species"]] = V.load_record(cur, v["species"])
                rec = records[v["species"]]
                from .questions import _USE_CUE
                any_use = bool(_USE_CUE.search(
                    f"{rec.ficha.discussion} {rec.ficha.distribution_paragraph}"))
                ext = {"stance": v.get("stance"), "abstains": v.get("abstains"),
                       "claims": v.get("claims") or []}
                answer = texts[(v["qid"], v["arm"], str(v.get("replicate", 0)))]
                again = V.check_direct(old[v["qid"]], answer, ext, any_use)
                mismatch += again != v["direct"]
                # The key's effect alone: both keys under the same, current checker.
                r["old"] = again
                r["new"] = V.check_direct(new[v["qid"]], answer, ext, any_use)
                if r["new"] != r["old"]:
                    flips[(old[v["qid"]]["qtype"], r["old"], r["new"])] += 1
            out.append(r)
    conn.close()
    print(f"\nre-scored answers whose key changed; old-key verdict differs from the stored "
          f"one in {mismatch} (the checker changed after the run; old and new keys are "
          f"compared under the same, current checker)")
    print(f"direct verdicts that change: {sum(flips.values())}")
    for k, n in flips.most_common():
        print(f"  {n:3d}  {k[0]}: {k[1]} -> {k[2]}")
    p_old, p_new = _precision(out, "old"), _precision(out, "new")
    print("\nprecision among answered, stored -> revised keys")
    worst = 0.0
    for a in sorted(p_old):
        d = p_new[a] - p_old[a]
        worst = max(worst, abs(d))
        print(f"  {a:4s} {p_old[a]:6.1%} -> {p_new[a]:6.1%}  ({100 * d:+.1f} points)")
    print(f"largest change: {100 * worst:.1f} points")


if __name__ == "__main__":
    main()
