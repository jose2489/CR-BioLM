"""Does the flora matter most for the species with fewest records?

Splits the forecast's test species (forecast/per_species.jsonl) into tertiles of the
records known by the test cut (2009) and reports, per tertile and score, the median AUC
and the pooled top-10 lift. The Manual's contribution is the per-species difference
between the model with all evidence and the model without the Manual (paired Wilcoxon);
the records' contribution, between all evidence and the model without records.

Run:  python -m benchmark.forecast_tertiles
"""
from __future__ import annotations

import collections
import io
import json

import numpy as np

from .forecast import CUTS, OUT, summarize_species

SCORES = ("gradient boosting, all evidence", "gradient boosting, without effort",
          "gradient boosting, without the Manual", "gradient boosting, without records",
          "logistic regression, all evidence", "nearest record", "system map (rule)")
FEW = 10                      # records known by the cut: the species the flora should help most


def known_records() -> dict[str, int]:
    from mpcr_rag.store import pg_store
    conn = pg_store.connect()
    with conn.cursor() as cur:
        cur.execute("""SELECT e.species, count(*) FROM mpcr.occurrences o
                       JOIN mpcr.species_evidence e ON e.taxon_key = o.species_key
                       WHERE o.year IS NOT NULL AND o.year <= %s GROUP BY e.species""",
                    (CUTS["test"][0],))
        out = dict(cur.fetchall())
    conn.close()
    return out


def main() -> None:
    from scipy.stats import wilcoxon
    rows = [json.loads(l) for l in io.open(OUT / "per_species.jsonl", encoding="utf-8")]
    n_known = known_records()
    species = sorted({r["species"] for r in rows})
    counts = np.array([n_known.get(s, 0) for s in species])
    edges = np.quantile(counts, [1 / 3, 2 / 3])
    tert = {s: int(np.searchsorted(edges, c, side="right")) for s, c in zip(species, counts)}
    labels = []
    for t in range(3):
        c = counts[[tert[s] == t for s in species]]
        labels.append(f"T{t + 1}: {c.min()}-{c.max()} records (median {int(np.median(c))}, "
                      f"{len(c)} species)")
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["score"]][r["species"]] = r
    result = {"edges": edges.tolist(), "tertiles": labels, "scores": {}}
    print(f"{'score':40s}" + "".join(f"{'T' + str(t + 1) + ' AUC':>9s}{'lift':>6s}" for t in range(3)))
    for name in SCORES:
        line, res = f"{name:40s}", []
        for t in range(3):
            s = summarize_species([r for r in by[name].values() if tert[r["species"]] == t])[name]
            line += f"{s['median_auc']:9.3f}{s['pooled_lift']:6.1f}"
            res.append(s)
        result["scores"][name] = res
        print(line)
    for t, lab in enumerate(labels):
        print(lab)
    for drop in ("gradient boosting, without the Manual", "gradient boosting, without records"):
        for t in range(3):
            sp = [s for s in species if tert[s] == t]
            d = [by["gradient boosting, all evidence"][s]["auc"] - by[drop][s]["auc"] for s in sp]
            p = wilcoxon(d).pvalue
            print(f"  T{t + 1}: all evidence minus {drop.split(', ')[1]}: median AUC "
                  f"{np.median(d):+.3f}, mean {np.mean(d):+.3f}, Wilcoxon p = {p:.2g}")
            result.setdefault("gain", {}).setdefault(drop, []).append(
                {"median": float(np.median(d)), "mean": float(np.mean(d)), "p": float(p)})
    few = [s for s in species if n_known.get(s, 0) <= FEW]
    d = np.array([by["gradient boosting, all evidence"][s]["auc"]
                  - by["gradient boosting, without the Manual"][s]["auc"] for s in few])
    print(f"  at most {FEW} records ({len(few)} species): Manual adds median AUC "
          f"{np.median(d):+.3f}, mean {d.mean():+.3f}; more than 0.05 for {np.mean(d > .05):.0%}, "
          f"loses more than 0.05 for {np.mean(d < -.05):.0%}; Wilcoxon p = {wilcoxon(d).pvalue:.2g}")
    result["few"] = {"max_records": FEW, "species": len(few), "median": float(np.median(d)),
                     "mean": float(d.mean()), "share_gain": float(np.mean(d > .05)),
                     "share_loss": float(np.mean(d < -.05)), "p": float(wilcoxon(d).pvalue)}
    (OUT / "tertiles.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
