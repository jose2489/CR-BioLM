"""Does the Manual leak what was collected after the forecast's cut?

The forecast is tested on what was known by 2009, to rank the cells where a species was
first recorded in 2010-2026. Every feature from the records is dated; the Manual is
not. Volumes II, III and VI appeared in 2003-2007 and cannot hold a collection made
after 2009; Volumes V (2010) and IV (2020) could. If a range statement in those two
volumes rests on a find made after the cut, the Manual's features would point at the
very cells the test counts as new.

The direct test removes the part of the Manual's range that the dated records cannot
account for: every region-slope pair the Manual names in which the species had no
record by 2009. What remains is a Manual reduced to what was already collected, which
cannot carry a later find. The system's model is scored with its Manual features (and
the system's map and map profile, which are built from them) recomputed on the reduced
range, on the same candidate cells.

  * If the Manual leaked, test positives would fall in those unrecorded pairs more often
    for the late volumes, and removing the pairs would cost the late volumes more.
  * The early volumes are the control: they cannot leak, so whatever removing the pairs
    costs them is the value of the Manual's knowledge beyond the records, not leakage.

The comparison is a difference in differences, per species, between the two groups.

Writes data/forecast/leakage.json.

Run:  python -m benchmark.forecast_leakage
"""
from __future__ import annotations

import collections
import io
import json
import time

import numpy as np

from . import forecast as F
from . import map_refine as R

PUBLISHED = {"II": 2003, "III": 2003, "VI": 2007, "V": 2010, "IV": 2020, "IV(2)": 2020}
TEST_CUT = F.CUTS["test"][0]


def volumes(species: list[str]) -> list[str]:
    from mpcr_rag.store import pg_store
    conn = pg_store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT species, ficha->>'volume' FROM mpcr.fichas")
        vol = dict(cur.fetchall())
    conn.close()
    return [vol.get(s, "") for s in species]


def main() -> None:
    from scipy.stats import mannwhitneyu, wilcoxon
    from mpcr_rag.store import pg_store
    t0 = time.time()
    species, sets, _rec = F.build()
    tr, te = sets["train"], sets["test"]
    conn = pg_store.connect()
    with conn.cursor() as cur:
        D = F.load(cur)
    conn.close()
    assert D["species"] == species, "cached tables and catalogue disagree; rebuild the forecast"
    G = F.Grid()
    vol = volumes(species)
    late = np.array([PUBLISHED.get(v, 0) > TEST_CUT for v in vol])
    known_vol = np.array([v in PUBLISHED for v in vol])
    print(f"species: {known_vol.sum()} with a known volume, {late.sum()} in volumes "
          f"published after {TEST_CUT} ({time.time()-t0:.0f}s)", flush=True)

    # pre-cut records per species, and the Manual pairs they do not account for
    pre = D["year"] <= TEST_CUT
    attrs_pre = collections.defaultdict(list)
    for s, a, k in zip(D["sp"].tolist(), D["attrs"], pre.tolist()):
        if k:
            attrs_pre[s].append(a)
    pruned_pairs, removed = {}, {}
    for s, ing in enumerate(D["ings"]):
        seen = {(r, sl) for r, sl in attrs_pre.get(s, []) if r and sl}
        keep = [p for p in ing["pairs"] if tuple(p) in seen]
        removed[s] = [tuple(p) for p in ing["pairs"] if tuple(p) not in seen]
        pruned_pairs[s] = keep

    # 1. where the test positives fall: inside a Manual pair with no pre-cut record?
    pos = np.flatnonzero(te["y"])
    cell_pairs = collections.defaultdict(set)            # cell -> region-slope pairs in it
    for c, pid in zip(G.pc.tolist(), G.pair_id.tolist()):
        cell_pairs[c].add(pid)
    share = {"early": [0, 0], "late": [0, 0]}
    for i in pos:
        s, c = int(te["s"][i]), int(te["c"][i])
        if not known_vol[s]:
            continue
        rem = {G.pair_index[f"{r}|{sl}"] for r, sl in removed[s] if f"{r}|{sl}" in G.pair_index}
        kept = {G.pair_index[f"{r}|{sl}"] for r, sl in pruned_pairs[s]
                if f"{r}|{sl}" in G.pair_index}
        here = cell_pairs[c]
        only_unrecorded = bool(here & rem) and not (here & kept)
        g = "late" if late[s] else "early"
        share[g][0] += only_unrecorded
        share[g][1] += 1
    for g, (k, n) in share.items():
        print(f"  {g} volumes: {k} of {n} new occurrences ({k / max(n, 1):.1%}) fall only in "
              f"Manual pairs with no record by {TEST_CUT}")

    # 2. the system's model scored with the Manual reduced to what was recorded
    cols = F._cols(F.SUBSETS["without effort"])
    gb, _lr, _imp = F.train(tr, cols, 3_000_000)
    print(f"  model trained ({time.time()-t0:.0f}s)", flush=True)
    X = te["X"].copy()
    fi = {n: F.FEATURES.index(n) for n in ("manual_pairs", "manual_map", "system_map",
                                            "clim_map_dist")}
    order = np.argsort(te["s"], kind="stable")
    g_s = te["s"][order]
    cuts = np.flatnonzero(np.diff(g_s)) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(g_s)]):
        s = int(g_s[a])
        if not removed[s]:
            continue
        rows = order[a:b]
        cc = te["c"][rows]
        ing = D["ings"][s]
        pm = G.pairs_mask(pruned_pairs[s])
        bm = G.band_mask(ing["elev"])
        X[rows, fi["manual_pairs"]] = G.frac(pm)[cc]
        X[rows, fi["manual_map"]] = G.frac(pm & bm)[cc]
        wide = R.widen({**ing, "pairs": pruned_pairs[s]}, attrs_pre.get(s, []))
        mk = G.pairs_mask(wide["pairs"]) & bm
        X[rows, fi["system_map"]] = G.frac(mk)[cc]
        envs = {v: (F._env(G.clim[v + "_px"][mk]) if mk.any() else None) for v in ("bio1", "bio12")}
        X[rows, fi["clim_map_dist"]] = F._clim_dist(G, envs)[cc]
    scores = {"published": gb.predict_proba(te["X"][:, cols])[:, 1],
              "manual reduced to pre-cut records": gb.predict_proba(X[:, cols])[:, 1]}
    ps = F.per_species(te, scores)
    by = collections.defaultdict(dict)
    for r in ps:
        by[r["score"]][r["species"]] = r
    sp = [s for s in by["published"] if known_vol[s]]
    d = {s: by["published"][s]["auc"] - by["manual reduced to pre-cut records"][s]["auc"] for s in sp}
    dl = {s: by["published"][s]["hits"] - by["manual reduced to pre-cut records"][s]["hits"]
          for s in sp}
    out = {"test_cut": TEST_CUT, "positives_only_in_unrecorded_pairs": share}
    for g, mask in (("early", False), ("late", True)):
        grp = [s for s in sp if late[s] == mask]
        x = [d[s] for s in grp]
        out[g] = {"species": len(grp),
                  "auc_published": float(np.median([by["published"][s]["auc"] for s in grp])),
                  "auc_reduced": float(np.median([by["manual reduced to pre-cut records"][s]["auc"]
                                                  for s in grp])),
                  "auc_cost_median": float(np.median(x)), "auc_cost_mean": float(np.mean(x)),
                  "auc_cost_wilcoxon_p": float(wilcoxon(x).pvalue) if any(x) else 1.0,
                  "top10_hits_cost_mean": float(np.mean([dl[s] for s in grp]))}
        print(f"  {g} volumes ({len(grp)} species): median AUC {out[g]['auc_published']:.3f} -> "
              f"{out[g]['auc_reduced']:.3f}; cost median {np.median(x):+.4f}, mean {np.mean(x):+.4f} "
              f"(p = {out[g]['auc_cost_wilcoxon_p']:.2g}); top-10 hits lost per species, mean "
              f"{out[g]['top10_hits_cost_mean']:+.3f}")
    a = [d[s] for s in sp if late[s]]
    b = [d[s] for s in sp if not late[s]]
    p = mannwhitneyu(a, b).pvalue
    out["late_minus_early_cost"] = {"median": float(np.median(a) - np.median(b)),
                                    "mean": float(np.mean(a) - np.mean(b)), "mannwhitney_p": float(p)}
    print(f"  late minus early, cost of removing unrecorded pairs: median "
          f"{np.median(a) - np.median(b):+.4f}, mean {np.mean(a) - np.mean(b):+.4f} "
          f"(Mann-Whitney p = {p:.2g})")
    n_spec = int(known_vol.sum())
    out["species_in_late_volumes"] = {"late": int(late.sum()), "with_volume": n_spec}
    (F.OUT / "leakage.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {F.OUT / 'leakage.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
