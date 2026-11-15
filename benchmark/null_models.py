"""Every map and climate envelope rescored against a collecting-bias null.

The map and climate comparisons (mapcompare.py, climate_compare.py, rq3_split.py) judge
a map by its lift: the share of a species' records it captures over the share of the
country it covers. That null assumes records could fall anywhere. They do not: they sit
along roads, near stations and in the Central Valley, and a map that happens to cover
well-collected ground is credited for it. The forecast already uses a target-group
background for this reason (Phillips et al. 2009); here the maps get the same.

Two nulls besides area, built from the frozen download of all vascular plants:

  sites     the share of distinct collection sites (30 arc-second pixels holding at
            least one record of any species) that the map covers: the capture expected
            if a species' records fell where any plant has been collected
  records   the same, each site weighted by its number of records

For a binary map, capture over the expected share under a null is the
predicted-to-expected ratio of the inside class, which is what the Boyce index
(Boyce et al. 2002; Hirzel et al. 2006) computes per class.

The test of "better than chance" no longer treats records as independent. Records of
one collecting trip are not separate evidence, so capture is resampled by blocks of
about 20 km (rq3_split.BLOCK): a map beats a null when the 5th percentile of the block
bootstrap of its capture lies above that null. A test over distinct sites (a binomial
on the species' pixels rather than its records) is reported too.

Capture is computed exactly as in the published comparisons, so the capture column
reproduces them; only the null and the test change. No model is called: the models'
answers are read from disk.

Writes data/null_models_scores.jsonl (one row per species, analysis and source).

Run:  python -m benchmark.null_models               (score and report)
      python -m benchmark.null_models --report      (report from the saved scores)
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import random
import statistics as st
import zlib
from functools import lru_cache

import numpy as np

from . import climate_compare as CC
from . import climate_profile as C
from . import map_refine as R
from . import mapcompare as M
from . import rq3_split as Q

DATA = M.DATA
OUT = DATA / "null_models_scores.jsonl"
N_BOOT = 2000


# ------------------------------------------------------------------------ the nulls

@lru_cache(maxsize=1)
def background() -> tuple[np.ndarray, np.ndarray]:
    """(site indicator, record count) over the land pixels, from every record."""
    from mpcr_rag.store import pg_store
    conn = pg_store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT lon, lat FROM mpcr.occurrences")
        lonlat = cur.fetchall()
    conn.close()
    pix = R.record_pixels(lonlat)
    pix = pix[pix >= 0]
    counts = np.bincount(pix, minlength=int((R.grid()["index"] >= 0).sum())).astype(float)
    return counts > 0, counts


def expected(mk: np.ndarray | None) -> dict:
    if mk is None:
        return {"area": 0.0, "p_sites": 0.0, "p_records": 0.0}
    sites, counts = background()
    return {"area": float(mk.mean()), "p_sites": float(mk[sites].mean()),
            "p_records": float(counts[mk].sum() / counts.sum())}


# ------------------------------------------------------------------------- the tests

def _blocks(lonlat) -> list[tuple]:
    return [(int(lo // Q.BLOCK), int(la // Q.BLOCK)) for lo, la in lonlat]


def block_bootstrap(inside: np.ndarray, blocks: list[tuple], seed: str) -> np.ndarray:
    """N_BOOT resampled captures, resampling whole blocks with replacement."""
    ids = {b: i for i, b in enumerate(sorted(set(blocks)))}
    bi = np.array([ids[b] for b in blocks])
    k = np.bincount(bi, weights=inside.astype(float), minlength=len(ids))
    n = np.bincount(bi, minlength=len(ids)).astype(float)
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    draw = rng.integers(len(ids), size=(N_BOOT, len(ids)))
    nn = n[draw].sum(1)
    return np.where(nn > 0, k[draw].sum(1) / np.maximum(nn, 1), 0.0)


def site_test(inside: np.ndarray, pix: np.ndarray, p: float) -> dict:
    """Binomial over the species' distinct pixels: a site is inside if any of its
    records is (for a map read from record attributes the two can differ at edges)."""
    from scipy.stats import binomtest
    ok = pix >= 0
    by = collections.defaultdict(bool)
    for q, i in zip(pix[ok], inside[ok]):
        by[int(q)] |= bool(i)
    n, k = len(by), sum(by.values())
    pval = binomtest(k, n, min(max(p, 1e-12), 1.0), alternative="greater").pvalue if n else 1.0
    return {"site_k": k, "site_n": n, "site_pval": float(pval)}


def row(sp: str, analysis: str, source: str, inside: np.ndarray, lonlat, pix,
        mks: list) -> dict:
    """mks: the map(s) the capture was counted against; the nulls are averaged over
    them, as area is in the published comparisons (one per fold for held-out maps)."""
    ex = [expected(m) for m in mks]
    out = {"species": sp, "analysis": analysis, "source": source, "n": int(len(inside)),
           "capture": float(inside.mean()) if len(inside) else 0.0}
    for k in ("area", "p_sites", "p_records"):
        out[k] = float(np.mean([e[k] for e in ex]))
    boot = block_bootstrap(inside, _blocks(lonlat), f"{sp}-{analysis}-{source}")
    out["boot_q05"] = float(np.quantile(boot, .05))
    out["boot_q95"] = float(np.quantile(boot, .95))
    out["n_blocks"] = len(set(_blocks(lonlat)))
    out.update(site_test(inside, np.asarray(pix), out["p_sites"]))
    return out


# --------------------------------------------------------------------------- scoring

def _inside_map(ing: dict, recs: list[tuple]) -> np.ndarray:
    return np.array([M.captured(ing, [r]) == 1 for r in recs], dtype=bool)


def _species():
    return [r["species"] for r in json.loads(
        (DATA / "maps_sample.json").read_text(encoding="utf-8"))["sample"]]


def score() -> list[dict]:
    from mpcr_rag.schema import Ficha
    from mpcr_rag.store import pg_store
    background()
    map_text = collections.defaultdict(dict)
    for r in M._jsonl(DATA / "answers_mapq.jsonl"):
        if not r.get("error") and r.get("answer_text"):
            map_text[r["species"]][r["model"]] = r["answer_text"]
    clim_text = {(r["species"], r["model"]): r["answer_text"]
                 for r in CC._jsonl(CC.ANSWERS) if not r.get("error") and r.get("answer_text")}
    rows: list[dict] = []
    conn = pg_store.connect()
    with conn.cursor() as cur:
        for k, sp in enumerate(_species(), 1):
            cur.execute("SELECT ficha FROM mpcr.fichas WHERE species = %s", (sp,))
            manual = M.from_ficha(Ficha(**cur.fetchone()[0]))
            cur.execute("""SELECT o.lon, o.lat, o.region, o.vertiente, o.elev_dem
                           FROM mpcr.occurrences o
                           JOIN mpcr.species_evidence e ON e.taxon_key = o.species_key
                           WHERE e.species = %s""", (sp,))
            got = cur.fetchall()
            lonlat = [(r[0], r[1]) for r in got]
            recs = [(r[2], None if r[3] is None else ("Caribe" if "arib" in r[3] else "Pacífico"),
                     r[4]) for r in got]
            pix = R.record_pixels(lonlat)

            # 1. maps, all records (Table S3): mapcompare.score
            srcs = {"manual": manual, "full": M.with_records(manual, recs)}
            for m, text in map_text.get(sp, {}).items():
                srcs[m] = M.from_text(text)
            for name, ing in srcs.items():
                mk = M.mask(ing)
                if mk is None or len(recs) < M.MIN_RECORDS:
                    continue
                rows.append(row(sp, "map_all", name, _inside_map(ing, recs), lonlat, pix, [mk]))

            # 2. maps and climate profiles held out (Table 5): rq3_split.score
            band = np.zeros(len(pix), bool)
            band[pix >= 0] = C.in_band(cur, sp, pix[pix >= 0])
            clim = C.climate("worldclim")
            for split in ("random", "spatial"):
                fs = Q.folds(sp, lonlat, split)
                if any(not tr or not te for tr, te in fs):
                    continue
                acc = {v: ([], [], [], [], []) for v in
                       ("map", "clim_records", "clim_map", "clim_5records")}
                for tr, te in fs:
                    ing = M.with_records(manual, [recs[i] for i in tr])
                    mk = M.mask(ing)
                    a = acc["map"]
                    a[0].append(_inside_map(ing, [recs[i] for i in te]))
                    a[1].extend(lonlat[i] for i in te)
                    a[2].extend(pix[i] for i in te)
                    a[3].append(mk if mk is not None else np.zeros(len(clim["bio1"]), bool))
                    trp = [i for i in tr if pix[i] >= 0]
                    tep = [i for i in te if pix[i] >= 0]
                    inb = [i for i in trp if band[i]] or trp
                    five = random.Random(f"{M.SEED}-{sp}-5").sample(inb, min(5, len(inb)))
                    envs = {"clim_records": Q._env(clim, pix[inb]) if inb else None,
                            "clim_5records": Q._env(clim, pix[five]) if five else None,
                            "clim_map": (Q._env(clim, np.flatnonzero(mk))
                                         if mk is not None and mk.any() else None)}
                    for v, e in envs.items():
                        em = CC.env_mask(e) if e else np.zeros(len(clim["bio1"]), bool)
                        a = acc[v]
                        a[0].append(em[pix[tep]] if tep else np.zeros(0, bool))
                        a[1].extend(lonlat[i] for i in tep)
                        a[2].extend(pix[i] for i in tep)
                        a[3].append(em)
                for v, (ins, ll, pp, mks, _) in acc.items():
                    inside = np.concatenate(ins) if ins else np.zeros(0, bool)
                    if len(inside):
                        rows.append(row(sp, f"heldout_{split}", v, inside, ll, pp, mks))

            # 3. climate envelopes (Table S4): climate_compare.score, both climatologies
            ok = pix >= 0
            cpix, cll = pix[ok], [p for p, o in zip(lonlat, ok) if o]
            n = len(cpix)
            for ref in ("worldclim", "chelsa"):
                C.REFERENCE = ref
                c = C.climate()
                rng = random.Random(f"{R.SEED}-{sp}")
                order = list(range(n))
                rng.shuffle(order)
                half = n // 2
                cband = C.in_band(cur, sp, cpix)
                ins, ll, pp, mks = [], [], [], []
                for train, test in ((order[:half], order[half:]), (order[half:], order[:half])):
                    train = [i for i in train if cband[i]] or train
                    env = {}
                    for v in ("bio1", "bio12"):
                        x = c[v][cpix[train]]
                        x = x[~np.isnan(x)]
                        env[v] = (float(np.quantile(x, .05)), float(np.quantile(x, .95)))
                    mk = CC.env_mask(env)
                    ins.append(mk[cpix[test]])
                    ll.extend(cll[i] for i in test)
                    pp.extend(cpix[i] for i in test)
                    mks.append(mk)
                rows.append(row(sp, f"climate_{ref}", "ours", np.concatenate(ins), ll, pp, mks))
                for m in CC.MODELS:
                    text = clim_text.get((sp, m))
                    if not text:
                        continue
                    env = CC.parse(text)
                    if env["bio1"] is None or env["bio12"] is None:
                        continue
                    mk = CC.env_mask(env)
                    rows.append(row(sp, f"climate_{ref}", m, mk[cpix], cll, cpix, [mk]))
            C.REFERENCE = "worldclim"
            if k % 25 == 0:
                print(f"  {k}/300", flush=True)
    conn.close()
    with io.open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


# --------------------------------------------------------------------------- report

def _lift(cap, p):
    return cap / p if p > 0 else None


def report(rows: list[dict]) -> None:
    sites, counts = background()
    print(f"background: {int(sites.sum())} collection sites, {int(counts.sum())} records")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["analysis"], r["source"])].append(r)
    print(f"\n{'analysis':18s} {'source':14s} {'n':>4s} {'capture':>8s} {'area':>6s} "
          f"{'E[sites]':>8s} {'E[recs]':>8s} {'lift_area':>9s} {'lift_site':>9s} "
          f"{'lift_rec':>8s} {'binom_area':>10s} {'boot_area':>9s} {'boot_site':>9s} "
          f"{'site_test':>9s} {'lift<1.5':>8s}")
    for (an, src), sub in groups.items():
        med = lambda key: st.median(r[key] for r in sub)
        la = [x for x in (_lift(r["capture"], r["area"]) for r in sub) if x is not None]
        ls = [x for x in (_lift(r["capture"], r["p_sites"]) for r in sub) if x is not None]
        lr = [x for x in (_lift(r["capture"], r["p_records"]) for r in sub) if x is not None]
        from scipy.stats import binomtest
        binom = sum(binomtest(round(r["capture"] * r["n"]), r["n"], min(max(r["area"], 1e-12), 1),
                              alternative="greater").pvalue < .05 for r in sub if r["n"])
        print(f"{an:18s} {src:14s} {len(sub):4d} {med('capture'):8.0%} {med('area'):6.0%} "
              f"{med('p_sites'):8.0%} {med('p_records'):8.0%} {st.median(la):9.2f} "
              f"{st.median(ls):9.2f} {st.median(lr):8.2f} {binom:10d} "
              f"{sum(r['boot_q05'] > r['area'] for r in sub):9d} "
              f"{sum(r['boot_q05'] > r['p_sites'] for r in sub):9d} "
              f"{sum(r['site_pval'] < .05 for r in sub):9d} "
              f"{sum(x < 1.5 for x in ls):8d}")
    # does the ordering hold under the site null? paired, on common species
    from scipy.stats import wilcoxon
    print("\npaired comparisons of lift over the site null (median difference, Wilcoxon)")
    pairs = [(("map_all", "full"), ("map_all", "paid")), (("map_all", "full"), ("map_all", "open")),
             (("map_all", "manual"), ("map_all", "paid")), (("map_all", "manual"), ("map_all", "open")),
             (("map_all", "full"), ("map_all", "manual")),
             (("heldout_random", "map"), ("map_all", "manual")),
             (("heldout_spatial", "clim_map"), ("heldout_spatial", "clim_records")),
             (("climate_worldclim", "ours"), ("climate_worldclim", "paid")),
             (("climate_worldclim", "ours"), ("climate_worldclim", "open")),
             (("climate_worldclim", "ours"), ("climate_worldclim", "paid_manual")),
             (("climate_worldclim", "ours"), ("climate_worldclim", "open_manual"))]
    for a, b in pairs:
        A = {r["species"]: _lift(r["capture"], r["p_sites"]) for r in groups[a]}
        B = {r["species"]: _lift(r["capture"], r["p_sites"]) for r in groups[b]}
        common = [s for s in A if s in B and A[s] is not None and B[s] is not None]
        if len(common) < 5:
            continue
        d = [A[s] - B[s] for s in common]
        print(f"  {a[0]}/{a[1]} minus {b[0]}/{b[1]}: {st.median(d):+.2f} "
              f"(n={len(common)}, p={wilcoxon(d).pvalue:.2g})")
    # area distribution of the system's map (reviewer Q2)
    full = groups[("map_all", "full")]
    big = [r for r in full if r["area"] > .40]
    print(f"\nsystem map as drawn: area quartiles "
          f"{np.quantile([r['area'] for r in full], [.25, .5, .75]).round(3).tolist()}, "
          f"{len(big)} of {len(full)} cover more than 40% of the country; their median lift "
          f"over area {st.median(r['capture'] / r['area'] for r in big) if big else float('nan'):.2f}, "
          f"over sites {st.median(r['capture'] / r['p_sites'] for r in big) if big else float('nan'):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="report from the saved scores")
    args = ap.parse_args()
    rows = ([json.loads(l) for l in io.open(OUT, encoding="utf-8")] if args.report
            else score())
    report(rows)


if __name__ == "__main__":
    main()
