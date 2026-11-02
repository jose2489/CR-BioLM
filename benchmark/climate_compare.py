"""The climate a person would get by asking a model what a plant prefers, against the
system's climate profile. A separate experiment from the benchmark, added after the
confirmatory run, and reported as such.

It is the map comparison (mapcompare.py) with climate in place of geography. Someone
asks, in plain Spanish, what climate a species prefers in Costa Rica. A model answers in
prose; the temperature and rainfall it states are, in effect, a climate envelope, and
the same two numbers judge it as judge a map:

    captured   the share of the species' records whose 1 km pixel falls inside the
               envelope (WorldClim 2.1 annual mean temperature, bio1, and annual
               precipitation, bio12)
    area       the share of the country's land inside the envelope

The system's answer is the profile of benchmark/climate_profile.py, whose stated ranges
are the 5th to 95th percentile of the records. Scored on the records it was built from
it would capture 90% by construction, so it is built from one random half of the
records and scored on the other half (both folds, the halves of map_refine). The models
are scored on all records, since they never see them.

Only the two variables every source can be compared on are scored: a model asked about
climate gives a temperature and a rainfall, if it gives numbers at all. An answer with
no numeric range for either is counted, not scored, as a description that names no
region is in the map comparison. A single number with no range ("unos 15 °C") is not a
range and is counted the same way; an open bound ("más de 2.000 mm") is a range open on
one side.

The species are the map comparison's 300 (benchmark/data/maps_sample.json): at least 30
records, disjoint from the benchmark.

Run:  python -u -m benchmark.climate_compare --generate --models paid,open,paid_manual,open_manual
      python -u -m benchmark.climate_compare --report [--reference chelsa]
      python -u -m benchmark.climate_compare --sheet
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import math
import random
import re
import statistics
from pathlib import Path

import numpy as np

from . import arms as A
from . import client, prompts
from . import climate_profile as C
from . import map_refine as R

DATA = Path(__file__).resolve().parent / "data"
QUESTION = ("¿Qué clima prefiere {sp} en Costa Rica? Indíqueme la temperatura media anual "
            "y la lluvia anual de los sitios donde crece.")
MODELS = {"open": A.OPEN, "paid": A.PAID,
          # The same models given the Manual's entry (citation, distribution and habitat,
          # elevation: the climate pack at level "flora") and not the profile. The Manual
          # describes climate only through forest types and elevation; these say how much
          # of the profile a model can infer from them. Free prompt, as DP and FP: the
          # careful one forbids adding what the evidence lacks, and degrees and
          # millimetres are exactly what it lacks.
          "open_manual": A.OPEN, "paid_manual": A.PAID}
ANSWERS = DATA / "answers_climateq.jsonl"


def _scores_path() -> Path:
    suffix = "" if C.REFERENCE == "worldclim" else f"_{C.REFERENCE}"
    return DATA / f"climate_compare_scores{suffix}.jsonl"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def _species() -> list[dict]:
    return json.loads((DATA / "maps_sample.json").read_text(encoding="utf-8"))["sample"]


# ------------------------------------------------------------------------ generation

def generate(models: list[str], max_cost: float, workers: int) -> None:
    done = {(r["species"], r["model"]) for r in _jsonl(ANSWERS) if not r.get("error")}
    jobs = [(item, m) for item in _species() for m in models
            if (item["species"], m) not in done]
    print(f"{len(jobs)} calls to make ({len(done)} already on disk)", flush=True)

    manual: dict[str, str] = {}
    if any(m.endswith("_manual") for _i, m in jobs):
        from mpcr_rag.store import pg_store
        from .evidence_pack import build as build_pack
        conn = pg_store.connect()
        with conn.cursor() as cur:
            for sp in sorted({i["species"] for i, m in jobs if m.endswith("_manual")}):
                manual[sp] = build_pack(cur, sp, "climate", level="flora").text()
        conn.close()

    def call(job):
        item, m = job
        spec = MODELS[m]
        base = {"species": item["species"], "model": m, "stratum": item["stratum"]}
        try:
            res = client.call(spec, [{"role": "system", "content": prompts.MINIMAL},
                                     {"role": "user",
                                      "content": prompts.user_message(
                                          QUESTION.format(sp=item["species"]),
                                          manual[item["species"]])
                                      if m.endswith("_manual")
                                      else QUESTION.format(sp=item["species"])}],
                              pin=A.PIN.get(spec, ()))
            return {**base, "answer_text": res["answer_text"], "cost_usd": res.get("cost_usd"),
                    "served_by": res.get("served_by"), "error": None}
        except Exception as e:
            return {**base, "answer_text": None, "error": f"{type(e).__name__}: {e}"}

    spent = 0.0
    with io.open(ANSWERS, "a", encoding="utf-8") as fh, cf.ThreadPoolExecutor(workers) as pool:
        for i, row in enumerate(pool.map(call, jobs), 1):
            spent += float(row.get("cost_usd") or 0)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ${spent:.3f}", flush=True)
            if spent > max_cost:
                raise SystemExit(f"cost ceiling ${max_cost:.2f} hit; re-run to resume")


# --------------------------------------------------------------------------- parsing

_N = r"(\d{1,2}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)"


def _num(s: str, thousands: bool) -> float:
    """'1.500' and '1,500' are fifteen hundred for rainfall; '17,5' is a decimal for
    temperature. The unit decides which reading applies."""
    if thousands and re.fullmatch(r"\d{1,2}[.,]\d{3}(?:[.,]\d{3})*", s):
        return float(re.sub(r"[.,]", "", s))
    return float(s.replace(",", "."))


# A short aside may sit between the ends: "entre 18 °C (en las partes altas) y 26 °C".
_RANGE = r"{n}\s*(?:{u})?\s*(?:\([^()]{{0,80}}\))?\s*(?:-|–|—|a|y|hasta)\s*{n}"
_OPEN_LO = (r"(?:más de|(?<!no )supera(?:n|r)?|superior(?:es)? a|por encima de|mayor(?:es)? (?:a|de)"
            r"|arriba de|excede(?:n)?|sobrepasa(?:n)?)(?:\s+(?:los|las|el|la))?\s*{n}")
_OPEN_HI = (r"(?:menos de|inferior(?:es)? a|por debajo de|menor(?:es)? (?:a|de)|hasta|no supera(?:n)?)"
            r"(?:\s+(?:los|las|el|la))?\s*{n}")


def _ranges(text: str, unit: str, thousands: bool) -> tuple[float, float] | None:
    """The widest range stated for one quantity. Only numbers followed, within the same
    phrase, by the unit count, so an elevation in metres is never read as rainfall."""
    lo, hi = math.inf, -math.inf
    found = False
    tail = rf"(?=[^\d;:\n°]{{0,25}}?{unit})"     # no other number or unit in between
    # The first number may carry only the same unit, so "18 °C y 2.000 mm" is not read
    # as a rainfall range of 18 to 2,000.
    for m in re.finditer(_RANGE.format(n=_N, u=unit) + tail, text, re.I):
        a, b = _num(m.group(1), thousands), _num(m.group(2), thousands)
        lo, hi, found = min(lo, a, b), max(hi, a, b), True
    for m in re.finditer(_OPEN_LO.format(n=_N) + tail, text, re.I):
        lo, hi, found = min(lo, _num(m.group(1), thousands)), math.inf, True
    for m in re.finditer(_OPEN_HI.format(n=_N) + tail, text, re.I):
        if not found:
            lo = -math.inf
        hi, found = max(hi, _num(m.group(1), thousands)) if hi != math.inf else hi, True
    return (lo, hi) if found else None


def parse(text: str) -> dict:
    t = (text.replace(" ", " ").replace(" ", " ").replace("\xa0", " ")
         .replace("**", ""))
    # "2 500 mm": a space as the thousands separator, as the frontier model writes it
    t = re.sub(r"\b(\d{1,2}) (\d{3})\b", r"\1\2", t)
    temp = _ranges(t, r"(?:°\s*C|ºC|grados)", thousands=False)
    rain = _ranges(t, r"(?:mm|milímetros)", thousands=True)
    # Ranges outside what Costa Rica can hold are a misreading, not an envelope.
    if temp and not (-5 <= temp[0] <= 40 or temp[0] == -math.inf):
        temp = None
    if rain and rain[1] != math.inf and rain[1] < 200:
        rain = None
    return {"bio1": temp, "bio12": rain}


# --------------------------------------------------------------------------- scoring

def env_mask(env: dict) -> np.ndarray:
    c = C.climate()
    m = np.ones(len(c["bio1"]), dtype=bool)
    for v, rng in env.items():
        if rng is None:
            continue
        x = c[v]
        m &= ~np.isnan(x) & (x >= rng[0]) & (x <= rng[1])
    return m


def _binom_p(k: int, n: int, p: float) -> float:
    """One-sided P(X >= k) for X ~ Binomial(n, p): does the envelope hold more records
    than its share of the country would by chance?"""
    from scipy.stats import binomtest
    if n == 0 or p <= 0:
        return 1.0
    return binomtest(k, n, min(p, 1.0), alternative="greater").pvalue


def score() -> list[dict]:
    from mpcr_rag.store import pg_store
    answers = {(r["species"], r["model"]): r for r in _jsonl(ANSWERS) if not r.get("error")}
    conn = pg_store.connect()
    rows = []
    with conn.cursor() as cur:
        for i, item in enumerate(_species(), 1):
            sp = item["species"]
            lonlat, _ = R._records(cur, sp)
            pix = R.record_pixels(lonlat)
            pix = pix[pix >= 0]
            n = len(pix)
            # ours, held out: the profile's 5th-95th percentile ranges from one half
            rng = random.Random(f"{R.SEED}-{sp}")
            order = list(range(n))
            rng.shuffle(order)
            half = n // 2
            cap, area = 0, []
            band = C.in_band(cur, sp, pix)
            for train, test in ((order[:half], order[half:]), (order[half:], order[:half])):
                c = C.climate()
                # built as the answer states it (records in the Manual's elevation band),
                # scored on every held-out record, in band or not
                train = [i for i in train if band[i]] or train
                env = {}
                for v in ("bio1", "bio12"):
                    x = c[v][pix[train]]
                    x = x[~np.isnan(x)]
                    env[v] = (float(np.quantile(x, .05)), float(np.quantile(x, .95)))
                mk = env_mask(env)
                cap += int(mk[pix[test]].sum())
                area.append(float(mk.mean()))
            a = float(np.mean(area))
            rows.append({"species": sp, "source": "ours", "n": n, "has_env": True,
                         "capture": cap / max(1, n), "area": a,
                         "p": _binom_p(cap, n, a)})
            for m in MODELS:
                ans = answers.get((sp, m))
                if not ans:
                    continue
                env = parse(ans["answer_text"] or "")
                has = env["bio1"] is not None and env["bio12"] is not None
                row = {"species": sp, "source": m, "n": n, "has_env": has,
                       "has_temp": env["bio1"] is not None, "has_rain": env["bio12"] is not None,
                       "env": {k: list(v) if v else None for k, v in env.items()}}
                if has:
                    mk = env_mask(env)
                    k = int(mk[pix].sum())
                    row.update({"capture": k / max(1, n), "area": float(mk.mean()),
                                "p": _binom_p(k, n, float(mk.mean()))})
                rows.append(row)
            if i % 50 == 0:
                print(f"  {i}/{len(_species())}", flush=True)
    conn.close()
    with io.open(_scores_path(), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return rows


def report() -> None:
    rows = score()
    print(f"\n{len(_species())} species; median per species over those with an envelope")
    print(f"{'source':6s} {'envelope':>9s} {'capture':>8s} {'area':>6s} {'lift':>5s} "
          f"{'beats chance':>13s}")
    for s in ("ours", "paid_manual", "open_manual", "paid", "open"):
        sub = [r for r in rows if r["source"] == s]
        if not sub:
            continue
        ok = [r for r in sub if r["has_env"]]
        if not ok:
            print(f"{s:6s} {0:4d}/{len(sub)}")
            continue
        lift = [r["capture"] / r["area"] for r in ok if r["area"] > 0]
        print(f"{s:6s} {len(ok):4d}/{len(sub):<4d} {100 * statistics.median(r['capture'] for r in ok):7.0f}%"
              f" {100 * statistics.median(r['area'] for r in ok):5.1f}%"
              f" {statistics.median(lift) if lift else float('nan'):5.1f}"
              f" {sum(r['p'] < .05 for r in ok):6d}/{len(ok)}")
        if s != "ours":
            print(f"       temperature only {sum(r['has_temp'] and not r['has_rain'] for r in sub)}, "
                  f"rainfall only {sum(r['has_rain'] and not r['has_temp'] for r in sub)}, "
                  f"neither {sum(not r['has_temp'] and not r['has_rain'] for r in sub)}")


SOURCE_ES = {"paid_manual": "Mistral + Manual", "open_manual": "Gemma + Manual",
             "paid": "Mistral libre", "open": "Gemma libre"}


def _fmt_range(e: dict | None) -> str:
    if not e:
        return ""

    def one(v):
        if not v:
            return "sin rango"
        lo, hi = float(v[0]), float(v[1])
        f = lambda x: f"{x:,.0f}".replace(",", ".")
        return (("<" if math.isinf(lo) else f(lo)) + "–" + (">" if math.isinf(hi) else f(hi)))
    return f"T {one(e.get('bio1'))} °C; P {one(e.get('bio12'))} mm"


def sheet() -> Path:
    """One row per species, for reading: the profile (WorldClim), what each source said
    and its capture and area against both climatologies. Written to benchmark/data,
    which is not under version control."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from mpcr_rag.store import pg_store
    scores = {}
    for ref in C.SOURCES:
        C.REFERENCE = ref
        for r in _jsonl(_scores_path()):
            scores[(r["species"], r["source"], ref)] = r
    C.REFERENCE = "worldclim"
    answers = {(r["species"], r["model"]): r["answer_text"] for r in _jsonl(ANSWERS)
               if not r.get("error")}
    wb = Workbook()
    ws = wb.active
    ws.title = "Clima por especie"
    head = ["Especie", "Ambiente", "Registros usados", "Fuera de banda", "Temp. media (°C)",
            "Lluvia anual (mm)", "Estacionalidad (CV %)", "Lluvia trim. seco (mm)",
            "Zonas de vida principales"]
    sources = ["ours"] + list(SOURCE_ES)
    for s in sources:
        name = "Nuestro" if s == "ours" else SOURCE_ES[s]
        if s != "ours":
            head.append(f"{name}: rango")
        for ref in C.SOURCES:
            head += [f"{name}: captura ({ref})", f"{name}: área ({ref})"]
    head += [f"Respuesta {SOURCE_ES[s]}" for s in SOURCE_ES] + ["Perfil (texto del sistema)"]
    ws.append(head)
    conn = pg_store.connect()
    with conn.cursor() as cur:
        for item in _species():
            sp = item["species"]
            p = C.profile(cur, sp)
            v = p["vars"]
            if not v:
                continue
            t, dry = v["bio1"]["median"], v["bio17"]["median"]
            amb = ("montaña" if t < 18 else "intermedio" if t < 24
                   else "bajas secas" if dry < 100 else "bajas húmedas")
            tot = sum(p["life_zones"].values()) or 1
            zones = "; ".join(f"{C._accents(z.lower())} {100 * n / tot:.0f}%"
                              for z, n in list(p["life_zones"].items())[:3])
            rng = lambda k: f"{v[k]['p05']:,.0f}–{v[k]['p95']:,.0f}".replace(",", ".")
            row = [sp, amb, p["n"], p["n_out_of_band"], rng("bio1"), rng("bio12"),
                   rng("bio15"), rng("bio17"), zones]
            for s in sources:
                if s != "ours":
                    row.append(_fmt_range(scores.get((sp, s, "worldclim"), {}).get("env")))
                for ref in C.SOURCES:
                    r = scores.get((sp, s, ref), {})
                    row += [r.get("capture"), r.get("area")]
            row += [answers.get((sp, s), "") for s in SOURCE_ES] + [C.profile_text(p)]
            ws.append(row)
    conn.close()
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="2F5597")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for col in ws.columns:
        h = col[0].value or ""
        wide = h.startswith(("Respuesta", "Perfil"))
        ws.column_dimensions[col[0].column_letter].width = (
            60 if wide else 40 if h.startswith("Zonas") else 26 if "rango" in h
            or h == "Especie" else 11)
        for c in col[1:]:
            c.alignment = Alignment(wrap_text=True, vertical="top")
            if "captura" in h or "área" in h:
                c.number_format = "0%"
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions
    out = DATA / "resultados_clima.xlsx"
    try:
        wb.save(out)
    except PermissionError:        # open in Excel, which locks it on Windows
        out = out.with_name("resultados_clima_nuevo.xlsx")
        wb.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--models", default="paid,open")
    ap.add_argument("--max-cost", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--sheet", action="store_true", help="write resultados_clima.xlsx")
    ap.add_argument("--reference", default="worldclim", choices=sorted(C.SOURCES),
                    help="climatology the envelopes are drawn on and scored against")
    args = ap.parse_args()
    C.REFERENCE = args.reference
    if args.generate:
        generate([m.strip() for m in args.models.split(",")], args.max_cost, args.workers)
    if args.report:
        report()
    if args.sheet:
        print(f"wrote {sheet()}")


if __name__ == "__main__":
    main()
