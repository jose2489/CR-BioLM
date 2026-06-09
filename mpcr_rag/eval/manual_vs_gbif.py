"""Manual-vs-GBIF agreement analysis.

Cross-checks the Manual's stated distribution (elevation + vertiente, extracted by
our pipeline) against real GBIF occurrence evidence. Two purposes:
  1. VALIDATION — what fraction of occurrences fall inside the stated range
     (an external check on extraction correctness).
  2. DISCOVERY — species whose occurrences systematically fall OUTSIDE the stated
     range: potential range extensions, under-documentation, or GBIF noise.

Uses DEM altitude at each occurrence (gbif_map), not GBIF's own elevation field.
"""
from __future__ import annotations

from ..query import retriever as R
from ..query import gbif_map
from ..store import local_store
from .. import config

# A spread of reasonably-collected species across families/elevations.
SAMPLE = [
    "Ocotea whitei", "Ocotea insularis", "Ocotea veraguensis", "Nectandra umbrosa",
    "Beilschmiedia pendula", "Aiouea costaricensis", "Cinnamomum brenesii",
    "Brosimum lactescens", "Castilla elastica", "Ficus schippii",
    "Chamaedorea pinnatifrons", "Calyptrogyne ghiesbreghtiana",
    "Calophyllum brasiliense", "Hypericum irazuense", "Banisteriopsis muricata",
]


def _norm(s: str) -> str:
    return gbif_map._norm(s)


def analyse(species: str, conn) -> dict | None:
    f = local_store.get(conn, species.replace(" ", "_"))
    if not f or f.elev_min is None:
        return None
    pts = gbif_map.get_points(species)
    pts = pts[pts["Altitud"].notna()] if not pts.empty else pts
    n = len(pts)
    if n == 0:
        return {"species": species, "family": f.family, "n": 0}

    emin, emax = f.elev_min, f.elev_max
    xlo = f.elev_outlier_min if f.elev_outlier_min is not None else emin
    xhi = f.elev_outlier_max if f.elev_outlier_max is not None else emax
    alt = pts["Altitud"]
    in_core = ((alt >= emin) & (alt <= emax)).mean()
    in_ext = ((alt >= xlo) & (alt <= xhi)).mean()
    below = (alt < xlo).mean()
    above = (alt > xhi).mean()

    # vertiente agreement (points joined to a region carrying a Vertiente)
    vkeys = [_norm(v)[:4] for v in f.vertientes]
    vv = pts["Vertiente"].dropna()
    vmatch = (vv.apply(lambda x: any(k in _norm(x) for k in vkeys)).mean()
              if vkeys and len(vv) else None)

    return {
        "species": species, "family": f.family, "n": n,
        "manual": f"{emin}-{emax}", "in_core": in_core, "in_ext": in_ext,
        "below": below, "above": above, "vmatch": vmatch,
        "elev_p5_p95": (int(alt.quantile(.05)), int(alt.quantile(.95))),
    }


def main() -> None:
    conn = local_store.connect(config.SQLITE_PATH)
    rows = [r for s in SAMPLE if (r := analyse(s, conn))]
    used = [r for r in rows if r.get("n", 0) >= 10]

    print(f"\n{'species':26} {'fam':14} {'n':>4} {'manual':>9} "
          f"{'core%':>6} {'ext%':>6} {'<':>5} {'>':>5} {'vert%':>6} {'GBIF p5-p95':>12}")
    for r in sorted(used, key=lambda x: -x["in_ext"]):
        vm = f"{r['vmatch']*100:.0f}" if r["vmatch"] is not None else "-"
        print(f"{r['species']:26} {r['family'][:14]:14} {r['n']:4} {r['manual']:>9} "
              f"{r['in_core']*100:5.0f} {r['in_ext']*100:5.0f} {r['below']*100:4.0f} "
              f"{r['above']*100:4.0f} {vm:>5} {str(r['elev_p5_p95']):>12}")

    if used:
        mc = sum(r["in_core"] for r in used) / len(used)
        me = sum(r["in_ext"] for r in used) / len(used)
        vm = [r["vmatch"] for r in used if r["vmatch"] is not None]
        print(f"\nAGREGADO (n>=10, {len(used)} spp): "
              f"in_core={mc*100:.0f}%  in_ext={me*100:.0f}%  "
              f"vert_match={sum(vm)/len(vm)*100:.0f}%")
        disc = [r for r in used if r["in_ext"] < 0.7]
        print(f"discrepancias (>30% fuera del rango extendido): "
              f"{[r['species'] for r in disc]}")


if __name__ == "__main__":
    main()
