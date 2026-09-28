"""Live demo of the whole mpcr_rag surface, in the order of docs/DEV_WALKTHROUGH.md.

Eight steps, each printing what a developer would wrap in an endpoint. Steps 1-6 are
free and local; step 7 loads the local embedding model (about 10 s the first time);
step 8 calls OpenRouter and costs a fraction of a cent.

Run:
    python -m mpcr_rag.scripts.demo_walkthrough              # steps 1-7
    python -m mpcr_rag.scripts.demo_walkthrough --with-llm   # adds step 8 (paid)
    python -m mpcr_rag.scripts.demo_walkthrough --species "Dalbergia retusa"
"""
from __future__ import annotations

import argparse
import time


def hr(n: int, title: str, cost: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}   [{cost}]\n{'=' * 78}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="Peltogyne purpurea")
    ap.add_argument("--common-name", default="cortez amarillo")
    ap.add_argument("--with-llm", action="store_true", help="run step 8 (paid API call)")
    a = ap.parse_args()
    sp = a.species

    from mpcr_rag import config
    from mpcr_rag.store import local_store

    conn = local_store.connect(config.SQLITE_PATH)

    hr(1, f"The catalog entry: everything the Manual states about {sp}", "L0 free")
    f = local_store.get(conn, sp.replace(" ", "_"))
    print(f"  species   : {f.species}  ({f.family}, Tomo {f.volume} p.{f.pages})")
    print(f"  elevation : {f.elev_min}-{f.elev_max} m")
    print(f"  slopes    : {', '.join(f.vertientes) or '-'}")
    print(f"  regions   : {', '.join(f.regions[:4])}{' ...' if len(f.regions) > 4 else ''}")
    print(f"  habits    : {f.habits}   endemic: {f.endemic_cr}")
    print(f"  flowering : {f.flowering_months}")
    print(f"  morphology: {(f.morphology or '')[:110]}...")
    print("  -> this dataclass IS the API payload (mpcr_rag/schema.py)")

    hr(2, "Structured search: exhaustive, no ranking, no top-k", "L0 free")
    from mpcr_rag.query.retriever import filter_all
    hits = filter_all(conn, habit="árbol", endemic=True, elev_lo=2000)
    print(f"  endemic trees reaching above 2000 m: {len(hits)}")
    for x in hits[:5]:
        print(f"    {x.species:34} {x.elev_min}-{x.elev_max} m")
    print("  -> use THIS for superlatives ('the one growing highest'), never top-k")

    hr(3, "Occurrence points: cleaned, cached, elevation-tagged", "L1 cached")
    from mpcr_rag.query import gbif_map
    t = time.time()
    pts = gbif_map.get_points(sp)
    print(f"  {len(pts)} points in {time.time() - t:.1f}s (accepted taxonKey, "
          f"uncertainty < 10 km, lat/lon swaps fixed)")
    if len(pts):
        print(f"  columns: {[c for c in pts.columns if c != 'geometry']}")
    print(f"  frozen-snapshot count (citable): {gbif_map.count_only(sp)}")

    hr(4, "The map", "L1 renders a PNG")
    path, n = gbif_map.single_species_map(f)
    print(f"  {path}")
    print(f"  {n} points drawn  -> the API must serve this as a URL, not a path")

    hr(5, "Where can I see it: parks with records, Manual vs record regions", "L0 free")
    from mpcr_rag.evidence.places import where_to_see
    w = where_to_see(sp, limit=4)
    for p in w["confirmed_places"]:
        print(f"    {p['place']:44} {p['n_records']:3} records   "
              f"{p['elev_p05']:.0f}-{p['elev_p95']:.0f} m")
    beyond = [r for r in w["record_regions"] if r not in set(w["manual_regions"])]
    print(f"  Manual regions        : {', '.join(w['manual_regions'][:4])}")
    print(f"  regions beyond Manual : {', '.join(beyond[:4]) or '-'}")
    print("  -> the two kinds of evidence stay separate, and an empty list is never absence")

    hr(6, "Common name -> species, with an explicit decision", "L0 free")
    from mpcr_rag.evidence import vernacular
    for name in ("cocobolo", a.common_name, "guarumo"):
        d = vernacular.resolve_decision(name)
        print(f"  '{name}' -> {d['decision'].upper()}"
              f"{'  = ' + d['species'] if d.get('species') else ''}")
        for c in d["candidates"][:3]:
            print(f"      {c['species']:30} {c['n_records'] or 0:4} CR records  "
                  f"via {','.join(sorted(c['sources']))}")
    print("  -> AMBIGUOUS must reach the UI as a choice; NONE means not in Tomos II-VI")

    hr(7, "Semantic search over the Manual text", "L1 local embeddings, needs Postgres")
    from mpcr_rag.query.retriever import pattern_b
    t = time.time()
    res = pattern_b("bosque nuboso en la Cordillera de Talamanca", top_k=5)
    print(f"  {len(res)} hits in {time.time() - t:.1f}s (first call loads the model)")
    for x, s in res[:5]:
        print(f"    {s:.3f}  {x.species:32} {x.elev_min}-{x.elev_max} m")

    if a.with_llm:
        hr(8, "Full natural-language answer", "L2 paid, OPENROUTER_API_KEY")
        from mpcr_rag.query.answer import answer
        ans = answer(f"¿Dónde puedo observar {sp}?")
        print(f"  mode: {ans['mode']}   map: {ans['map_path'].name}")
        print("  " + ans["text"][:900].replace("\n", "\n  "))
    else:
        print("\n(step 8 skipped: add --with-llm to run the paid natural-language answer)")

    print("\nSame surface as MCP tools:  python -m mpcr_rag.mcp.server   (11 tools)")
    conn.close()


if __name__ == "__main__":
    main()
