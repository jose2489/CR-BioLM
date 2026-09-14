# Extending the system: evidence providers

How to add a capability (common-name parsing, image identification, a new data
source) so the system can still answer — and still say where every claim came from.

## The rule

Every capability is an **evidence provider**. It turns some input (a Manual entry, a
species name, a place, a photo) into **evidence items** with a fixed shape. The answer
layer only consumes evidence items; it never needs to know how one was produced. A new
provider therefore adds what the system can answer without changing how it answers.

## The evidence item

```json
{
  "claim_type": "common_name",
  "subject":    {"species": "Petiveria alliacea", "taxon_key": 3084785},
  "value":      ["Ajillo", "Zorrillo"],
  "level":      "stated",
  "source":     "MPCR",
  "citation":   "Manual de Plantas de Costa Rica, Tomo IV, p. 245",
  "method":     "common_names@0.1",
  "confidence": "exact",
  "caveat":     ""
}
```

| Field | Meaning |
|---|---|
| `claim_type` | What is asserted. Existing: `elevation_range`, `occurs_in_region`, `vertiente`, `forest_type`, `flowering_month`, `habit`, `endemic`, `morphology_text`. New providers add their own (`common_name`, `trait:flower_color`, `identification`, …) |
| `subject` | The species (Manual name + GBIF accepted key when resolved), or a place/photo for providers that take those |
| `value` | The asserted value, typed per `claim_type` |
| `level` | **stated** (the source says it) · **inferred_data** (derived from occurrences, rasters, a model) · **inferred_relatives** (genus/family-level text) · **none** (nothing supports an answer — the correct output is to abstain) |
| `source` | `MPCR`, `GBIF`, `DEM`, `WORLDCLIM`, `WORLDCOVER`, `IMAGE_MODEL`, … |
| `citation` | Enough for a person to check it: Tomo/page, GBIF download DOI, dataset + version |
| `method` | Provider name and version, so results can be traced after the provider changes |
| `confidence` | `exact` · `estimated` · `insufficient`, or a probability for model outputs |
| `caveat` | Anything that should reduce trust ("collection effort, not abundance") |

This is the MCP envelope (`value, source, citation, confidence, caveat`) plus the three
fields the thesis evaluation needs: `claim_type`, `level`, `method`. The `level` field is
what lets an answer be verified claim by claim and lets the system abstain.

## Where a provider plugs in

| Kind of provider | Input | Plugs into | Examples |
|---|---|---|---|
| **Extractor** | a Manual entry (`RawFicha`) | `mpcr_rag/ingest/field_extractor.py` → a `Ficha` field → SQLite → `pg_store sync` → MCP `get_species` | common names, uses, traits (height, flower color) |
| **Data provider** | a species or a place | a function or table under `mpcr_rag/evidence/` → an MCP tool | occurrences and `species_evidence`, climate at a location, land cover |
| **Input resolver** | a photo, a vernacular name, free text | resolves to candidate species **with confidence**, then the other providers answer | photo identification, common name → species |

## Worked example: common names (bachelor team)

**Where they are.** At the end of the species header, after the protologue and synonyms
(`mpcr_rag.schema.Ficha.common_names` exists but is empty today):

```
Petiveria alliacea L., Sp. pl. 342. 1753. Ajillo, Zorrillo.
Tetrathylacium macrophyllum ... T. costaricense Standl. Lengua de Vaca, Zapote .
Musa coccinea ... A NTORCHA , A NTORCHA DE B RASIL .
Stanhopea costaricensis ... T ORITO BLANCO .
```

**Pitfalls seen in the real text.**
- Synonyms occupy the same position: `B. subpeltata Cogn.` is a synonym, not a name
- Small capitals are OCR'd with split letters: `A NTORCHA DE B RASIL` → `Antorcha de Brasil`
- Names can follow a nomenclatural note: `… Trel., nom. illeg. H ITAVO , I TABO .`
- Roughly a quarter of headers have a name-like tail; most species have no common name

**Contract.**
```python
def extract_common_names(header_block: str, species: str) -> list[str]:
    """Vernacular names stated in the header, normalized (small caps rejoined,
    capitalization fixed, trailing punctuation removed). Empty list when none."""
```

**Acceptance.**
1. A hand-labelled gold set of ≥150 headers, stratified: with names, without names,
   synonyms only, small caps, nomenclatural notes
2. Report precision and recall on it; no synonym may be returned as a name
3. Plug into `field_extractor.extract`, rebuild, and run
   `python -m mpcr_rag.eval.catalog_diff OLD NEW`: only `common_names` may change
4. Then name → species resolution for questions ("¿dónde crece el guarumo?") and a
   `names` section in pgvector become possible — separate steps

## Future: images

Two different problems — keep them apart.

**Illustrations in the Manual.** An extractor: figure → species it depicts (the figure
credit text already names it). Output: `claim_type = "illustration"` with the page.

**User photos ("what is this plant?").** An input resolver:
1. Photo → ranked candidate species with probabilities (a plant-ID service such as
   Pl@ntNet or iNaturalist computer vision, or a vision-language model)
2. Filter to species in the catalog and plausible at the photo's location/elevation if
   known (occurrence evidence can re-rank)
3. The answer is conditional on the identification: *"Probably* X *(0.72). If so, it
   grows at …"*; below a threshold, say the identification is uncertain and list
   candidates instead of answering as if one were certain
4. Evaluation: identification accuracy is measured separately (top-1/top-5 on photos with
   known species, e.g. research-grade iNaturalist observations from Costa Rica), and the
   answer's truthfulness is scored with the identification uncertainty carried through

## Checklist for any new provider

- [ ] Emits evidence items with `level`, `citation`, `method`
- [ ] Has a gold set and reported precision/recall (or accuracy)
- [ ] Catalog rebuild diff shows only the intended fields changed
- [ ] Exposed as an MCP tool or a `get_species` field, with cost level in the docstring
- [ ] States when it cannot answer (`level: "none"` / `confidence: "insufficient"`)
