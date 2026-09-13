# Vector backend parity: Pinecone vs pgvector

*Generated 2026-09-13 — model `intfloat/multilingual-e5-large` (local) vs Pinecone hosted `multilingual-e5-large`; k=10; restricted to 3159 species with identical distribution text AND filterable metadata in both indexes.*

| Check | Pinecone | pgvector |
|---|---|---|
| Self-retrieval @1 (n=200) | 100.0% | 100.0% |
| Self-retrieval @10 | 100.0% | 100.0% |

| Agreement between backends | Self queries | Filtered NL queries |
|---|---|---|
| Mean Jaccard@10 | 0.924 | 0.924 |
| Top-1 agreement | 100.0% | 92.9% |

## Filtered NL queries

| Query | Filters | Jaccard@10 | Top-1 agree | n (P / G) |
|---|---|---|---|---|
| bosque nuboso de altura en la Cordillera de Talamanca | `{}` | 1.00 | yes | 10 / 10 |
| arbustos de tierras bajas | `{'habit': 'arbusto', 'elev_lo': 150, 'elev_hi': 300}` | 0.82 | yes | 10 / 10 |
| bosque de altura, robledales | `{'elev_lo': 2000, 'vertiente': 'Pacífico', 'endemic': True}` | 1.00 | no | 10 / 10 |
| palmas del Pacífico | `{'habit': 'palma', 'vertiente': 'Pacífico'}` | 1.00 | yes | 5 / 5 |
| epífitas de bosque pluvial | `{'habit': 'epífita', 'forest_type': 'pluvial'}` | 1.00 | yes | 10 / 10 |
| árboles de bosque seco en Guanacaste | `{'habit': 'árbol', 'forest_type': 'seco'}` | 1.00 | yes | 10 / 10 |
| hierbas de páramo | `{'habit': 'hierba', 'elev_lo': 3000}` | 1.00 | yes | 10 / 10 |
| especies de manglar y zonas costeras | `{'elev_hi': 50}` | 0.82 | yes | 10 / 10 |
| Lauraceae de la vertiente Caribe | `{'family': 'Lauraceae', 'vertiente': 'Caribe'}` | 0.82 | yes | 10 / 10 |
| plantas que florecen en enero en la Península de Osa | `{'flowering_month': 1}` | 1.00 | yes | 10 / 10 |
| bejucos de bosque muy húmedo | `{'habit': 'bejuco', 'forest_type': 'muy húmedo'}` | 0.82 | yes | 10 / 10 |
| endémicas de la Cordillera Central | `{'region': 'Cordillera Central', 'endemic': True}` | 1.00 | yes | 10 / 10 |
| orillas de caminos y potreros | `{}` | 1.00 | yes | 10 / 10 |
| sotobosque de bosque primario | `{'elev_hi': 800}` | 0.67 | yes | 10 / 10 |
