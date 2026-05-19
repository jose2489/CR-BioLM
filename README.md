# CR-BioLM: Pipeline Multimodal de Modelado de Distribución de Especies para Costa Rica

CR-BioLM es una arquitectura modular de investigación basada en **LangGraph** que integra 
**registros de presencia GBIF**, **variables bioclimáticas WorldClim**, y **modelos de 
lenguaje (LLM)** para generar análisis ecológicos automáticos de plantas de Costa Rica.

El sistema entrena un modelo Random Forest por especie, lo explica con SHAP y LIME, 
genera visualizaciones espaciales, y consulta un LLM configurable vía OpenRouter para 
producir un perfil ecológico estructurado.

---

## Arquitectura General

```
Fuentes de Datos                  Pipeline LangGraph                    Salidas
─────────────────                 ──────────────────                    ───────

GBIF (Mesoamérica) ──┐
WorldClim (6 bio)  ──┼──► N01: LoadGeometry
DEM (altitud CR)   ──┤    N02: FetchGBIF
Límites CR + Meso  ──┘    N03: LoadClimate
                          N04: LoadHabitatMap (opcional)
                          N05: EnrichAltitude
                               │
                               ▼
                          N06: BuildMatrix
                               │
                               ▼
                          N07: TrainRandomForest
                               │
                               ▼
                          N08: Predict
                               │
                               ▼
                          N09: ExplainSHAP
                          N10: ExplainLIME
                               │
                               ▼
                          N11: ExportMaps ──────────────────► mapas .png
                               │
                               ▼
                          N12: ConservationContext
                               │
                               ▼
                          N13: GenerateReport ────────────► perfil .txt
                               │                           
                               ▼
                             FIN
```

### Nodos del Pipeline (LangGraph)

| Grupo | Nodo | Responsabilidad |
|-------|------|----------------|
| Ingesta | N01 LoadGeometry | Límites CR + Mesoamérica |
| Ingesta | N02 FetchGBIF | Ocurrencias de la especie |
| Ingesta | N03 LoadClimate | Variables bioclimáticas WorldClim |
| Ingesta | N04 LoadHabitatMap | Mapa botánico (opcional) |
| Transformación | N05 EnrichAltitude | Altitud desde DEM |
| Transformación | N06 BuildMatrix | Matriz ambiental + pseudo-ausencias |
| Modelado | N07 TrainRandomForest | Entrenamiento y métricas |
| Modelado | N08 Predict | Predicciones y probabilidades |
| Explicabilidad | N09 ExplainSHAP | Importancia global de variables |
| Explicabilidad | N10 ExplainLIME | Explicación local por instancia |
| Salidas | N11 ExportMaps | Mapas de distribución y métricas |
| Salidas | N12 ConservationContext | Cruce con ASPs, humedales, corredores |
| Salidas | N13 GenerateReport | Reporte LLM vía OpenRouter |

---

## Prerequisitos

- Python 3.10 o superior
- Una clave de API de **OpenRouter** (`OPENROUTER_API_KEY`)
- Archivos de datos geoespaciales en `data_raw/` (ver abajo)

## Instalación

```bash
cd CR-BioLM
pip install -r requirements.txt

# Instalar Dependencias adicionales
pip install langgraph pandas geopandas dotenv pygbif rasterio scikit-learn shap lime seaborn
```

Crea un archivo `.env` con tu clave de OpenRouter:
OPENROUTER_API_KEY=tu_llave_aqui

### Datos geoespaciales requeridos (`data_raw/`)

Los archivos geoespaciales no se distribuyen en este repositorio por tamaño y licencias.
Contactar al autor en jose2489@gmail.com para obtenerlos.

| Archivo | Descripción |
|---------|-------------|
| `topography/altitud_cr.tif` | DEM de Costa Rica (Int16, EPSG:4326) |
| `climate/wc2.1_30s_bio_*.tif` | Variables bioclimáticas WorldClim (19 capas) |
| `vectors/areas_protegidas_cr.shp` | Áreas Silvestres Protegidas de CR |
| `vectors/humedales_cr.shp` | Humedales de CR |
| `vectors/corredores_biologicos_cr.shp` | Corredores Biológicos de CR |
| `vectors/areas_conservacion_cr.shp` | Áreas de Conservación de CR |

---

## Configuración del Modelo LLM

El modelo LLM se configura en **`nodes/config_llm.py`**:

```python
LLM_MODEL = "openrouter/free"   # Modelo a usar
LLM_IS_MULTIMODAL = False       # True → adjunta imágenes al prompt
```

---

## Uso

### Especie individual
```bash
python workflow.py -s "Quercus costaricensis"
```

### Con pregunta para el LLM
```bash
python workflow.py -s "Quercus costaricensis" -q "¿Cómo le afecta el cambio climático?"
```

### Con pregunta del banco por perfil de usuario
```bash
python workflow.py -s "Quercus costaricensis" --persona botanico
python workflow.py -s "Quercus costaricensis" --persona municipalidad --canton Nicoya
python workflow.py -s "Quercus costaricensis" --persona municipalidad --canton Nicoya --proyecto "proyecto hidroeléctrico"
```

### Modo batch (múltiples especies)
```bash
python workflow.py -f especies.txt
python workflow.py -f especies.txt -q "¿Cuál es el rango altitudinal óptimo?"
python workflow.py -f especies.txt --persona turista
```

El archivo `especies.txt` contiene un nombre científico por línea:
```
Quercus costaricensis
Myrsine coriacea
Schlegelia parviflora
# Esta línea es un comentario y se ignora
Bertiera bracteosa
```

> **Nota batch + `--persona`**: cada especie recibe una pregunta aleatoria independiente del banco.  
> **Nota batch + `-q`**: todas las especies reciben la misma pregunta fija.

---
Por cada ejecución se crea `outputs/{Especie}/run_{timestamp}/` con:

| Archivo | Descripción |
|---------|-------------|
| `mapa_distribucion_mesoamerica.png` | Distribución regional (entrenamiento RF) |
| `shap_summary.png` | Importancia global de variables (SHAP) |
| `lime_local_explanation.png` | Explicabilidad local (LIME) |
| `matriz_confusion.png` | Evaluación del modelo Random Forest |
| `llm_profile_BIMODAL_*.txt` | Perfil ecológico generado por LLM |
| `n01_load_geometry.json` | Salida estructurada de cada nodo |
| `n02_fetch_gbif.json` | ↑ |
| `...` | ↑ (un JSON por nodo) |

---

## Estructura del Proyecto
CR-BioLM/
├── workflow.py              # Punto de entrada + grafo LangGraph
├── config.py                # Configuración global (paths, seed)
├── nodes/
│   ├── graph_state.py       # Estado compartido del pipeline
│   ├── config_llm.py        # Configuración del modelo LLM ← editar aquí
│   ├── n01_load_geometry.py
│   ├── n02_fetch_gbif.py
│   ├── n03_load_climate.py
│   ├── n04_load_habitat_map.py
│   ├── n05_enrich_altitude.py
│   ├── n06_build_matrix.py
│   ├── n07_train_random_forest.py
│   ├── n08_predict.py
│   ├── n09_explain_shap.py
│   ├── n10_explain_lime.py
│   ├── n11_export_maps.py
│   ├── n12_conservation_context.py
│   └── n13_generate_report.py
├── data/                    # Módulos de carga de datos
├── models/                  # RandomForestSDM
├── xai/                     # SHAP, LIME, GradCAM
├── llm/                     # OpenRouterClient
└── utils/                   # Visualizer, geoprocesamiento

---

## Cita / Referencia

> Araya, J. (2026). *CR-BioLM: Pipeline multimodal de modelado de distribución de 
> especies de plantas de Costa Rica usando Machine Learning e Inteligencia Artificial 
> Generativa*. Tesis de Maestría.