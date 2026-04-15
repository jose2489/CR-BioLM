# CR-BioLM: Pipeline Multimodal de Modelado de Distribución de Especies para Costa Rica

CR-BioLM es una arquitectura modular de investigación que integra **registros de presencia GBIF**, **variables bioclimáticas WorldClim**, **mapas de hábitat botánico** basados en el *Manual de Plantas de Costa Rica* (Hammel et al.) y **modelos de lenguaje multimodal (LLM)** para generar análisis ecológicos automáticos de plantas de Costa Rica.

El sistema entrena un modelo Random Forest por especie, lo explica con SHAP y LIME, genera dos mapas complementarios, y consulta a uno o más LLMs (GPT-4o, Claude) con ambas imágenes y métricas para producir una respuesta ecológica estructurada.

---

## Arquitectura General

```
GBIF (Mesoamérica) ──► Random Forest ──► SHAP / LIME ──────────────────────────┐
WorldClim (19 bio) ──►                                                          │
                                                                                ▼
Manual de Plantas CR ──► Mapa de Hábitat Botánico ──►  LLM Multimodal (OpenRouter)
Unidades Fitogeográficas    (3 capas: gris/muted/cyan)   GPT-4o + Claude Opus
DEM (altitud) ──────────►  + puntos GBIF en CR                                  │
                                                                                ▼
                                                              Perfil Ecológico (.txt)
```

---

## Prerequisitos

- Python 3.10 o superior
- Una clave de API de **OpenRouter** (`OPENROUTER_API_KEY`)
- Archivos de datos geoespaciales en `data_raw/` (ver abajo)

No se requiere R ni ninguna otra dependencia externa.

## Instalación

```bash
git clone https://github.com/jose2489/CR-BioLM
cd CR-BioLM
pip install -r requirements.txt
```

Crea un archivo `.env` con tu clave de OpenRouter:
```
OPENROUTER_API_KEY=tu_llave_aqui
```

### Datos geoespaciales requeridos (`data_raw/`)

Descarga desde: https://drive.google.com/drive/folders/10jiTbZTVk_1yVn-YLimAWxNzfDQEOb0H?usp=sharing

| Archivo | Descripción |
|---|---|
| `altitud_cr.tif` | DEM de Costa Rica (Int16, EPSG:4326) |
| `unidades_fitogeograficas_cr/` | Shapefile Unidades Fitogeográficas (Hammel 2014) |
| `wc2.1_30s_bio_*.tif` | Variables bioclimáticas WorldClim (19 capas) |
| `ASP_2023/` | Shapefile de Áreas Silvestres Protegidas de CR |

---

## Uso

### Especie individual

```bash
python main.py -s "Quercus costaricensis"
```

### Con pregunta libre para el LLM

```bash
python main.py -s "Quercus costaricensis" -q "¿Cómo le afecta el cambio climático?"
```

### Con pregunta del banco por perfil de usuario

```bash
# Perfil turista (preguntas de experiencia y lugar)
python main.py -s "Guzmania nicaraguensis" --persona turista

# Perfil botánico (variables climáticas, nicho, coherencia con el Manual)
python main.py -s "Guzmania nicaraguensis" --persona botanico

# Perfil municipalidad (impacto territorial, cantón y proyecto específicos)
python main.py -s "Cecropia obtusifolia" --persona municipalidad --canton "Sarapiquí" --proyecto "proyecto hidroeléctrico"

# Municipalidad con cantón/proyecto aleatorio
python main.py -s "Cecropia obtusifolia" --persona municipalidad
```

### Modo batch (múltiples especies)

```bash
# Pregunta fija para todas
python main.py -f lista_especies.txt -q "¿Cuál es el rango altitudinal óptimo?"

# Pregunta aleatoria del banco por especie (ideal para evaluación)
python main.py -f lista_especies.txt --persona botanico
```

El archivo de lista es un `.txt` con un nombre científico por línea.

---

## Salidas por Especie

Por cada ejecución se crea `outputs/{Especie}/run_{timestamp}/` con:

| Archivo | Descripción |
|---|---|
| `mapa_habitat_manual.png` | Mapa de hábitat botánico (gris / muted / cyan + GBIF) |
| `mapa_distribucion_mesoamerica.png` | Distribución regional Mesoamericana (entrenamiento RF) |
| `shap_summary.png` | Importancia global de variables (SHAP) |
| `lime_local_explanation.png` | Explicabilidad local para un punto de alta idoneidad |
| `matriz_confusion.png` | Evaluación del modelo Random Forest |
| `{Especie}_ficha_MdP.txt` | Ficha de referencia del Manual de Plantas (para evaluación) |
| `llm_profile_BIMODAL_openai_gpt_4o.txt` | Análisis ecológico GPT-4o (Razonamiento + Respuesta) |
| `llm_profile_BIMODAL_anthropic_claude_opus_4_5.txt` | Análisis ecológico Claude Opus |

### Formato de respuesta LLM

Cada perfil sigue la estructura:
```
## Razonamiento
• Máximo 5 viñetas cruzando las tres fuentes (mapa botánico, modelo climático, GBIF)

## Respuesta
3-4 oraciones directas respondiendo la pregunta con zonas geográficas concretas.
```

---

## Banco de Preguntas (`utils/question_bank.py`)

28 preguntas organizadas en 3 perfiles:

| Perfil | Foco | Ejemplos |
|---|---|---|
| `turista` | Lugar, experiencia, floración | "¿Puedo verla en Braulio Carrillo?" |
| `botanico` | Nicho climático, variables, coherencia Manual vs modelo | "¿Cuál es la variable más limitante?" |
| `municipalidad` | Impacto territorial, EIA, corredores biológicos | "¿Hay impacto si se aprueba un {proyecto} en {canton}?" |

Las preguntas de municipalidad aceptan `--canton` y `--proyecto` como parámetros; si se omiten, se eligen aleatoriamente de las listas incluidas (82 cantones, 10 tipos de proyecto).

---

## Catálogo de Especies (`outputs/picked_species_enhanced_clean.csv`)

Contiene ~100 especies seleccionadas del *Manual de Plantas de Costa Rica* con:
- Nombre científico, familia, volumen
- Hábitat, tipo de ecosistema
- Rango altitudinal (min/max)
- Notas geográficas (para traducción al shapefile de Unidades Fitogeográficas)
- Número de ocurrencias GBIF disponibles

---

## Cita / Referencia

> Araya, J. (2026). *CR-BioLM: Pipeline multimodal de modelado de distribución de especies de plantas de Costa Rica usando Machine Learning e Inteligencia Artificial Generativa*. Tesis de Maestría.
