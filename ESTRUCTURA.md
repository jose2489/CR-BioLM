# Arquitectura del Proyecto CR-BioLM

El proyecto sigue un patron de diseño modular y orientado a objetos (Strategy Pattern) para asegurar la escalabilidad, el aislamiento de errores y la reproducibilidad academica.

## Arbol de Directorios

BioCR_LLM_V2/
 |
 |-- main.py                    # Orquestador principal del pipeline.
 |-- config.py                  # Configuraciones globales, semillas (seed=42) y rutas.
 |
 |-- data/                      # Modulo de Ingesta y Geoprocesamiento
 |   |-- __init__.py
 |   |-- climate_loader.py      # Gestor de rasters de WorldClim (19 variables).
 |   |-- download_bien.R        # Script puente para descargar poligonos de BIEN.
 |   |-- expert_maps.py         # Cargador de shapefiles y base territorial.
 |   |-- gbif_extractor.py      # Conexion API y limpieza de coordenadas.
 |   |-- geoprocessor.py        # Generacion de pseudo-ausencias y extraccion de pixeles.
 |
 |-- llm/                       # Modulo de Inteligencia Artificial Generativa
 |   |-- __init__.py
 |   |-- base_llm.py            # Interfaz abstracta para modelos de lenguaje.
 |   |-- groq_client.py         # Cliente de conexion API.
 |   |-- prompt_templates.py    # Gestor de instrucciones y reglas de redaccion.
 |
 |-- models/                    # Modulo de Machine Learning
 |   |-- __init__.py
 |   |-- base_model.py          # Interfaz abstracta para algoritmos predictivos.
 |   |-- random_forest.py       # Implementacion de Random Forest SDM.
 |
 |-- xai/                       # Modulo de Explicabilidad
 |   |-- __init__.py
 |   |-- shap_explainer.py      # Implementacion de valores de Shapley.
 |
 |-- utils/                     # Modulo de Utilidades y Graficos
 |   |-- __init__.py
 |   |-- visualizer.py          # Generador de mapas cartograficos (Matplotlib/GeoPandas).
 |
 |-- data_raw/                  # Directorio de almacenamiento temporal (Cache)
 |   |-- climate_rasters/       # Rasters .tif globales y recortados.
 |   |-- expert_maps/           # Shapefiles .shp descargados de BIEN.
 |
 |-- outputs/                   # Directorio final de resultados
     |-- Especie_Nombre/        # Subcarpetas generadas dinamicamente por ejecucion.

Con estos dos doc