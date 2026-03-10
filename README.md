# CR-BioLM: Pipeline Automatizado de ingesta y generación de respuesta de especies de plantas en Costa Rica con contexto geoespacial

CR-BioLM es una arquitectura modular diseñada para automatizar el generar respuestas biologicas de especies de plantas de Costa Rica con contexto geoespacial. Integra la descarga de datos espaciales (GBIF, BIEN, WorldClim), el entrenamiento de modelos de Machine Learning (Random Forest), explicabilidad espacial (SHAP) y la sintesis de resultados utilizando Inteligencia Artificial Generativa (Llama 3.3, Mixtral, Gemma, etc).

## Prerequisitos del Sistema

Para garantizar la ejecucion correcta de este pipeline, el sistema debe contar con:
1. Python 3.8 o superior.
2. R y Rscript instalados y agregados a las Variables de Entorno (PATH) del sistema operativo.
3. Una clave de API de Groq configurada como variable de entorno (`GROQ_API_KEY`).

## Instalacion y Configuracion

1. Clona el repositorio: `git clone https://github.com/jose2489/CR-BioLM`
2. Instala las dependencias: `pip install -r requirements.txt`
3. Descarga los mapas base (`data_raw`) desde [https://drive.google.com/drive/folders/10jiTbZTVk_1yVn-YLimAWxNzfDQEOb0H?usp=sharing] y colócalos en el folder de data_raw.
4. Crea un archivo `.env` y coloca tu API Key: `GROQ_API_KEY=tu_llave`

## Uso
Para correr el pipeline para una especie, ejecuta:
`python main.py -s "Quercus costaricensis"`
## Instrucciones de Uso

El pipeline esta diseñado para ser ejecutado desde la terminal a traves del archivo main.py. Soporta dos modos de ejecucion:
Modo Individualgit init

Para modelar y analizar una sola especie, utilice el argumento -s (species) seguido del nombre cientifico entre comillas:
Bash

    python main.py -s "Quercus costaricensis"

## Modo por Lotes (Batch)

Para procesar multiples especies de forma desatendida, cree un archivo de texto (ej. lista_especies.txt) con un nombre cientifico por linea y utilice el argumento -f (file):
Bash

    python main.py -f lista_especies.txt

## Resultados Esperados

Por cada especie procesada, el sistema creara dinamicamente una carpeta en el directorio outputs/ etiquetada con la fecha y hora de ejecucion. Dentro de esta carpeta encontrara:

    Mapas de solapamiento espacial en alta resolucion (.png).

    Graficos de impacto global y explicabilidad SHAP (.png).

    Perfiles de nicho ecologico redactados por los diferentes modelos LLM (.txt).