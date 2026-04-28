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
5. Continuar a `cr-biolm-frontend/README.md`