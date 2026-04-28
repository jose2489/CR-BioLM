# CR-BioLM — Frontend

Interfaz web para el sistema CR-BioLM de modelado de distribución de especies en Costa Rica. Construida con React y Vite, se comunica con un backend FastAPI que ejecuta el pipeline completo de análisis geoespacial, machine learning y generación de perfiles ecológicos mediante LLMs.

---

## Descripción

CR-BioLM permite a investigadores y conservacionistas ingresar el nombre científico de una especie costarricense y obtener automáticamente:

- Registros de presencia limpios desde GBIF
- Modelado de distribución con Random Forest y CNN Multimodal
- Explicabilidad espacial con SHAP, LIME y Grad-CAM
- Perfiles ecológicos generados por LLaMA 3.3 70B, Qwen 3 32B y un Baseline Dual visión-texto

---

## Stack tecnológico

- React 18
- Vite
- Tailwind CSS v4
- Zustand (estado global)
- Axios (comunicación con API)

---

## Requisitos previos

- Python 3.10 - 3.13
- Node.js 18 o superior
- El backend CR-BioLM corriendo localmente (ver repositorio principal)
- Copiar los contenidos de `data_raw` desde el repositorio principal a la carpeta `backend/data_raw`

---

## Instalación

```bash
# Desde la raíz del repositorio principal
cd cr-biolm-frontend

# Instalar dependencias
npm install
```

---

## Configuración

El frontend apunta por defecto a `http://127.0.0.1:8000`. Si el backend corre en otro puerto o host, actualice las URLs en:

src/services/api.js
src/pages/Dashboard.jsx

---

## Uso

**1. Iniciar el backend primero** (desde la carpeta `backend/`):

```bash
python -m uvicorn api:app
```

**2. Iniciar el frontend:** (desde la carpeta `cr-biolm-frontend/`):

```bash
npm run dev
```

**3. Abrir en el navegador:**

http://localhost:5173

---

## Pantallas

**Consulta** — Ingrese el nombre científico binomial de la especie y una pregunta opcional para el LLM. Haga clic en Ejecutar para iniciar el pipeline. La consola de estado muestra el progreso en tiempo real.

**Resultados** — Una vez completado el análisis, muestra las visualizaciones geoespaciales (mapa de solapamiento, SHAP, LIME, Grad-CAM) y los perfiles ecológicos generados por cada modelo LLM comparables entre sí.

**Ayuda** — Guía de uso del sistema, descripción de cada componente del pipeline y ejemplos de especies válidas.

---

## Fuentes de datos

| Fuente | Descripción |
|--------|-------------|
| GBIF | Registros de presencia de especies |
| WorldClim V2.1 | Variables bioclimáticas (bio_1–bio_19) y elevación |
| SINAC / SNIT | Áreas silvestres protegidas, corredores biológicos, humedales |
| IUCN / Map of Life | Mapas expertos de rango de distribución |

---

## Estructura del proyecto

cr-biolm-frontend/
├── src/
│   ├── components/
│   │   ├── InputPanel.jsx      # Formulario de consulta
│   │   └── StatusConsole.jsx   # Consola de logs en tiempo real
│   ├── hooks/
│   │   └── useJob.js           # Controlador del ciclo de vida del job
│   ├── pages/
│   │   └── Dashboard.jsx       # Layout principal con las 3 pantallas
│   ├── services/
│   │   └── api.js              # Endpoints del backend
│   └── store/
│       └── usePipelineStore.js # Estado global (Zustand)
├── index.html
├── tailwind.config.js
└── vite.config.js

---

## Limitaciones conocidas

- Las especies deben tener registros de presencia en Costa Rica dentro de GBIF
- El pipeline completo toma entre 5 y 15 minutos por especie
- El manejo de rutas de archivos de salida está optimizado para Windows
- El backend almacena los jobs en `jobs.json` — este archivo se crea automáticamente en el primer uso

## Posibles Problemas Técnicos

- Verificar una versión de Python compatible con el proyecto
- Verificar interferencias de antivirus
