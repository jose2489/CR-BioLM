# MPCR Map & Catalog API — Guía de instalación local

Servicios de distribución de plantas de Costa Rica: catálogo estructurado del
*Manual de Plantas de Costa Rica* (5,791 especies), ocurrencias GBIF filtradas, y
mapas de distribución pre-renderizados.

> **Estado: BORRADOR.** Este documento describe el estado objetivo. La API y el
> contenedor están en construcción — ver [SHIP_PLAN.md](SHIP_PLAN.md). El contrato
> (`openapi.json`) se congela primero; pueden empezar a construir contra él antes de
> que exista la implementación.

---

## Lo primero: ¿qué llaves o contraseñas necesito?

**Ninguna.**

Todo lo que ustedes van a consumir (catálogo, ocurrencias, mapas) funciona sin
credenciales:

| Servicio | ¿Necesita llave? | Por qué |
|---|---|---|
| Catálogo de especies | No | SQLite local, viene en el bundle |
| Ocurrencias GBIF | No | La API de búsqueda de GBIF es pública, y además el bundle trae el caché |
| Mapas de distribución | No | Pre-renderizados, se sirven como archivos estáticos |
| Parseo de texto de distribución | No | Es determinístico, sin LLM |

No necesitan cuenta de GBIF, ni de Pinecone, ni de OpenRouter, ni de OpenAI.
**Si alguien les pide una llave para usar esta API, algo está mal — pregunten.**

Solo hay una excepción, y está fuera del alcance de ustedes: las funciones de
búsqueda semántica y respuesta en lenguaje natural (`semantic_search`,
`answer_question`) sí requieren llaves pagadas. Esas **no** forman parte de esta
entrega.

---

## Requisitos

- **Docker Desktop** (Windows / macOS) o Docker Engine + Compose (Linux)
- **~2 GB de disco** para el bundle base, o ~5 GB si incluyen el corpus de mapas
- Nada más. No necesitan Python, ni GDAL, ni QGIS instalados.

---

## Instalación (5 pasos)

### 1. Clonar el repositorio

```bash
git clone <URL-del-repo> CR-BioLM
cd CR-BioLM
```

### 2. Descargar el bundle de datos

Los datos geoespaciales **no están en el repositorio** (son ~115 MB de shapefiles y
rásters, y están en `.gitignore`). Se descargan aparte:

- Enlace: *(pendiente — se los enviamos por el canal acordado)*
- Archivo: `mpcr-data-bundle-v1.tar.gz` (~115 MB)
- Opcional: `mpcr-maps-v1.tar.gz` (~3 GB, mapas pre-renderizados)

Verifiquen la descarga antes de extraer:

```bash
sha256sum mpcr-data-bundle-v1.tar.gz     # comparar con el SHA256 publicado
```

### 3. Extraer en `./data`

```bash
mkdir -p data
tar -xzf mpcr-data-bundle-v1.tar.gz -C data
```

Debe quedar así:

```
data/
├── topography/altitud_cr.tif
├── regiones_botanicas/
├── vectors/
├── Cartografia/
├── gazetteer/entities.csv
├── fichas.sqlite
├── gbif_cache/
├── species_counts.json
├── maps/                    <- solo si extrajeron el bundle de mapas
└── ATTRIBUTION.json         <- fuentes y licencias de cada capa
```

### 4. Levantar el servicio

```bash
docker compose up
```

La primera vez tarda unos minutos construyendo la imagen. Después arranca en segundos.

### 5. Verificar

```bash
curl http://localhost:8080/health
```

Respuesta esperada:

```json
{"status": "ok", "catalog_species": 5791, "maps_available": 5791, "data_version": "v1"}
```

Documentación interactiva: **http://localhost:8080/docs**

Si `maps_available` es 0, no extrajeron el bundle de mapas — la API funciona igual,
pero renderiza bajo demanda y es más lenta.

---

## Uso

Base URL local: `http://localhost:8080`

### Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/health` | Estado del servicio |
| GET | `/v1/vocabulary` | **Empiecen por aquí.** Valores válidos para todos los filtros |
| GET | `/v1/species` | Búsqueda estructurada (filtros combinados con AND) |
| GET | `/v1/species/{nombre}` | Ficha completa de una especie |
| GET | `/v1/species/{nombre}/occurrences` | Ocurrencias GBIF filtradas + estadísticas |
| GET | `/v1/species/{nombre}/map` | Mapa de distribución (URL del PNG) |
| POST | `/v1/parse-distribution` | Parsea texto crudo de distribución a JSON estructurado |

### Ejemplos

```bash
# Vocabulario controlado — háganlo primero
curl http://localhost:8080/v1/vocabulary

# Arbustos endémicos sobre los 2000 m
curl "http://localhost:8080/v1/species?habit=arbusto&endemic=true&elev_lo=2000"

# Ficha completa
curl "http://localhost:8080/v1/species/Peltogyne%20purpurea"

# Mapa
curl "http://localhost:8080/v1/species/Peltogyne%20purpurea/map"

# Ocurrencias GBIF en la vertiente Caribe
curl "http://localhost:8080/v1/species/Peltogyne%20purpurea/occurrences?vertiente=Caribe"
```

**Ojo con los nombres**: van URL-encoded (`%20` por el espacio). El servicio también
acepta guion bajo: `Peltogyne_purpurea`.

### Filtros de `/v1/species`

Todos opcionales, se combinan con AND. Los valores válidos salen de `/v1/vocabulary`.

| Parámetro | Tipo | Ejemplo |
|---|---|---|
| `habit` | texto | `árbol`, `arbusto`, `epifita`, `hierba` |
| `elev_lo` / `elev_hi` | entero (m) | `0`, `2000` |
| `vertiente` | texto | `Caribe`, `Pacífico` |
| `region` | texto | `Cordillera de Talamanca` |
| `forest_type` | texto | `muy húmedo`, `pluvial`, `nuboso`, `páramo` |
| `family` | texto | `Lauraceae`, `Orchidaceae` |
| `flowering_month` | 1–12 | `3` |
| `endemic` | booleano | `true` |
| `limit` / `offset` | entero | paginación |

---

## El formato de respuesta: lean esto

**Toda** respuesta viene envuelta en un sobre de procedencia. No es decoración — es
el mecanismo que hace verificable cada dato, y su interfaz debería mostrarlo.

```json
{
  "value": { "...": "el dato en sí" },
  "source": "MPCR",
  "citation": "Manual de Plantas de Costa Rica, Tomo IV, p. 320",
  "confidence": "exact",
  "caveat": ""
}
```

| Campo | Valores | Qué significa |
|---|---|---|
| `source` | `MPCR`, `GBIF`, `DEM`, `SINAC`, o combinados con `+` | De dónde viene el dato |
| `citation` | texto | Cita bibliográfica. **Muéstrenla en la UI** |
| `confidence` | `exact` | Viene textual del Manual, es confiable |
| | `estimated` | Derivado por reglas o heurísticas (parser, GBIF, DEM) |
| | `insufficient` | No hay datos suficientes. `value` puede venir `null` |
| `caveat` | texto | **Si no está vacío, muéstrenlo al usuario** |

### Reglas de oro

1. **`caveat` no vacío se muestra.** Nunca lo descarten. Avisa cuando el dato es
   sospechoso y el usuario necesita saberlo.
2. **`confidence: "insufficient"` no es un error.** Es una respuesta legítima que
   significa "no sabemos". No lo traten como 500.
3. **`citation` se muestra.** Es el punto entero del proyecto: cada afirmación es
   rastreable al Manual, con tomo y página.

---

## Casos raros que van a encontrar

Diseñen para estos desde el principio, no los descubran en producción:

**Especie con elevación de un solo punto.** Cuando el Manual dice "ca. 2950 m",
`elev_min == elev_max`, y la máscara de elevación del DEM sale vacía. El mapa se
genera sin la capa cian y llega un `caveat` explicándolo.
Ejemplo: *Talamancaster minusculus*.

**Especie sin ocurrencias GBIF.** Muchas especies raras o de descripción reciente no
tienen registros. Devuelve `{"n": 0}` con `confidence: "insufficient"`. El mapa se
genera igual, basado solo en el texto del Manual.

**Menos de 5 puntos GBIF.** El mapa omite la capa de puntos rojos por no ser
estadísticamente útil. `n_pts` viene igual en la respuesta.

**Nombre de especie no encontrado.** 404 con un `caveat` que sugiere revisar la
ortografía o usar `/v1/species`. Puede ser un sinónimo taxonómico — el Manual usa
la nomenclatura vigente a su fecha de publicación.

**Topónimos sin resolver.** El parser es basado en reglas (regex + gazetteer), no ML.
Cuando no logra ubicar un nombre de lugar, lo reporta en `unresolved_tokens` en vez de
inventarlo. Si les aparece seguido para cierta región, avísennos: significa que hay que
crecer el gazetteer.

---

## Instalación sin Docker (alternativa)

Solo si Docker no es opción. Es bastante más frágil, especialmente en Windows.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements_api.txt

export CRBIOLM_DATA_DIR=$(pwd)/data
export MPCR_DATA_DIR=$(pwd)/data
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

Requiere Python 3.12. **No usen `requirements.txt` de la raíz** — está en UTF-16 y trae
todo el stack de desarrollo (jupyter, torch, etc.). Usen `requirements_api.txt`.

---

## Problemas comunes

| Síntoma | Causa | Solución |
|---|---|---|
| `503` al arrancar, `data volume not found` | El bundle no está en `./data` | Revisar el paso 3 y la estructura de carpetas |
| `maps_available: 0` en `/health` | Falta el bundle de mapas | Opcional. Renderiza bajo demanda (más lento) |
| El mapa tarda 10–15 s | Renderizado bajo demanda | Instalar el bundle de mapas |
| `404` en una especie que sí existe | Ortografía o sinónimo taxonómico | Buscar con `/v1/species?family=...` |
| Error de CORS desde el frontend | Origen no permitido | Avisarnos el puerto de su dev server |
| Docker se queda sin memoria | Renderizado bajo demanda con poca RAM | Subir el límite de Docker a 4 GB, o instalar el bundle de mapas |

---

## Reglas de trabajo

- **El contrato está congelado.** Los endpoints y los esquemas de respuesta en
  `docs/openapi.json` no cambian sin avisar. Si necesitan algo que no está, pregunten
  antes de asumir o de parsear alrededor.
- **No dependan de nuestro servidor.** El contenedor local es el modo soportado de
  desarrollo. Si les damos una URL compartida, es para demos e integración, no para
  su ciclo diario de trabajo.
- **Reporten los `caveat` raros.** Si ven uno que no está documentado aquí, es
  información útil para nosotros.

Contacto: *(pendiente — canal acordado)*

---

## Créditos y licencias

Los datos vienen de fuentes con licencias distintas. `data/ATTRIBUTION.json` tiene el
detalle por capa. Resumen:

- **Manual de Plantas de Costa Rica** — Hammel, Grayum, Herrera & Zamora (eds.),
  Missouri Botanical Garden. Texto con derechos reservados; citar tomo y página.
- **Ocurrencias GBIF** — snapshot citable con DOI, ver `ATTRIBUTION.json`.
- **Cartografía base** — Instituto Geográfico Nacional (IGN), Costa Rica.
- **Áreas protegidas** — SINAC.
- **Regiones botánicas** — José Araya, 2026.

Si van a publicar algo (tesis, presentación, sitio web) con estos mapas, hablen con
nosotros antes sobre atribución. Las condiciones de redistribución de las capas base
todavía se están confirmando.
