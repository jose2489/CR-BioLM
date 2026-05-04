# Plan: Regiones Botánicas → Hábitat por Especie

Fuente del mapa: `C:\Users\Jose\Documents\Tesis\raw_data\Manual de Especies\Mapa_Manuales.png`
Imagen: 1401 × 1026 px, RGBA

---

## Lógica de las tres capas

```
Texto del Manual: "vert. Carib. Cords. Central y Talamanca, 400–1400 m"
                         │                   │                    │
              Máscara de vertiente    Polígonos de región    Rango de elevación
              (Caribe vs Pacífico)     (regiones 7 y 15)    (altitud_cr.tif)
                         └──────────── INTERSECCIÓN ─────────────┘
                                              │
                              Zona de hábitat precisa
```

Las cordilleras (regiones 3, 5, 7, 15) atraviesan AMBAS vertientes.
"vert. Carib. Cord. Central" = ladera caribeña de la Cordillera Central solamente.
Las tres capas juntas dan la zona exacta.

---

## Checklist de ejecución

### Instalación de dependencias
```bash
pip install pysheds scikit-learn scikit-image
```

---

### PASO 1 — Recortar el inset del mapa de regiones
```bash
python utils/map_gen/paso1_recortar_inset.py
```
- [x] Ejecutado (2026-04-09)
- [x] Validar: `data_raw/regiones_botanicas/inset_regiones.png` ✓ — muestra solo el mapa de regiones

---

### PASO 2 — Segmentar los 25 colores → máscaras
```bash
python utils/map_gen/paso2_segmentar_regiones.py
```
- [x] Ejecutado (2026-04-09)
- [x] Validar: `data_raw/regiones_botanicas/validacion_segmentos.png` ✓ — 25 regiones bien separadas

---

### PASO 3 — Georreferenciar + vectorizar → shapefile
```bash
python utils/map_gen/paso3_vectorizar_regiones.py
```
- [x] Ejecutado (2026-04-09)
- [x] Validar: `data_raw/regiones_botanicas/validacion_shapefile.png` ✓ — polígonos ubicados sobre CR
- [!] **Calibración pendiente**: los IDs 1-25 se asignan por centroide más cercano a coordenadas
      conocidas de cada región. Verificar en el mapa de validación si Nicoya (4) está al NO,
      Talamanca (15) en el SE-centro, Guanacaste (1) al NO.
      Ajustar `CENTROIDES_CONOCIDOS` en `paso3_vectorizar_regiones.py` si es necesario.

---

### PASO 4 — División continental → máscaras de vertiente
```bash
python utils/map_gen/paso4_division_continental.py
```
- [x] Ejecutado (2026-04-09)
- [x] Validar: `data_raw/regiones_botanicas/validacion_vertientes.png` ✓ — línea roja separa Caribe (E/azul) de Pacífico (O-SO/verde), cordilleras visibles
- [!] Nota: se usó watershed inverso con scipy/skimage (pysheds no instalado). Para mayor
      precisión hidrológica instalar pysheds y re-ejecutar.

---

### PASO 5 — Mapas de prueba para 5 especies
```bash
python utils/map_gen/paso5_test_mapas.py
```
- [x] Ejecutado (2026-04-09)
- [x] Resultado: `outputs/test_manual_maps/test_5_especies.png` generado con 5 especies
- [x] El script usa fallback progresivo: región+vertiente+elev → región+elev → solo elev
- [!] Validar: ¿zonas azules con sentido botánico? ¿puntos GBIF (si existen) cerca de zonas azules?
      Especies sin GBIF en carpeta de salida no muestran puntos rojos.

---

### PASO 6 — Integrar al pipeline principal
- [ ] Editar `main.py` (instrucciones al final de este documento)
- [ ] Editar `llm/openrouter_client.py` (soporte para 2 imágenes)
- [ ] Editar `llm/prompt_templates.py` (añadir FUENTE 3)
- [ ] Ejecutar pipeline completo con especie de prueba
- [ ] Validar calidad de respuesta del LLM

---

## Datos de salida por paso

| Paso | Archivos generados |
|------|--------------------|
| 1 | `data_raw/regiones_botanicas/inset_regiones.png` |
| 2 | `data_raw/regiones_botanicas/segmentos.npz`, `validacion_segmentos.png` |
| 3 | `regiones_botanicas_cr.shp` (+.dbf .shx .prj .cpg), `validacion_shapefile.png` |
| 4 | `vertiente_caribe.tif`, `vertiente_pacifico.tif`, `division_continental.shp`, `validacion_vertientes.png` |
| 5 | `outputs/test_manual_maps/*.png` (una imagen por especie) |

---

## Tabla de regiones (para referencia)

| ID | Nombre | Vertiente principal |
|----|--------|-------------------|
| 1  | Llanuras de Guanacaste | Pacífico |
| 2  | Llanura de Los Guatusos | Caribe |
| 3  | Cordillera de Guanacaste | Ambas |
| 4  | Península de Nicoya | Pacífico |
| 5  | Cordillera de Tilarán | Ambas |
| 6  | Llanuras de San Carlos | Caribe |
| 7  | Cordillera Central | Ambas |
| 8  | Llanuras de Tortuguero/Santa Clara | Caribe |
| 9  | Valle Central Occidental | Pacífico |
| 10 | Valle Central Oriental | Caribe |
| 11 | Puriscal-Los Santos | Pacífico |
| 12 | Turrubares | Pacífico |
| 13 | Tárcoles-Térraba | Pacífico |
| 14 | Baja Talamanca | Caribe |
| 15 | Cordillera de Talamanca | Ambas |
| 16 | Filas Chonta y Nara | Caribe |
| 17 | Valle del General | Pacífico |
| 18 | Valle de Coto Brus | Pacífico |
| 19 | Fila Costeña Norte | Pacífico |
| 20 | Fila Costeña Sur | Pacífico |
| 21 | Valle del Diquís | Pacífico |
| 22 | Península de Osa-Golfito | Pacífico |
| 23 | Valle de Coto Colorado | Pacífico |
| 24 | Punta Burica | Pacífico |
| 25 | Isla del Coco | Pacífico |

---

## Instrucciones de integración (Paso 6)

### main.py — añadir después del mapa CR

```python
from data.manual_range_generator import ManualRangeGenerator
import pandas as pd

_manual_catalog = pd.read_csv("outputs/picked_species_enhanced.csv")
_manual_gen = ManualRangeGenerator()

_row = _manual_catalog[_manual_catalog['species'] == especie_nombre]
if not _row.empty:
    r = _row.iloc[0]
    ruta_mapa_manual = _manual_gen.generate(
        species_name     = especie_nombre,
        geographic_notes = r['geographic_notes'],
        elevation_min    = r['elevation_min_m'],
        elevation_max    = r['elevation_max_m'],
        habitat_type     = r['habitat_type'],
        presencias_cr    = presencias_cr,
        cr_bounds        = cr_bounds,
        output_dir       = out_dir
    )
    texto_manual = _manual_gen.get_summary_text(r)
else:
    ruta_mapa_manual = None
    texto_manual = ""
```

### openrouter_client.py — pasar imagen adicional

Buscar la llamada donde se construye el mensaje con la imagen y añadir
`ruta_mapa_manual` como segundo elemento de la lista de imágenes.

### prompt_templates.py — añadir FUENTE 3 al BIMODAL_PROMPT

```
FUENTE 3: REFERENCIA BOTÁNICA (Manual de Plantas de Costa Rica)
{texto_manual}
La imagen 2 adjunta muestra el hábitat potencial según el Manual,
cruzado con vertiente y elevación. Úsala para validar o contrastar
los hallazgos matemáticos del modelo predictivo.
```
