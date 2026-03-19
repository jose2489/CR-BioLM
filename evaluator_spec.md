# Spec: Evaluador Automático de Perfiles Ecológicos (LLM-as-a-Judge)

## 1. Objetivo
Crear un script en Python (`evaluator.py`) que actúe como un juez ciego. Leerá los perfiles ecológicos generados en la carpeta `outputs/`, los evaluará usando `llama-3.3-70b-versatile` (vía Groq), y generará un archivo `resultados_anova.csv`.

## 2. Entradas (Inputs)
- Directorio de búsqueda: `outputs/`
- El script buscará recursivamente todos los archivos de texto que comiencen con `llm_profile_`.
- De cada archivo, extraerá:
  - Nombre del Modelo (Primera línea del txt).
  - Especie evaluada (Segunda línea del txt).
  - Texto completo del perfil.

## 3. Rúbrica de Evaluación (Prompt del Juez)
Por cada perfil, el script enviará este prompt estricto a Groq:

"Eres un ecólogo experto evaluando respuestas de modelos de IA para una tesis de maestría.
Lee este perfil ecológico:
[TEXTO DEL PERFIL]

Evalúa los siguientes 3 criterios asignando un puntaje entero del 1 al 5 (donde 1 es deficiente y 5 es excelente):
1. Precision_Biologica: ¿La explicación del nicho ecológico tiene sentido fisiológico y no contradice las leyes de la biología tropical?
2. Coherencia_Espacial: ¿La distribución descrita se alinea lógicamente con la geografía y zonas de vida mencionadas?
3. Causalidad_Matematica: ¿El texto basa sus conclusiones en datos o métricas específicas (ej. variables climáticas, impacto negativo/positivo) en lugar de hacer suposiciones geográficas genéricas?

Devuelve ÚNICAMENTE un objeto JSON válido con este formato exacto:
{"Precision_Biologica": INT, "Coherencia_Espacial": INT, "Causalidad_Matematica": INT}"

## 4. Salidas (Outputs)
- Una tabla CSV llamada `resultados_anova.csv` guardada en la raíz del proyecto.
- Columnas esperadas: `[Especie, Modelo, Precision_Biologica, Coherencia_Espacial, Causalidad_Matematica, Promedio_Total]`
- Consola: Mensajes de estado indicando qué archivo se está evaluando.

## 5. Reglas Técnicas
- Usar la librería `openai` para conectarse a Groq (reutilizando `GROQ_API_KEY` de `config.py`).
- Implementar `response_format={"type": "json_object"}` en la llamada a la API para forzar la salida JSON.
- Manejar errores con `try/except` por si Groq rechaza el formato o el archivo no se puede leer.