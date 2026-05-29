const HELP_ITEMS = [
  {
    title: "¿Qué es CR-BioLM?",
    body: "CR-BioLM es una plataforma de modelado de distribución de especies para Costa Rica. Combina datos de presencia de GBIF, variables climáticas de WorldClim, un modelo Random Forest, técnicas de explicabilidad (SHAP y LIME) y análisis generativo mediante un LLM configurable vía OpenRouter."
  },
  {
    title: "¿Cómo realizar una consulta?",
    body: "En la pestaña Consulta, ingrese el nombre científico completo de la especie (por ejemplo: Quercus costaricensis). Opcionalmente, puede formular una pregunta específica que el modelo LLM responderá en el análisis. Haga clic en Ejecutar para iniciar el pipeline."
  },
  {
    title: "¿Cuánto tarda el análisis?",
    body: "El pipeline completo toma entre 5 y 15 minutos dependiendo de la especie y la disponibilidad de datos en GBIF. Durante la ejecución, la consola de estado muestra el progreso en tiempo real."
  },
  {
    title: "¿Qué muestran las Visualizaciones?",
    body: "Mapa de Distribución Mesoamericana: distribución geográfica de la especie y puntos GBIF. Mapa de Hábitat Botánico: generado desde el Manual de Plantas de Costa Rica (solo si la especie está en el catálogo). Matriz de Confusión: rendimiento del modelo Random Forest. SHAP: importancia global de las variables climáticas. LIME: explicación local de una predicción individual. Pase el cursor sobre el ícono ? junto a cada imagen para ver una descripción."
  },
  {
    title: "¿Qué es el Análisis LLM?",
    body: "Tras el modelado, el modelo de lenguaje activo genera un perfil ecológico de la especie interpretando las métricas del modelo, los valores SHAP, la altitud observada y el contexto de conservación. El modelo activo se configura en nodes/config_llm.py."
  },
  {
    title: "¿Cómo exportar los resultados?",
    body: "En la pestaña Resultados encontrará el botón Exportar ZIP en la esquina superior derecha. Al hacer clic descargará un archivo comprimido con todas las imágenes generadas, el perfil ecológico del LLM y los archivos JSON de trazabilidad de cada nodo del pipeline."
  },
  {
    title: "Nombres de especies válidos",
    body: "Utilice el nombre científico binomial completo en latín (género + epíteto específico). Ejemplos válidos: Quercus costaricensis, Myrsine coriacea, Tapirus bairdii. El sistema consulta directamente la base de datos GBIF, por lo que especies con pocos registros o nombres incorrectos pueden no arrojar resultados."
  },
];

/**
 * Tab de ayuda: guía de uso de la plataforma.
 * Sin props; contenido estático.
 */
export default function AyudaTab() {
  return (
    <div style={{ maxWidth: 760, margin: "0 auto" }}>
      <div style={{ marginBottom: "3rem" }}>
        <h1 style={{ color: "#e8f5e8", fontSize: "2rem", fontWeight: 300, marginBottom: "0.5rem" }}>
          Guía de uso
        </h1>
        <p style={{ color: "#4a6a4a", fontSize: "0.9rem" }}>
          Todo lo que necesita saber para utilizar CR-BioLM correctamente.
        </p>
      </div>

      {HELP_ITEMS.map((item, i) => (
        <div key={i} style={{ borderTop: "1px solid #2a3d2a", padding: "1.75rem 0" }}>
          <h3 style={{ color: "#a8d5a8", fontSize: "1rem", fontWeight: 600, marginBottom: "0.75rem", letterSpacing: "0.01em" }}>
            {item.title}
          </h3>
          <p style={{ color: "#6a8a6a", fontSize: "0.9rem", lineHeight: 1.8, margin: 0 }}>
            {item.body}
          </p>
        </div>
      ))}

      <div style={{ borderTop: "1px solid #2a3d2a", paddingTop: "2rem", marginTop: "1rem" }}>
        <p style={{ color: "#3a5a3a", fontSize: "0.8rem", lineHeight: 1.7 }}>
          CR-BioLM · Práctica Profesional · Instituto Tecnológico de Costa Rica<br />
          Datos: GBIF · WorldClim V2.1 · SINAC · SNIT
        </p>
      </div>
    </div>
  );
}
