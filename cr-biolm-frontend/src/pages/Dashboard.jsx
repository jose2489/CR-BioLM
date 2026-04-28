import { useState, useEffect } from "react";
import InputPanel from "../components/InputPanel";
import StatusConsole from "../components/StatusConsole";
import { getLLMProfile } from "../services/api";

const MODELS = [
  { key: "llama", label: "LLaMA 3.3 70B" },
  { key: "qwen", label: "Qwen 3 32B" },
  { key: "baseline_dual", label: "Baseline Dual" },
];

const IMAGE_LABELS = {
  map: "Mapa de Solapamiento Espacial",
  confusion_matrix: "Matriz de Confusión",
  shap: "SHAP — Importancia Global de Variables",
  lime: "LIME — Explicación Local",
  gradcam: "Grad-CAM — Mapa de Calor CNN",
};

export default function Dashboard({ startJob, status, results, jobId }) {
  const [activeTab, setActiveTab] = useState("visualizations");
  const [activeModel, setActiveModel] = useState("llama");
  const [llmTexts, setLlmTexts] = useState({});
  const [loadingLLM, setLoadingLLM] = useState(false);
  const [mainTab, setMainTab] = useState("consulta"); // consulta | resultados | ayuda

  const toUrl = (absPath) => {
    if (!absPath) return null;
    const normalized = absPath.replace(/\\/g, "/");
    const after = normalized.split("/outputs/")[1];
    return `http://127.0.0.1:8000/static/${after}`;
  };

  useEffect(() => {
    if (!results?.success || !jobId) return;
    setMainTab("resultados");
    const fetchLLM = async () => {
      setLoadingLLM(true);
      const fetched = {};
      for (const m of MODELS) {
        try {
          const res = await getLLMProfile(jobId, m.key);
          fetched[m.key] = res.data.content;
        } catch {
          fetched[m.key] = "[Error al cargar el perfil]";
        }
      }
      setLlmTexts(fetched);
      setLoadingLLM(false);
    };
    fetchLLM();
  }, [results, jobId]);

  const resultTabs = [
    { id: "visualizations", label: "Visualizaciones" },
    { id: "llm", label: "Análisis LLM" },
  ];

  return (
    <div className="min-h-screen" style={{ background: "#0f1a0f", fontFamily: "'Georgia', serif" }}>

      {/* Top Nav */}
      <header style={{
        borderBottom: "1px solid #2a3d2a",
        background: "#0f1a0f",
        position: "sticky",
        top: 0,
        zIndex: 50,
      }}>
        <div style={{ maxWidth: 1100, margin: "0 auto", padding: "0 2rem", display: "flex", alignItems: "center", justifyContent: "space-between", height: 64 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem" }}>
            <span style={{ color: "#e8f5e8", fontSize: "1.25rem", fontWeight: 700, letterSpacing: "0.04em" }}>CR-BioLM</span>
            <span style={{ color: "#4a7c4a", fontSize: "0.75rem", letterSpacing: "0.12em", textTransform: "uppercase" }}>Costa Rica</span>
          </div>
          <nav style={{ display: "flex", gap: "0.25rem" }}>
            {["consulta", "resultados", "ayuda"].map((tab) => (
              <button
                key={tab}
                onClick={() => setMainTab(tab)}
                style={{
                  padding: "0.4rem 1.1rem",
                  borderRadius: 4,
                  border: "none",
                  background: mainTab === tab ? "#1e3a1e" : "transparent",
                  color: mainTab === tab ? "#a8d5a8" : "#5a7a5a",
                  fontSize: "0.8rem",
                  letterSpacing: "0.08em",
                  textTransform: "capitalize",
                  cursor: "pointer",
                  transition: "all 0.15s",
                  fontFamily: "inherit",
                }}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main style={{ maxWidth: 1100, margin: "0 auto", padding: "3rem 2rem" }}>

        {/* ── CONSULTA ── */}
        {mainTab === "consulta" && (
          <div style={{ maxWidth: 680, margin: "0 auto" }}>
            <div style={{ marginBottom: "3rem" }}>
              <h1 style={{ color: "#e8f5e8", fontSize: "2.5rem", fontWeight: 300, lineHeight: 1.2, marginBottom: "0.75rem", letterSpacing: "-0.01em" }}>
                Análisis de Distribución<br />
                <span style={{ color: "#6aaa6a", fontStyle: "italic" }}>de Especies</span>
              </h1>
              <p style={{ color: "#4a6a4a", fontSize: "0.95rem", lineHeight: 1.7 }}>
                Ingrese el nombre científico de una especie costarricense para ejecutar el pipeline completo de modelado de distribución con inteligencia artificial.
              </p>
            </div>

            {/* Input card */}
            <div style={{
              background: "#141f14",
              border: "1px solid #2a3d2a",
              borderRadius: 8,
              padding: "2rem",
              marginBottom: "1.5rem",
            }}>
              <InputPanel startJob={startJob} />
            </div>

            {/* Status */}
            {status === "running" && (
              <div style={{ marginTop: "1.5rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: "50%",
                    background: "#6aaa6a",
                    boxShadow: "0 0 8px #6aaa6a",
                    animation: "pulse 1.5s ease-in-out infinite",
                  }} />
                  <span style={{ color: "#6aaa6a", fontSize: "0.85rem", letterSpacing: "0.05em" }}>
                    Pipeline en ejecución — esto puede tomar varios minutos
                  </span>
                </div>
                <StatusConsole />
              </div>
            )}

            {status === "done" && !results?.success && (
              <div style={{ marginTop: "1rem" }}>
                <StatusConsole />
              </div>
            )}

            {results && !results.success && (
              <div style={{
                marginTop: "1rem",
                background: "#1f0f0f",
                border: "1px solid #5a2020",
                borderRadius: 6,
                padding: "1rem 1.25rem",
                color: "#d08080",
                fontSize: "0.875rem",
              }}>
                {results.error}
              </div>
            )}

            {results?.success && (
              <div style={{
                marginTop: "1rem",
                background: "#0f1f0f",
                border: "1px solid #2a5a2a",
                borderRadius: 6,
                padding: "1rem 1.25rem",
                color: "#6aaa6a",
                fontSize: "0.875rem",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}>
                <span>Análisis completado para <em>{results.species}</em></span>
                <button
                  onClick={() => setMainTab("resultados")}
                  style={{
                    background: "#1e3a1e",
                    border: "1px solid #3a6a3a",
                    color: "#a8d5a8",
                    padding: "0.35rem 0.9rem",
                    borderRadius: 4,
                    fontSize: "0.8rem",
                    cursor: "pointer",
                    fontFamily: "inherit",
                  }}
                >
                  Ver resultados
                </button>
              </div>
            )}
          </div>
        )}

        {/* ── RESULTADOS ── */}
        {mainTab === "resultados" && (
          <div>
            {!results && (
              <div style={{ textAlign: "center", padding: "6rem 0", color: "#3a5a3a" }}>
                <p style={{ fontSize: "1.1rem", marginBottom: "0.5rem" }}>Sin resultados todavía</p>
                <p style={{ fontSize: "0.85rem" }}>Ejecute un análisis desde la pestaña Consulta</p>
              </div>
            )}

            {results?.success && (
              <div>
                {/* Species header */}
                <div style={{ marginBottom: "2.5rem", borderBottom: "1px solid #2a3d2a", paddingBottom: "1.5rem" }}>
                  <p style={{ color: "#4a6a4a", fontSize: "0.75rem", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "0.4rem" }}>
                    Especie analizada
                  </p>
                  <h2 style={{ color: "#e8f5e8", fontSize: "2rem", fontWeight: 300, fontStyle: "italic", margin: 0 }}>
                    {results.species}
                  </h2>
                  <p style={{ color: "#3a5a3a", fontSize: "0.75rem", marginTop: "0.4rem", fontFamily: "monospace" }}>
                    {results.output_dir.split("\\").pop()}
                  </p>
                </div>

                {/* Result tabs */}
                <div style={{ display: "flex", gap: "0", marginBottom: "2rem", borderBottom: "1px solid #2a3d2a" }}>
                  {resultTabs.map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      style={{
                        padding: "0.65rem 1.5rem",
                        border: "none",
                        borderBottom: activeTab === tab.id ? "2px solid #6aaa6a" : "2px solid transparent",
                        background: "transparent",
                        color: activeTab === tab.id ? "#a8d5a8" : "#4a6a4a",
                        fontSize: "0.85rem",
                        letterSpacing: "0.05em",
                        cursor: "pointer",
                        transition: "all 0.15s",
                        fontFamily: "inherit",
                        marginBottom: -1,
                      }}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                {/* Visualizations */}
                {activeTab === "visualizations" && (
                  <div>
                    <div style={{ marginBottom: "2rem" }}>
                      <p style={{ color: "#4a6a4a", fontSize: "0.7rem", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "0.75rem" }}>
                        {IMAGE_LABELS["map"]}
                      </p>
                      <img
                        src={toUrl(results.images.map)}
                        alt="map"
                        style={{ width: "100%", borderRadius: 6, border: "1px solid #2a3d2a" }}
                        onError={(e) => (e.target.style.display = "none")}
                      />
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
                      {Object.entries(results.images)
                        .filter(([key]) => key !== "map")
                        .map(([key, path]) => (
                          <div key={key}>
                            <p style={{ color: "#4a6a4a", fontSize: "0.7rem", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "0.75rem" }}>
                              {IMAGE_LABELS[key] || key}
                            </p>
                            <img
                              src={toUrl(path)}
                              alt={key}
                              style={{ width: "100%", borderRadius: 6, border: "1px solid #2a3d2a" }}
                              onError={(e) => (e.target.style.display = "none")}
                            />
                          </div>
                        ))}
                    </div>
                  </div>
                )}

                {/* LLM */}
                {activeTab === "llm" && (
                  <div>
                    <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1.5rem" }}>
                      {MODELS.map((m) => (
                        <button
                          key={m.key}
                          onClick={() => setActiveModel(m.key)}
                          style={{
                            padding: "0.4rem 1rem",
                            borderRadius: 4,
                            border: "1px solid",
                            borderColor: activeModel === m.key ? "#3a6a3a" : "#2a3d2a",
                            background: activeModel === m.key ? "#1e3a1e" : "transparent",
                            color: activeModel === m.key ? "#a8d5a8" : "#4a6a4a",
                            fontSize: "0.8rem",
                            cursor: "pointer",
                            transition: "all 0.15s",
                            fontFamily: "inherit",
                          }}
                        >
                          {m.label}
                        </button>
                      ))}
                    </div>

                    {loadingLLM ? (
                      <div style={{ color: "#4a6a4a", fontSize: "0.85rem", padding: "3rem 0" }}>
                        Cargando perfiles LLM...
                      </div>
                    ) : (
                      <div style={{
                        background: "#141f14",
                        border: "1px solid #2a3d2a",
                        borderRadius: 6,
                        padding: "2rem",
                        maxHeight: 600,
                        overflowY: "auto",
                      }}>
                        <pre style={{
                          whiteSpace: "pre-wrap",
                          color: "#c8e8c8",
                          fontSize: "0.85rem",
                          lineHeight: 1.8,
                          fontFamily: "Georgia, serif",
                          margin: 0,
                        }}>
                          {llmTexts[activeModel] || "No disponible"}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── AYUDA ── */}
        {mainTab === "ayuda" && (
          <div style={{ maxWidth: 760, margin: "0 auto" }}>
            <div style={{ marginBottom: "3rem" }}>
              <h1 style={{ color: "#e8f5e8", fontSize: "2rem", fontWeight: 300, marginBottom: "0.5rem" }}>
                Guía de uso
              </h1>
              <p style={{ color: "#4a6a4a", fontSize: "0.9rem" }}>
                Todo lo que necesita saber para utilizar CR-BioLM correctamente.
              </p>
            </div>

            {[
              {
                title: "¿Qué es CR-BioLM?",
                body: "CR-BioLM es una plataforma de modelado de distribución de especies para Costa Rica. Combina datos de presencia de GBIF, variables climáticas de WorldClim, modelos de Machine Learning (Random Forest) y Deep Learning (CNN Multimodal), técnicas de explicabilidad (SHAP, LIME, Grad-CAM) y análisis generativo mediante LLMs (LLaMA, Qwen)."
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
                body: "Mapa de Solapamiento Espacial: distribución geográfica de la especie en Costa Rica. Matriz de Confusión: rendimiento de clasificación del modelo. SHAP: importancia global de las variables climáticas. LIME: explicación local de una predicción individual. Grad-CAM: mapa de calor espacial generado por la CNN."
              },
              {
                title: "¿Qué es el Análisis LLM?",
                body: "Tras el modelado, tres modelos de lenguaje (LLaMA 3.3 70B, Qwen 3 32B y Baseline Dual) generan un perfil ecológico de la especie interpretando las métricas del modelo, los valores SHAP y el contexto de conservación. Puede comparar las tres respuestas usando los botones de selección."
              },
              {
                title: "Nombres de especies válidos",
                body: "Utilice el nombre científico binomial completo en latín (género + epíteto específico). Ejemplos válidos: Quercus costaricensis, Tapirus bairdii, Morpho peleides. El sistema consulta directamente la base de datos GBIF, por lo que especies con pocos registros o nombres incorrectos pueden no arrojar resultados."
              },
            ].map((item, i) => (
              <div key={i} style={{
                borderTop: "1px solid #2a3d2a",
                padding: "1.75rem 0",
              }}>
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
                CR-BioLM · Proyecto de investigación · Universidad de Costa Rica<br />
                Datos: GBIF, WorldClim V2.1, SINAC, SNIT
              </p>
            </div>
          </div>
        )}
      </main>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0f1a0f; }
        ::-webkit-scrollbar-thumb { background: #2a3d2a; border-radius: 3px; }
      `}</style>
    </div>
  );
}
