import { useState, useEffect } from "react";
import { getLLMProfile, exportZip } from "../../services/api";
import ImageCard from "../../components/ImageCard";

const RESULT_TABS = [
  { id: "visualizations", label: "Visualizaciones" },
  { id: "llm",            label: "Análisis LLM"   },
];

/**
 * Tab de resultados: visualizaciones, análisis LLM y exportación.
 *
 * Props:
 *   results {object} - resultados del pipeline (null si no hay)
 *   jobId   {string} - ID del job activo
 */
export default function ResultadosTab({ results, jobId }) {
  const [activeTab, setActiveTab] = useState("visualizations");
  const [llmText,   setLlmText]   = useState(null);
  const [loadingLLM, setLoadingLLM] = useState(false);

  useEffect(() => {
    if (!results?.success || !jobId) return;
    const fetchLLM = async () => {
      setLoadingLLM(true);
      try {
        const res = await getLLMProfile(jobId);
        setLlmText(res.data.content);
      } catch {
        setLlmText("[Error al cargar el perfil]");
      }
      setLoadingLLM(false);
    };
    fetchLLM();
  }, [results, jobId]);

  if (!results) {
    return (
      <div style={{ textAlign: "center", padding: "6rem 0", color: "#3a5a3a" }}>
        <p style={{ fontSize: "1.1rem", marginBottom: "0.5rem" }}>Sin resultados todavía</p>
        <p style={{ fontSize: "0.85rem" }}>Ejecute un análisis desde la pestaña Consulta</p>
      </div>
    );
  }

  if (!results.success) return null;

  return (
    <div>
      {/* Encabezado de especie y botón exportar */}
      <div style={{
        marginBottom: "2.5rem", borderBottom: "1px solid #2a3d2a",
        paddingBottom: "1.5rem", display: "flex",
        alignItems: "flex-end", justifyContent: "space-between",
      }}>
        <div>
          <p style={{ color: "#4a6a4a", fontSize: "0.75rem", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "0.4rem" }}>
            Especie analizada
          </p>
          <h2 style={{ color: "#e8f5e8", fontSize: "2rem", fontWeight: 300, fontStyle: "italic", margin: 0 }}>
            {results.species}
          </h2>
          <p style={{ color: "#3a5a3a", fontSize: "0.75rem", marginTop: "0.4rem", fontFamily: "monospace" }}>
            {results.output_dir.split(/[\\/]/).pop()}
          </p>
        </div>
        <button
          onClick={() => exportZip(jobId)}
          style={{
            background: "#141f14", border: "1px solid #2a3d2a",
            color: "#6a8a6a", padding: "0.5rem 1.1rem",
            borderRadius: 4, fontSize: "0.8rem",
            letterSpacing: "0.05em", cursor: "pointer",
            fontFamily: "inherit", transition: "all 0.15s",
          }}
        >
          Exportar ZIP
        </button>
      </div>

      {/* Tabs internas */}
      <div style={{ display: "flex", gap: 0, marginBottom: "2rem", borderBottom: "1px solid #2a3d2a" }}>
        {RESULT_TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: "0.65rem 1.5rem", border: "none",
              borderBottom: activeTab === tab.id ? "2px solid #6aaa6a" : "2px solid transparent",
              background: "transparent",
              color: activeTab === tab.id ? "#a8d5a8" : "#4a6a4a",
              fontSize: "0.85rem", letterSpacing: "0.05em",
              cursor: "pointer", transition: "all 0.15s",
              fontFamily: "inherit", marginBottom: -1,
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Visualizaciones */}
      {activeTab === "visualizations" && (
        <div>
          <ImageCard imgKey="mapa_distribucion" images={results.images} fullWidth />
          <ImageCard imgKey="habitat_manual"    images={results.images} fullWidth />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
            <ImageCard imgKey="confusion_matrix" images={results.images} />
            <ImageCard imgKey="shap"             images={results.images} />
            <ImageCard imgKey="lime"             images={results.images} />
          </div>
        </div>
      )}

      {/* Análisis LLM */}
      {activeTab === "llm" && (
        <div>
          {results.llm_model && (
            <div style={{
              display: "inline-flex", alignItems: "center", gap: "0.5rem",
              marginBottom: "1.5rem", background: "#141f14",
              border: "1px solid #2a3d2a", borderRadius: 4,
              padding: "0.4rem 0.9rem",
            }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#6aaa6a" }} />
              <span style={{ color: "#6a8a6a", fontSize: "0.75rem", fontFamily: "monospace" }}>
                {results.llm_model}
              </span>
            </div>
          )}

          {loadingLLM ? (
            <div style={{ color: "#4a6a4a", fontSize: "0.85rem", padding: "3rem 0" }}>
              Cargando perfil LLM...
            </div>
          ) : (
            <div style={{
              background: "#141f14", border: "1px solid #2a3d2a",
              borderRadius: 6, padding: "2rem",
              maxHeight: 600, overflowY: "auto",
            }}>
              <pre style={{
                whiteSpace: "pre-wrap", color: "#f0f0f0",
                fontSize: "0.85rem", lineHeight: 1.8,
                fontFamily: "Georgia, serif", margin: 0,
              }}>
                {llmText || "No disponible"}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
