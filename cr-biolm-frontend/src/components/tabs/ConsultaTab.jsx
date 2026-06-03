import InputPanel from "../../components/InputPanel";
import StatusConsole from "../../components/StatusConsole";

/**
 * Tab de consulta: formulario de entrada y estado del pipeline en ejecución.
 *
 * Props:
 *   startJob      {function} - inicia un job con (species, question)
 *   status        {string}   - estado actual del job ('running', 'done', 'error', null)
 *   results       {object}   - resultados del pipeline (null si no hay)
 *   onViewResults {function} - navega a la tab de resultados
 */
export default function ConsultaTab({ startJob, status, results, onViewResults }) {
  return (
    <div style={{ maxWidth: 680, margin: "0 auto" }}>
      <div style={{ marginBottom: "3rem" }}>
        <h1 style={{ color: "#e8f5e8", fontSize: "2.5rem", fontWeight: 300, lineHeight: 1.2, marginBottom: "0.75rem", letterSpacing: "-0.01em" }}>
          Análisis de Distribución<br />
          <span style={{ color: "#6aaa6a", fontStyle: "italic" }}>de Especies</span>
        </h1>
        <p style={{ color: "#4a6a4a", fontSize: "0.95rem", lineHeight: 1.7 }}>
          Ingrese el nombre científico de una especie costarricense para ejecutar
          el pipeline completo de modelado de distribución con inteligencia artificial.
        </p>
      </div>

      <div style={{
        background: "#141f14", border: "1px solid #2a3d2a",
        borderRadius: 8, padding: "2rem", marginBottom: "1.5rem",
      }}>
        <InputPanel startJob={startJob} />
      </div>

      {/* Error — siempre arriba de la consola */}
      {results && !results.success && (
        <div style={{
          marginTop: "1rem", background: "#1f0f0f",
          border: "1px solid #5a2020", borderRadius: 6,
          padding: "1rem 1.25rem", color: "#d08080", fontSize: "0.875rem",
        }}>
          {results.error}
        </div>
      )}

      {/* Consola — pipeline corriendo */}
      {status === "running" && (
        <div style={{ marginTop: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: "#6aaa6a", boxShadow: "0 0 8px #6aaa6a",
              animation: "pulse 1.5s ease-in-out infinite",
            }} />
            <span style={{ color: "#6aaa6a", fontSize: "0.85rem", letterSpacing: "0.05em" }}>
              Pipeline en ejecución — esto puede tomar varios minutos
            </span>
          </div>
          <StatusConsole />
        </div>
      )}

      {/* Consola — pipeline terminado con error */}
      {status === "done" && !results?.success && (
        <div style={{ marginTop: "1rem" }}>
          <StatusConsole />
        </div>
      )}

      {/* Éxito */}
      {results?.success && (
        <div style={{
          marginTop: "1rem", background: "#0f1f0f",
          border: "1px solid #2a5a2a", borderRadius: 6,
          padding: "1rem 1.25rem", color: "#6aaa6a", fontSize: "0.875rem",
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <span>Análisis completado para <em>{results.species}</em></span>
          <button
            onClick={onViewResults}
            style={{
              background: "#1e3a1e", border: "1px solid #3a6a3a",
              color: "#a8d5a8", padding: "0.35rem 0.9rem",
              borderRadius: 4, fontSize: "0.8rem",
              cursor: "pointer", fontFamily: "inherit",
            }}
          >
            Ver resultados
          </button>
        </div>
      )}
    </div>
  );
}
