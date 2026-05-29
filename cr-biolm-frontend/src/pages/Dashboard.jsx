import { useState, useEffect } from "react";
import { usePipelineStore } from "../store/usePipelineStore";
import { useJob } from "../hooks/useJob";
import ConsultaTab   from "../components/tabs/ConsultaTab";
import ResultadosTab from "../components/tabs/ResultadosTab";
import AyudaTab      from "../components/tabs/AyudaTab";

const MAIN_TABS = ["consulta", "resultados", "ayuda"];

/**
 * Layout principal de la aplicación.
 * Orquesta navegación entre tabs y provee startJob a ConsultaTab.
 * Todo el estado del pipeline se lee desde usePipelineStore.
 */
export default function Dashboard() {
  const [mainTab, setMainTab] = useState("consulta");
  const { startJob } = useJob();
  const { status, results, jobId } = usePipelineStore();

  // Navegar automáticamente a resultados al completar
  useEffect(() => {
    if (results?.success) setMainTab("resultados");
  }, [results]);

  return (
    <div className="min-h-screen" style={{ background: "#0f1a0f", fontFamily: "'Georgia', serif" }}>

      {/* Top Nav */}
      <header style={{
        borderBottom: "1px solid #2a3d2a", background: "#0f1a0f",
        position: "sticky", top: 0, zIndex: 50,
      }}>
        <div style={{ maxWidth: 1100, margin: "0 auto", padding: "0 2rem", display: "flex", alignItems: "center", justifyContent: "space-between", height: 64 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem" }}>
            <span style={{ color: "#e8f5e8", fontSize: "1.25rem", fontWeight: 700, letterSpacing: "0.04em" }}>CR-BioLM</span>
            <span style={{ color: "#4a7c4a", fontSize: "0.75rem", letterSpacing: "0.12em", textTransform: "uppercase" }}>Costa Rica</span>
          </div>
          <nav style={{ display: "flex", gap: "0.25rem" }}>
            {MAIN_TABS.map((tab) => (
              <button
                key={tab}
                onClick={() => setMainTab(tab)}
                style={{
                  padding: "0.4rem 1.1rem", borderRadius: 4, border: "none",
                  background: mainTab === tab ? "#1e3a1e" : "transparent",
                  color: mainTab === tab ? "#a8d5a8" : "#5a7a5a",
                  fontSize: "0.8rem", letterSpacing: "0.08em",
                  textTransform: "capitalize", cursor: "pointer",
                  transition: "all 0.15s", fontFamily: "inherit",
                }}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main style={{ maxWidth: 1100, margin: "0 auto", padding: "3rem 2rem" }}>
        {mainTab === "consulta"   && (
          <ConsultaTab
            startJob={startJob}
            status={status}
            results={results}
            onViewResults={() => setMainTab("resultados")}
          />
        )}
        {mainTab === "resultados" && <ResultadosTab results={results} jobId={jobId} />}
        {mainTab === "ayuda"      && <AyudaTab />}
      </main>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0f1a0f; }
        ::-webkit-scrollbar-thumb { background: #2a3d2a; border-radius: 3px; }
        .tooltip-wrapper:hover .tooltip-box { display: block !important; }
      `}</style>
    </div>
  );
}
