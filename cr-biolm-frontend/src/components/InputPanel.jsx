import { useState } from "react";

export default function InputPanel({ startJob }) {
  const [species, setSpecies] = useState("");
  const [question, setQuestion] = useState("");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div>
        <label style={{ color: "#4a6a4a", fontSize: "0.75rem", letterSpacing: "0.1em", textTransform: "uppercase", display: "block", marginBottom: "0.5rem" }}>
          Nombre científico
        </label>
        <input
          value={species}
          onChange={(e) => setSpecies(e.target.value)}
          placeholder="Quercus costaricensis"
          style={{
            width: "100%",
            background: "#1a2f1a",
            border: "1px solid #2a3d2a",
            borderRadius: 4,
            padding: "0.6rem 0.875rem",
            color: "#e8f5e8",
            fontSize: "0.9rem",
            fontFamily: "Georgia, serif",
            outline: "none",
            boxSizing: "border-box",
          }}
        />
      </div>

      <div>
        <label style={{ color: "#4a6a4a", fontSize: "0.75rem", letterSpacing: "0.1em", textTransform: "uppercase", display: "block", marginBottom: "0.5rem" }}>
          Pregunta para el LLM <span style={{ color: "#3a5a3a" }}>(opcional)</span>
        </label>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="¿Cómo le afecta el cambio climático?"
          rows={3}
          style={{
            width: "100%",
            background: "#1a2f1a",
            border: "1px solid #2a3d2a",
            borderRadius: 4,
            padding: "0.6rem 0.875rem",
            color: "#e8f5e8",
            fontSize: "0.9rem",
            fontFamily: "Georgia, serif",
            outline: "none",
            resize: "vertical",
            boxSizing: "border-box",
          }}
        />
      </div>

      <button
        onClick={() => startJob(species, question)}
        disabled={!species.trim()}
        style={{
          background: species.trim() ? "#1e3a1e" : "#141f14",
          border: "1px solid",
          borderColor: species.trim() ? "#3a6a3a" : "#2a3d2a",
          color: species.trim() ? "#a8d5a8" : "#3a5a3a",
          padding: "0.65rem 1.5rem",
          borderRadius: 4,
          fontSize: "0.85rem",
          letterSpacing: "0.05em",
          cursor: species.trim() ? "pointer" : "not-allowed",
          fontFamily: "Georgia, serif",
          transition: "all 0.15s",
          alignSelf: "flex-start",
        }}
      >
        Ejecutar análisis
      </button>
    </div>
  );
}