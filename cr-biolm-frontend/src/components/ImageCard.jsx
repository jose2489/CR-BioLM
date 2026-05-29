const IMAGE_LABELS = {
  mapa_distribucion: "Mapa de Distribución Mesoamericana",
  confusion_matrix:  "Matriz de Confusión",
  shap:              "SHAP — Importancia Global de Variables",
  lime:              "LIME — Explicación Local",
  habitat_manual:    "Mapa de Hábitat Botánico (Manual de Plantas)",
};

const TOOLTIPS = {
  mapa_distribucion: "Distribución geográfica de los registros de presencia en Mesoamérica usados para entrenar el modelo Random Forest.",
  confusion_matrix:  "Muestra cuántas predicciones del modelo fueron correctas (verdaderos positivos/negativos) e incorrectas (falsos positivos/negativos).",
  shap:              "Importancia global de cada variable climática. Las barras más largas indican mayor influencia sobre la predicción.",
  lime:              "Explicación local para un punto de alta idoneidad. Muestra qué variables empujaron la predicción hacia presencia o ausencia.",
  habitat_manual:    "Mapa botánico generado desde el Manual de Plantas de Costa Rica, mostrando la distribución histórica documentada.",
};

const BASE_URL = import.meta.env.VITE_API_BASE_URL;

/**
 * Tarjeta de imagen con etiqueta y tooltip informativo.
 *
 * Props:
 *   imgKey    {string}  - clave de la imagen en results.images
 *   images    {object}  - dict de rutas de imágenes desde el resultado del pipeline
 *   fullWidth {boolean} - si ocupa ancho completo (default: false)
 */
export default function ImageCard({ imgKey, images, fullWidth = false }) {
  const toUrl = (absPath) => {
    if (!absPath) return null;
    const normalized = absPath.replace(/\\/g, "/");
    const after = normalized.split("/outputs/")[1];
    return `${BASE_URL}/static/${after}`;
  };

  const path = images?.[imgKey];
  if (!path) return null;

  return (
    <div style={{ marginBottom: fullWidth ? "2rem" : 0 }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
        <p style={{ color: "#4a6a4a", fontSize: "0.7rem", letterSpacing: "0.1em", textTransform: "uppercase", margin: 0 }}>
          {IMAGE_LABELS[imgKey]}
        </p>
        <div style={{ position: "relative", display: "inline-block" }} className="tooltip-wrapper">
          <span style={{
            display: "inline-flex", alignItems: "center", justifyContent: "center",
            width: 14, height: 14, borderRadius: "50%",
            border: "1px solid #3a5a3a", color: "#4a6a4a",
            fontSize: "0.65rem", cursor: "default", lineHeight: 1, userSelect: "none",
          }}>?</span>
          <div style={{
            display: "none",
            position: "absolute",
            bottom: "120%", left: "50%",
            transform: "translateX(-50%)",
            background: "#1a2f1a",
            border: "1px solid #2a3d2a",
            borderRadius: 4,
            padding: "0.5rem 0.75rem",
            width: 240,
            fontSize: "0.75rem",
            color: "#8aaa8a",
            lineHeight: 1.6,
            zIndex: 100,
            pointerEvents: "none",
          }} className="tooltip-box">
            {TOOLTIPS[imgKey]}
          </div>
        </div>
      </div>
      <img
        src={toUrl(path)}
        alt={imgKey}
        style={{ width: "100%", borderRadius: 6, border: "1px solid #2a3d2a" }}
        onError={(e) => (e.target.style.display = "none")}
      />
    </div>
  );
}
