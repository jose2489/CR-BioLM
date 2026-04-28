import { useState } from "react";
import { runPipeline, getStatus } from "../services/api";

export function useJob() {
  const [status, setStatus] = useState(null);
  const [log, setLog] = useState("");

  const startJob = async (species, question) => {
    const res = await runPipeline({ species, question });

    const interval = setInterval(async () => {
      const data = await getStatus(res.data.job_id);

      setStatus(data.data.status);
      setLog(data.data.log || "");

      // ✅ DETENER cuando termina
      if (data.data.status === "done" || data.data.status === "error") {
        clearInterval(interval);
      }
    }, 3000); // 🔽 bajar frecuencia (3s)
  };

  return { startJob, status, log };
}