import { useRef, useEffect } from "react";
import { runPipeline, getStatus, getResults } from "../services/api";
import { usePipelineStore } from "../store/usePipelineStore";

const POLL_INTERVAL_MS = 3000;

/**
 * Orquesta el ciclo de vida de un job del pipeline.
 * Escribe todo el estado en usePipelineStore; no mantiene estado propio.
 */
export function useJob() {
  const { setJobId, setStatus, setResults, addLog, reset } = usePipelineStore();
  const intervalRef = useRef(null);

  const startJob = async (species, question) => {
    // Limpiar job anterior
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    reset();

    const res = await runPipeline({ species, question });
    const id = res.data.job_id;
    setJobId(id);
    setStatus("running");

    intervalRef.current = setInterval(async () => {
      const resStatus = await getStatus(id);
      const s = resStatus.data.status;

      setStatus(s);
      if (resStatus.data.log) addLog(resStatus.data.log);

      if (s === "done" || s === "error" || s === "not_found") {
        clearInterval(intervalRef.current);
        intervalRef.current = null;

        if (s === "done") {
          const resResults = await getResults(id);
          setResults(resResults.data);
        }
      }
    }, POLL_INTERVAL_MS);
  };

  // Limpieza al desmontar
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, []);

  return { startJob };
}