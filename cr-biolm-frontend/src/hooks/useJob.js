import { useState, useRef, useEffect } from "react";
import { runPipeline, getStatus, getResults } from "../services/api";
export function useJob() {
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [log, setLog] = useState("");
  const [results, setResults] = useState(null);
  const intervalRef = useRef(null);
  const startJob = async (species, question) => {
    // limpiar interval anterior si existe
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    const res = await runPipeline({ species, question });
    const id = res.data.job_id;
    setJobId(id);
    setStatus("running");
    setResults(null);
    intervalRef.current = setInterval(async () => {
      const resStatus = await getStatus(id);
      const s = resStatus.data.status;
      console.log("Polling status:", s, "| interval ref:", intervalRef.current);
      setStatus(s);
      setLog(resStatus.data.log || "");
      if (s === "done" || s === "error" || s === "not_found") {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
        if (s === "done") {
          const resResults = await getResults(id);
          setResults(resResults.data);
        }
      }
    }, 3000);
  };
  // limpieza al desmontar componente
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, []);
  return { startJob, status, log, results, jobId };
}