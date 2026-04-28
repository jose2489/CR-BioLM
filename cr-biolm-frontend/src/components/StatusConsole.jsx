import { useEffect } from "react";
import { usePipelineStore } from "../store/usePipelineStore";
import { getStatus } from "../services/api";

export default function StatusConsole() {
  const { jobId, logs, addLog } = usePipelineStore();

  useEffect(() => {
    if (!jobId) return;
    const i = setInterval(async () => {
      const res = await getStatus(jobId);
      if (res.data.log) addLog(res.data.log);
      if (
        res.data.status === "done" ||
        res.data.status === "error" ||
        res.data.status === "not_found"
      ) {
        clearInterval(i);
      }
    }, 2000);
    return () => clearInterval(i);
  }, [jobId]);

  return (
    <div className="bg-black text-green-400 p-4 h-60 overflow-auto rounded">
      {logs.map((l, i) => (
        <div key={i}>{l}</div>
      ))}
    </div>
  );
}