import { usePipelineStore } from "../store/usePipelineStore";

/**
 * Muestra los logs del pipeline en tiempo real.
 * Solo lee del store; el polling lo maneja useJob.
 */
export default function StatusConsole() {
  const logs = usePipelineStore((state) => state.logs);

  return (
    <div className="bg-black text-green-400 p-4 h-60 overflow-auto rounded">
      {logs.map((log, i) => (
        <div key={i}>{log}</div>
      ))}
    </div>
  );
}
