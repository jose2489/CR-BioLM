import Dashboard from "./pages/Dashboard";
import { useJob } from "./hooks/useJob";

export default function App() {
  const { startJob, status, log, results, jobId } = useJob();
  return (
    <Dashboard
      startJob={startJob}
      status={status}
      log={log}
      results={results}
      jobId={jobId}
    />
  );
}