import { create } from "zustand";

/**
 * Fuente de verdad global del pipeline.
 * Todos los componentes leen desde aquí; useJob escribe aquí.
 */
export const usePipelineStore = create((set) => ({
  jobId:   null,
  status:  null,
  logs:    [],
  results: null,

  setJobId:   (jobId)   => set({ jobId }),
  setStatus:  (status)  => set({ status }),
  setResults: (results) => set({ results }),
  addLog:     (log)     => set((state) => ({ logs: [...state.logs, log] })),
  reset:      ()        => set({ jobId: null, status: null, logs: [], results: null }),
}));