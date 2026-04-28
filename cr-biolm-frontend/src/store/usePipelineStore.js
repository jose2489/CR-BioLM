import { create } from "zustand";

export const usePipelineStore = create((set) => ({
  jobId: null,
  logs: [],
  results: null,

  setJobId: (id) => set({ jobId: id }),
  addLog: (log) =>
    set((state) => ({ logs: [...state.logs, log] })),
  setResults: (results) => set({ results }),
}));