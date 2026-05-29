import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL;

const API = axios.create({
  baseURL: BASE_URL,
});

// ===== ENDPOINTS =====
export const runPipeline = (data) => {
  return API.post("/run", data);
};

export const getStatus = (jobId) => {
  return API.get(`/status/${jobId}`);
};

export const getResults = (jobId) => {
  return API.get(`/results/${jobId}`);
};

export const getLLMProfile = (jobId) => {
  return API.get(`/llm/${jobId}`);
};

export const exportZip = (jobId) => {
  window.open(`${BASE_URL}/export/${jobId}`, "_blank");
};