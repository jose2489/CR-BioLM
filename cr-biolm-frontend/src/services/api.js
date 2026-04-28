import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
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

export const getLLMProfile = (jobId, model) => {
  return API.get(`/llm/${jobId}/${model}`);
};