from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from job_store import create_job, get_job
from runner import run_pipeline

import os

app = FastAPI()

# Outputs del pipeline
OUTPUT_DIR = os.path.abspath("../outputs")
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")


# ===== MODELO REQUEST =====
class RunRequest(BaseModel):
    species: str
    question: str | None = None


# ===== ENDPOINTS =====

@app.post("/run")
def run(req: RunRequest, background_tasks: BackgroundTasks):
    job_id = create_job()

    background_tasks.add_task(
        run_pipeline,
        job_id,
        req.species,
        req.question
    )

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = get_job(job_id)

    if job is None:
        return {"status": "not_found", "log": None}

    return {
        "status": job["status"],
        "log": job["logs"][-1] if job["logs"] else None
    }


@app.get("/results/{job_id}")
def results(job_id: str):
    job = get_job(job_id)

    if job is None:
        return {"error": "job not found"}

    return job.get("results", {})

@app.get("/llm/{job_id}/{model}")
def llm_profile(job_id: str, model: str):
    job = get_job(job_id)
    if job is None:
        return {"error": "job not found"}
    results = job.get("results", {})
    if not results or not results.get("llm_profiles"):
        return {"error": "no llm profiles found"}
    path = results["llm_profiles"].get(model)
    if not path or not os.path.exists(path):
        return {"error": f"profile not found for model: {model}"}
    with open(path, "r", encoding="utf-8") as f:
        return {"model": model, "content": f.read()}


# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)