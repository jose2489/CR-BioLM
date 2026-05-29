import os
import glob
import json
import zipfile
import tempfile

from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from job_store import create_job, get_job
from runner import run_pipeline

app = FastAPI()

# ===== CONFIGURACIÓN POR VARIABLES DE ENTORNO =====
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.abspath("../outputs"))
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173"
).split(",")

app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")


# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    job_results = job.get("results", {})
    if not job_results:
        return job_results

    # Leer modelo activo desde el JSON del nodo N13
    output_dir = job_results.get("output_dir")
    llm_model = None
    if output_dir:
        n13_path = os.path.join(output_dir, "n13_generate_report.json")
        if os.path.exists(n13_path):
            with open(n13_path, encoding="utf-8") as f:
                llm_model = json.load(f).get("modelo_usado")

    return {**job_results, "llm_model": llm_model}


@app.get("/llm/{job_id}")
def llm_profile(job_id: str):
    job = get_job(job_id)
    if job is None:
        return {"error": "job not found"}

    job_results = job.get("results", {})
    if not job_results:
        return {"error": "no results found"}

    output_dir = job_results.get("output_dir")
    if not output_dir:
        return {"error": "output_dir not found in results"}

    # Buscar primero el patrón específico, luego el genérico
    files = glob.glob(os.path.join(output_dir, "llm_profile_BIMODAL_*.txt"))
    if not files:
        files = glob.glob(os.path.join(output_dir, "llm_profile_*.txt"))
    if not files:
        return {"error": f"no llm profile file found in {output_dir}"}

    with open(files[0], encoding="utf-8") as f:
        return {"content": f.read()}


@app.get("/export/{job_id}")
def export_zip(job_id: str):
    job = get_job(job_id)
    if job is None:
        return {"error": "job not found"}

    job_results = job.get("results", {})
    output_dir = job_results.get("output_dir")
    if not output_dir or not os.path.exists(output_dir):
        return {"error": "output directory not found"}

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    species_clean = job_results.get("species", "export").replace(" ", "_")

    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for filepath in glob.glob(os.path.join(output_dir, "*.png")):
            zf.write(filepath, os.path.basename(filepath))
        for filepath in glob.glob(os.path.join(output_dir, "llm_profile_*.txt")):
            zf.write(filepath, os.path.basename(filepath))
        for filepath in glob.glob(os.path.join(output_dir, "n*.json")):
            zf.write(filepath, os.path.basename(filepath))

    return FileResponse(
        path=tmp.name,
        media_type="application/zip",
        filename=f"CR-BioLM_{species_clean}.zip"
    )