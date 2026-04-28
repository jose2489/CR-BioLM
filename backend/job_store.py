import uuid
import json
import os

JOBS_FILE = "jobs.json"

def _load():
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE) as f:
                content = f.read()
                if content.strip():
                    return json.loads(content)
        except (json.JSONDecodeError, IOError):
            pass
    return {}

def _save(jobs):
    try:
        tmp = JOBS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(jobs, f)
        os.replace(tmp, JOBS_FILE)
    except PermissionError:
        # Fallback for Windows permission issues
        with open(JOBS_FILE, "w") as f:
            json.dump(jobs, f)

def create_job():
    jobs = _load()
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "running",
        "logs": [],
        "results": None
    }
    _save(jobs)
    print("Job created:", job_id, "| Saved to:", os.path.abspath(JOBS_FILE))
    return job_id

def add_log(job_id, message):
    jobs = _load()
    if job_id in jobs:
        jobs[job_id]["logs"].append(message)
        _save(jobs)

def set_done(job_id, results):
    jobs = _load()
    if job_id in jobs:
        jobs[job_id]["status"] = "done"
        jobs[job_id]["results"] = results
        _save(jobs)

def get_job(job_id):
    jobs = _load()
    return jobs.get(job_id)