import os
import time
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

import google.auth
from google.auth.transport.requests import AuthorizedSession

# ---------------------------------------------------------
# APP
# ---------------------------------------------------------
app = FastAPI(title="Cloud Run Job Trigger Service (API-based)")

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
PROJECT_ID = os.environ["GCP_PROJECT"]
REGION = os.environ.get("REGION", "us-east4")
JOB_NAME = os.environ.get("JOB_NAME", "paddle-ocr-job")

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
def log(msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {msg}", flush=True)

# ---------------------------------------------------------
# AUTH SESSION (REUSED)
# ---------------------------------------------------------
credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
authed_session = AuthorizedSession(credentials)

# ---------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------
class TriggerRequest(BaseModel):
    request_id: str

# ---------------------------------------------------------
# STARTUP
# ---------------------------------------------------------
@app.on_event("startup")
def startup():
    log("Trigger service started")
    log(f"PROJECT_ID={PROJECT_ID}")
    log(f"REGION={REGION}")
    log(f"JOB_NAME={JOB_NAME}")

# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------------------------------------
# TRIGGER ENDPOINT
# ---------------------------------------------------------
@app.post("/trigger")
def trigger_job(payload: TriggerRequest, request: Request):
    internal_id = str(uuid.uuid4())
    start = time.time()

    log("=" * 80)
    log(f"NEW REQUEST | internal_id={internal_id}")
    log(f"Client: {request.client.host if request.client else 'unknown'}")
    log(f"Payload: {payload.dict()}")

    request_id = payload.request_id.strip()
    if not request_id:
        log("ERROR: Empty request_id")
        raise HTTPException(status_code=400, detail="request_id is required")

    # -----------------------------------------------------
    # CLOUD RUN JOBS EXECUTION API
    # -----------------------------------------------------
    url = (
        f"https://run.googleapis.com/v2/projects/{PROJECT_ID}"
        f"/locations/{REGION}/jobs/{JOB_NAME}:run"
    )

    body = {
        "overrides": {
            "containerOverrides": [
                {
                    "env": [
                        {"name": "REQUEST_ID", "value": request_id}
                    ]
                }
            ]
        }
    }

    log(f"POST {url}")
    log(f"Request body: {body}")

    response = authed_session.post(url, json=body)

    elapsed = time.time() - start
    log(f"HTTP status from Run API: {response.status_code}")
    log(f"Response body: {response.text}")

    if response.status_code not in (200, 201):
        log("ERROR: Job execution API call failed")
        raise HTTPException(
            status_code=500,
            detail=response.text
        )

    log(f"SUCCESS: Job triggered | request_id={request_id}")
    log(f"Total handling time: {elapsed:.2f}s")
    log("=" * 80)

    return {
        "status": "job_triggered",
        "job_name": JOB_NAME,
        "request_id": request_id,
        "internal_request_id": internal_id,
        "latency_seconds": round(elapsed, 2),
    }
    