import os
import json
import time
import shutil
import gc
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional

import boto3
import requests

# ---------------------------------------------------------------------
# ENV SAFETY (UNCHANGED)
# ---------------------------------------------------------------------
os.environ["HOME"] = "/opt/paddle_models"
os.environ["TMPDIR"] = "/tmp"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------------------------------------------------------
# BASIC CONFIG (UNCHANGED)
# ---------------------------------------------------------------------
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
S3_BUCKET = os.environ.get("S3_BUCKET")
DATA_PREFIX = "uploads"
LAMBDA_TRIGGER_URL = os.environ.get(
    "LAMBDA_TRIGGER_URL",
    "https://gzgrswe52e.execute-api.ap-south-1.amazonaws.com/dev/ocr2ano"
)
REQUEST_ID = os.environ.get("REQUEST_ID")

if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET environment variable is required")

if not REQUEST_ID:
    raise RuntimeError("REQUEST_ID environment variable is required")

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------
def log(msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {msg}", flush=True)

# ---------------------------------------------------------------------
# AWS CLIENT
# ---------------------------------------------------------------------
s3 = boto3.client("s3", region_name=AWS_REGION)

# ---------------------------------------------------------------------
# OCR INIT (UNCHANGED LOGIC)
# ---------------------------------------------------------------------
ocr: Optional[Any] = None
ocr_lock = threading.Lock()

def initialize_ocr():
    global ocr
    if ocr is not None:
        return

    with ocr_lock:
        if ocr is not None:
            return

        log("Initializing PaddleOCR PP-OCRv5...")
        import paddle
        from paddleocr import PaddleOCR

        gpu_count = paddle.device.cuda.device_count() if hasattr(paddle.device, "cuda") else 0
        log(f"Paddle version: {paddle.__version__}")
        log(f"GPU available: {gpu_count > 0}, count={gpu_count}")

        ocr = PaddleOCR(
            lang="en",
            ocr_version="PP-OCRv5",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False
        )

        gc.collect()
        log("PaddleOCR initialized")

# ---------------------------------------------------------------------
# HELPERS (UNCHANGED)
# ---------------------------------------------------------------------
def list_all_images_recursively(prefix: str) -> List[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if "ANO_NNCMFAGSSS_22246_" in key:
                log(f"Skipping ANO file: {key}")
                continue

            if key.lower().endswith(".png"):
                keys.append(key)

    return keys

def parse_s3_key(key: str, base_prefix: str):
    parts = key.replace(base_prefix + "/", "").split("/")
    if len(parts) < 4:
        return "unknown_doc", "unknown_page"
    return parts[-2], os.path.splitext(parts[-1])[0]

def extract_ocr_result(result_obj, image_idx: int) -> List[Dict[str, Any]]:
    entities = []

    if not result_obj:
        return entities

    result_dict = result_obj.json if hasattr(result_obj, "json") else result_obj
    ocr_data = result_dict.get("res", result_dict)

    polys = ocr_data.get("dt_polys", [])
    texts = ocr_data.get("rec_texts", [])
    scores = ocr_data.get("rec_scores", [])

    for i, text in enumerate(texts):
        bbox = polys[i] if i < len(polys) else []
        score = scores[i] if i < len(scores) else 0.0

        if hasattr(bbox, "tolist"):
            bbox = bbox.tolist()

        entities.append({
            "id": i + 1,
            "text": str(text),
            "confidence": float(score),
            "bbox": bbox
        })

    return entities

def trigger_lambda(request_id: str) -> bool:
    try:
        log(f"Triggering Lambda | request_id={request_id}")
        r = requests.post(
            LAMBDA_TRIGGER_URL,
            json={"request_id": request_id},
            timeout=30
        )
        return r.status_code in (200, 201, 202, 204)
    except Exception as e:
        log(f"Lambda trigger failed: {e}")
        return False

# ---------------------------------------------------------------------
# MAIN JOB LOGIC (UNCHANGED)
# ---------------------------------------------------------------------
def run_ocr_job(request_id: str):
    initialize_ocr()
    log(f"OCR job started | request_id={request_id}")

    data_prefix = f"{DATA_PREFIX}/{request_id}/data"
    result_prefix = f"{DATA_PREFIX}/{request_id}/results"

    image_keys = list_all_images_recursively(data_prefix)
    documents = defaultdict(dict)

    if image_keys:
        workdir = f"/tmp/{request_id}"
        os.makedirs(workdir, exist_ok=True)

        for idx, key in enumerate(image_keys, 1):
            local_path = os.path.join(workdir, os.path.basename(key))
            s3.download_file(S3_BUCKET, key, local_path)

            doc, page = parse_s3_key(key, DATA_PREFIX)
            result = list(ocr.predict(local_path))[0]
            documents[doc][page] = extract_ocr_result(result, idx)

            os.remove(local_path)
            gc.collect()

        shutil.rmtree(workdir, ignore_errors=True)

    output = {
        "request_id": request_id,
        "total_images": len(image_keys),
        "documents": dict(documents),
        "model": "PP-OCRv5",
        "version": "1.0.0",
        "status": "success" if image_keys else "no_images_found",
        "timestamp": datetime.utcnow().isoformat()
    }

    result_path = f"/tmp/ocr_{request_id}.json"
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2)

    s3_key = f"{result_prefix}/ocr_full.json"
    s3.upload_file(result_path, S3_BUCKET, s3_key)
    os.remove(result_path)

    trigger_lambda(request_id)
    log("OCR job completed successfully")

# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_ocr_job(REQUEST_ID)