import os
import json
import time
import uuid
import shutil
import gc
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import boto3
import requests

# ---------------------------------------------------------------------
# ENV SAFETY
# ---------------------------------------------------------------------
os.environ["HOME"] = "/opt/paddle_models"
os.environ["TMPDIR"] = "/tmp"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------------------------------------------------------
# BASIC CONFIG
# ---------------------------------------------------------------------
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
S3_BUCKET = os.environ.get("S3_BUCKET")
DATA_PREFIX = "uploads"
LAMBDA_TRIGGER_URL = "https://gzgrswe52e.execute-api.ap-south-1.amazonaws.com/dev/ocr2ano"

if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET environment variable is required")

# ---------------------------------------------------------------------
# LOGGING HELPER
# ---------------------------------------------------------------------
def log(msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {msg}", flush=True)

# ---------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------
app = FastAPI(
    title="PaddleOCR PP-OCRv5 GPU Service",
    description="High-accuracy OCR using PP-OCRv5 with GPU",
    version="1.0.0"
)

# ---------------------------------------------------------------------
# AWS CLIENT (global, reused)
# ---------------------------------------------------------------------
s3 = boto3.client("s3", region_name=AWS_REGION)

# ---------------------------------------------------------------------
# LAZY OCR INIT - Load on first request (when GPU is available)
# ---------------------------------------------------------------------
ocr: Optional[Any] = None
ocr_lock = threading.Lock()

def initialize_ocr():
    """Initialize PaddleOCR - called on first request."""
    global ocr
    
    if ocr is not None:
        return
    
    with ocr_lock:
        if ocr is not None:
            return
        
        log("Initializing PaddleOCR with PP-OCRv5 (first request)...")
        try:
            import paddle
            from paddleocr import PaddleOCR
            
            gpu_count = paddle.device.cuda.device_count() if hasattr(paddle.device, 'cuda') else 0
            log(f"PaddlePaddle version: {paddle.__version__}")
            log(f"GPU available: {gpu_count > 0}, GPU count: {gpu_count}")
            
            # PaddleOCR 3.x: Disable problematic doc preprocessing
            ocr = PaddleOCR(
                lang='en',
                ocr_version='PP-OCRv5',
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
            gc.collect()
            log("PaddleOCR PP-OCRv5 initialized successfully")
        except Exception as e:
            log(f"FATAL: PaddleOCR initialization failed: {str(e)}")
            raise

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def list_all_images_recursively(prefix: str) -> List[str]:
    """
    List all PNG image files from S3 bucket with given prefix.
    Skips folders with prefix "ANO_NNCMFAGSSS_22246_".
    Only returns .png files.
    """
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    
    try:
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                
                # Skip folders with the specified prefix
                if "ANO_NNCMFAGSSS_22246_" in key:
                    log(f"Skipping file in ANO folder: {key}")
                    continue
                
                # Only accept .png files (case-insensitive)
                if key.lower().endswith(".png"):
                    keys.append(key)
                    
    except Exception as e:
        log(f"ERROR listing S3 objects: {str(e)}")
        raise
    
    return keys

def parse_s3_key(key: str, base_prefix: str) -> tuple[str, str]:
    """
    Parse S3 key to extract document and page identifiers.
    Expected format: uploads/{request_id}/data/{doc}/{page}.png
    """
    parts = key.replace(base_prefix + "/", "").split("/")
    
    if len(parts) < 4:
        return "unknown_doc", "unknown_page"
    
    doc = parts[-2]
    page = os.path.splitext(parts[-1])[0]
    
    return doc, page

def extract_ocr_result(result_obj, image_idx: int) -> List[Dict[str, Any]]:
    """
    Extract OCR results from PaddleOCR 3.x result object.
    
    PaddleOCR 3.x returns a result object with .json attribute containing:
    - res: Container with actual OCR data
      - dt_polys: Detection bounding boxes
      - rec_texts: Recognized text strings
      - rec_scores: Confidence scores for recognition
    """
    entities = []
    
    try:
        # DEBUG: Log the result object type
        log(f"[{image_idx}] Result object type: {type(result_obj)}")
        
        # PaddleOCR 3.x: Access the .json attribute
        if hasattr(result_obj, 'json'):
            result_dict = result_obj.json
            log(f"[{image_idx}] Accessed .json attribute, type: {type(result_dict)}")
        elif isinstance(result_obj, dict):
            result_dict = result_obj
            log(f"[{image_idx}] Result is already dict")
        else:
            log(f"[{image_idx}] WARNING: Unexpected result format: {type(result_obj)}")
            return entities
        
        # Extract OCR data from the result dictionary
        if isinstance(result_dict, dict):
            # PaddleOCR 3.x wraps everything in a 'res' key
            if 'res' in result_dict:
                ocr_data = result_dict['res']
                log(f"[{image_idx}] Found 'res' key, extracting OCR data...")
            else:
                ocr_data = result_dict
            
            # Now extract dt_polys, rec_texts, rec_scores from ocr_data
            if 'dt_polys' in ocr_data and 'rec_texts' in ocr_data:
                polys = ocr_data.get('dt_polys', [])
                texts = ocr_data.get('rec_texts', [])
                scores = ocr_data.get('rec_scores', [])
                
                log(f"[{image_idx}] Found OCR data: {len(texts)} text(s), {len(polys)} bbox(es), {len(scores)} score(s)")
                
                for i, text in enumerate(texts):
                    bbox = polys[i] if i < len(polys) else []
                    score = scores[i] if i < len(scores) else 0.0
                    
                    # Convert numpy arrays to lists
                    if hasattr(bbox, 'tolist'):
                        bbox = bbox.tolist()
                    
                    entities.append({
                        "id": i + 1,
                        "text": str(text),
                        "confidence": float(score),
                        "bbox": bbox
                    })
                
                log(f"[{image_idx}] Successfully extracted {len(entities)} entities")
            else:
                log(f"[{image_idx}] WARNING: 'dt_polys' or 'rec_texts' not found in OCR data")
                log(f"[{image_idx}] Available keys in ocr_data: {list(ocr_data.keys())}")
    
    except Exception as e:
        log(f"[{image_idx}] ERROR in extract_ocr_result: {str(e)}")
        import traceback
        log(f"[{image_idx}] Traceback: {traceback.format_exc()}")
    
    return entities

def trigger_lambda(request_id: str) -> bool:
    """
    Trigger the next Lambda function after OCR completion.
    
    Args:
        request_id: The request ID to pass to the Lambda function
        
    Returns:
        True if successful, False otherwise
    """
    try:
        log(f"Triggering Lambda function for request_id: {request_id}")
        
        payload = {
            "request_id": request_id
        }
        
        response = requests.post(
            LAMBDA_TRIGGER_URL,
            json=payload,
            timeout=30
        )
        
        log(f"Lambda trigger response: status={response.status_code}, body={response.text[:500]}")
        
        if response.status_code in [200, 201, 202, 204]:
            log(f"Lambda triggered successfully for request_id: {request_id}")
            return True
        else:
            log(f"WARNING: Lambda trigger returned status {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        log(f"WARNING: Lambda trigger timeout for request_id: {request_id}")
        return False
    except Exception as e:
        log(f"ERROR triggering Lambda: {str(e)}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        return False

# ---------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    """Health check endpoint for Cloud Run."""
    try:
        import paddle
        gpu_count = paddle.device.cuda.device_count() if hasattr(paddle.device, 'cuda') else 0
    except:
        gpu_count = 0
    
    return {
        "status": "ok",
        "ocr_ready": ocr is not None,
        "model": "PP-OCRv5",
        "hardware": "GPU" if gpu_count > 0 else "CPU",
        "gpu_count": gpu_count,
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/ocr")
@app.post("/ocr")
async def run_ocr(request: Request) -> Dict[str, Any]:
    """
    Main OCR endpoint.
    
    Expected payload:
    {
        "request_id": "optional-unique-id"
    }
    
    Images are read from: s3://{S3_BUCKET}/uploads/{request_id}/data/**/*.png
    (Skips folders with prefix "ANO_NNCMFAGSSS_22246_")
    Results written to: s3://{S3_BUCKET}/uploads/{request_id}/results/ocr_full.json
    Triggers Lambda: POST to LAMBDA_TRIGGER_URL with request_id (ALWAYS, regardless of OCR status)
    """
    # Initialize OCR on first request
    if ocr is None:
        initialize_ocr()
    
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    request_id = payload.get("request_id") or str(uuid.uuid4())
    log(f"OCR started | request_id={request_id}")
    
    data_prefix = f"{DATA_PREFIX}/{request_id}/data"
    result_prefix = f"{DATA_PREFIX}/{request_id}/results"
    
    # Variables to track processing status
    images_processed = 0
    result_s3_uri = None
    ocr_status = "no_images_found"
    
    try:
        # -----------------------------------------------------------------
        # LIST IMAGES
        # -----------------------------------------------------------------
        log(f"Listing PNG images from s3://{S3_BUCKET}/{data_prefix}")
        log(f"Skipping folders with prefix: ANO_NNCMFAGSSS_22246_")
        image_keys = list_all_images_recursively(data_prefix)
        log(f"Found {len(image_keys)} PNG image(s) for processing")
        
        if not image_keys:
            log(f"No PNG images found for request_id: {request_id} (after filtering)")
            # Still create an empty result file
            output = {
                "request_id": request_id,
                "total_images": 0,
                "documents": {},
                "model": "PP-OCRv5",
                "version": "1.0.0",
                "status": "no_images_found",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            result_path = f"/tmp/ocr_{request_id}.json"
            with open(result_path, "w") as f:
                json.dump(output, f, indent=2)
            
            s3_key_out = f"{result_prefix}/ocr_full.json"
            log(f"Uploading empty result to s3://{S3_BUCKET}/{s3_key_out}")
            s3.upload_file(result_path, S3_BUCKET, s3_key_out)
            log("Empty result upload completed")
            
            if os.path.exists(result_path):
                os.remove(result_path)
            
            result_s3_uri = f"s3://{S3_BUCKET}/{s3_key_out}"
            ocr_status = "no_images_found"
            images_processed = 0
            
        else:
            # -----------------------------------------------------------------
            # LOCAL WORK DIR
            # -----------------------------------------------------------------
            workdir = f"/tmp/{request_id}"
            os.makedirs(workdir, exist_ok=True)
            documents = defaultdict(dict)
            
            try:
                # -------------------------------------------------------------
                # DOWNLOAD + OCR
                # -------------------------------------------------------------
                for idx, key in enumerate(image_keys, start=1):
                    filename = os.path.basename(key)
                    local_path = os.path.join(workdir, filename)
                    
                    log(f"[{idx}/{len(image_keys)}] Downloading {key}")
                    s3.download_file(S3_BUCKET, key, local_path)
                    log(f"[{idx}/{len(image_keys)}] Downloaded to {local_path}")
                    
                    doc, page = parse_s3_key(key, DATA_PREFIX)
                    log(f"[{idx}/{len(image_keys)}] OCR starting | doc={doc} page={page}")
                    
                    t0 = time.time()
                    
                    # -------------------- OCR CALL --------------------
                    try:
                        log(f"[{idx}/{len(image_keys)}] Calling ocr.predict()...")
                        result_gen = ocr.predict(local_path)
                        log(f"[{idx}/{len(image_keys)}] ocr.predict() returned, type: {type(result_gen)}")
                        
                        # Handle generator or list
                        if hasattr(result_gen, '__iter__') and not isinstance(result_gen, (str, dict)):
                            results = list(result_gen)
                            log(f"[{idx}/{len(image_keys)}] Converted to list, length: {len(results)}")
                            result_obj = results[0] if results else None
                        else:
                            result_obj = result_gen
                        
                        if result_obj is None:
                            log(f"[{idx}/{len(image_keys)}] WARNING: Empty OCR result")
                            entities = []
                        else:
                            log(f"[{idx}/{len(image_keys)}] Extracting entities from result...")
                            entities = extract_ocr_result(result_obj, idx)
                            
                    except Exception as ocr_error:
                        log(f"[{idx}/{len(image_keys)}] OCR failed: {str(ocr_error)}")
                        import traceback
                        log(f"[{idx}/{len(image_keys)}] Traceback: {traceback.format_exc()}")
                        raise
                    # --------------------------------------------------
                    
                    elapsed = time.time() - t0
                    log(f"[{idx}/{len(image_keys)}] OCR finished | time={elapsed:.2f}s | texts={len(entities)}")
                    
                    documents[doc][page] = entities
                    
                    # Cleanup
                    os.remove(local_path)
                    log(f"[{idx}/{len(image_keys)}] Cleaned up local file")
                    
                    del result_gen, result_obj, entities
                    gc.collect()
                
                # -------------------------------------------------------------
                # WRITE RESULT JSON
                # -------------------------------------------------------------
                output = {
                    "request_id": request_id,
                    "total_images": len(image_keys),
                    "documents": dict(documents),
                    "model": "PP-OCRv5",
                    "version": "1.0.0",
                    "status": "success",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                result_path = f"/tmp/ocr_{request_id}.json"
                with open(result_path, "w") as f:
                    json.dump(output, f, indent=2)
                
                s3_key_out = f"{result_prefix}/ocr_full.json"
                log(f"Uploading result to s3://{S3_BUCKET}/{s3_key_out}")
                s3.upload_file(result_path, S3_BUCKET, s3_key_out)
                log("Upload completed")
                
                if os.path.exists(result_path):
                    os.remove(result_path)
                
                result_s3_uri = f"s3://{S3_BUCKET}/{s3_key_out}"
                images_processed = len(image_keys)
                ocr_status = "success"
                
            finally:
                if os.path.exists(workdir):
                    shutil.rmtree(workdir, ignore_errors=True)
                    log("Work directory cleaned")
                
                gc.collect()
        
        # -------------------------------------------------------------
        # ALWAYS TRIGGER LAMBDA FUNCTION (regardless of OCR status)
        # -------------------------------------------------------------
        log(f"Triggering next Lambda function for request_id: {request_id} (status: {ocr_status})")
        lambda_success = trigger_lambda(request_id)
        
        if lambda_success:
            log(f"Lambda triggered successfully for request_id: {request_id}")
        else:
            log(f"WARNING: Lambda trigger failed for request_id: {request_id}")
        
        return {
            "status": ocr_status,
            "request_id": request_id,
            "images_processed": images_processed,
            "result_s3_uri": result_s3_uri,
            "lambda_triggered": lambda_success
        }
    
    except Exception as e:
        log(f"ERROR during OCR | request_id={request_id} | error={str(e)}")
        import traceback
        log(f"Full traceback: {traceback.format_exc()}")
        
        # STILL TRIGGER LAMBDA EVEN ON ERROR
        log(f"Triggering Lambda despite error for request_id: {request_id}")
        lambda_success = trigger_lambda(request_id)
        
        if lambda_success:
            log(f"Lambda triggered successfully despite error for request_id: {request_id}")
        else:
            log(f"WARNING: Lambda trigger also failed for request_id: {request_id}")
        
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )