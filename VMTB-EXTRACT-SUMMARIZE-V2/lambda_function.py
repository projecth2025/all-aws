import boto3

s3 = boto3.client("s3")

PROMPT_BUCKET = "vmtb-bedrock-qwen-bucket-v2"
PROMPT_PREFIX = "prompts/"

def load_prompt_from_s3(filename: str) -> str:
    response = s3.get_object(
        Bucket=PROMPT_BUCKET,
        Key=f"{PROMPT_PREFIX}{filename}"
    )
    return response["Body"].read().decode("utf-8")

OCR_PROMPT = load_prompt_from_s3("ocr_prompt.txt")
SUMMARY_PROMPT = load_prompt_from_s3("summary_prompt.txt")

import json
import boto3
from botocore.config import Config
import io
import os
import uuid
import time
import re
import requests
from typing import TypedDict, List, Dict, Any, Annotated
from PIL import Image
from langgraph.graph import StateGraph, END
from langsmith import traceable
import operator
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import create_client, Client

# ============================================================================
# Configuration
# ============================================================================
BUCKET = "vmtb-bedrock-qwen-bucket-v2"
MODEL_ID_VL = "qwen.qwen3-vl-235b-a22b"  # For OCR/extraction
MODEL_ID_TEXT = "qwen.qwen3-235b-a22b-2507-v1:0"  # For summarization
BATCH_SIZE = 1
# ANONYMIZE_API = "https://gzgrswe52e.execute-api.ap-south-1.amazonaws.com/dev/anonymize"

# Bedrock Pricing
INPUT_PRICE_PER_1K_TOKENS = 0.00053
OUTPUT_PRICE_PER_1K_TOKENS = 0.00266

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# ============================================================================
# Clients
# ============================================================================
boto_config = Config(
    read_timeout=1000,
    connect_timeout=10,
    retries={'max_attempts': 3, 'mode': 'adaptive'}
)

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1", config=boto_config)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ============================================================================
# State Definition
# ============================================================================
class GraphState(TypedDict):
    clinical_data: Dict[str, List[str]]
    additional_data: str
    ocr_results: Annotated[Dict[str, str], operator.or_]
    timing_metrics: Annotated[Dict[str, Any], operator.or_]
    cost_metrics: Annotated[Dict[str, Any], operator.or_]
    final_summary: str
    pipeline_start_time: float
    pipeline_end_time: float
    intermediate_ocr_combined: str
    request_id: str
    case_id: str

# ============================================================================
# Supabase Service
# ============================================================================
def update_case_summary(case_id: str, summary_text: str) -> None:
    """Update case summary after ML processing completion."""
    supabase.table("cases").update({
        "summary": summary_text,
        "ai_generated_summary": summary_text,
        "summary_status": "unverified"
    }).eq("id", case_id).execute()

# ============================================================================
# Helper Functions
# ============================================================================
def download_from_s3(bucket: str, key: str) -> str:
    """Download file from S3 to /tmp."""
    local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(key)}"
    s3.download_file(bucket, key, local_path)
    return local_path

def image_to_bytes(path: str) -> bytes:
    """Convert and optimize image to raw bytes for Bedrock."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((800, 800), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()

def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost based on token usage."""
    input_cost = (input_tokens / 1000) * INPUT_PRICE_PER_1K_TOKENS
    output_cost = (output_tokens / 1000) * OUTPUT_PRICE_PER_1K_TOKENS
    return round(input_cost + output_cost, 6)

@traceable(name="bedrock_converse")
def call_bedrock(prompt: str, image_paths: List[str] = None, model_id: str = MODEL_ID_VL) -> Dict[str, Any]:
    """Call Bedrock with optional images."""
    content = []
    
    if image_paths:
        for image_path in image_paths:
            image_bytes = image_to_bytes(image_path)
            content.append({
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": image_bytes}
                }
            })
    
    content.append({"text": prompt})
    
    response = bedrock.converse(
        modelId=model_id,
        messages=[{
            "role": "user",
            "content": content
        }],
        inferenceConfig={
            "maxTokens": 4096,
            "temperature": 0.2
        }
    )
    
    usage = response.get("usage", {})
    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    
    return {
        "text": response["output"]["message"]["content"][0]["text"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": calculate_cost(input_tokens, output_tokens)
    }

def list_documents_from_s3(request_id: str) -> Dict[str, List[str]]:
    """
    List all documents and their pages from S3.
    Returns: Dict[document_name, List[s3_keys]] sorted by page number
    """
    prefix = f"uploads/{request_id}/data/"
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    
    if 'Contents' not in response:
        return {}
    
    documents = {}
    
    for obj in response['Contents']:
        key = obj['Key']
        
        # Skip if not a PNG file
        if not key.endswith('.png'):
            continue
        
        # Extract document name and page number
        # Format: uploads/{request_id}/data/{document_name}/page_{N}.png
        parts = key.replace(prefix, '').split('/')
        if len(parts) != 2:
            continue
        
        document_name = parts[0]
        page_file = parts[1]
        
        # Extract page number
        match = re.match(r'page_(\d+)\.png', page_file)
        if not match:
            continue
        
        page_num = int(match.group(1))
        
        if document_name not in documents:
            documents[document_name] = []
        
        documents[document_name].append((page_num, key))
    
    # Sort pages by page number for each document
    for doc_name in documents:
        documents[doc_name].sort(key=lambda x: x[0])
        documents[doc_name] = [key for _, key in documents[doc_name]]
    
    return documents

def process_single_batch(batch_s3_keys: List[str], batch_info: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single batch of images."""
    batch_id = batch_info["batch_id"]
    doc_id = batch_info["doc_id"]
    total_pages = batch_info["total_pages"]
    page_start = batch_info["page_start"]
    page_end = batch_info["page_end"]
    
    print(f"  [Doc {doc_id}] Batch {batch_id}: Processing pages {page_start}-{page_end}")
    
    batch_start_time = time.time()
    
    # Download images for this batch
    batch_local_images = []
    for s3_key in batch_s3_keys:
        local_path = download_from_s3(BUCKET, s3_key)
        batch_local_images.append(local_path)
    
    batch_prompt = f"""{OCR_PROMPT}

Context: This is a {total_pages}-page clinical document. You are processing pages {page_start} to {page_end}.
Extract all text exactly as it appears on these pages."""
    
    try:
        result = call_bedrock(batch_prompt, batch_local_images)
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        
        print(f"  [Doc {doc_id}] Batch {batch_id}: Completed in {batch_duration:.2f}s "
              f"(tokens: {result['input_tokens']}in/{result['output_tokens']}out, "
              f"cost: ${result['cost']:.6f})")
        
        return {
            "batch_id": batch_id,
            "text": result["text"],
            "page_range": f"{page_start}-{page_end}",
            "duration_seconds": round(batch_duration, 2),
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cost_usd": result["cost"]
        }
    
    finally:
        for local_path in batch_local_images:
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
            except Exception as e:
                print(f"  [Doc {doc_id}] Batch {batch_id}: Warning - could not remove {local_path}: {e}")

# ============================================================================
# Graph Nodes
# ============================================================================
@traceable(name="start_node")
def start_node(state: GraphState) -> Dict:
    """Initialize pipeline and record start time."""
    print(f"Starting pipeline with {len(state['clinical_data'])} documents")
    return {
        "pipeline_start_time": time.time()
    }

@traceable(name="ocr_document")
def ocr_document_node(state: GraphState, document_id: str) -> Dict:
    """Process a single document with parallel batch processing."""
    doc_start_time = time.time()
    print(f"\n[Document {document_id}] Starting OCR")
    
    image_keys = state["clinical_data"][document_id]
    total_pages = len(image_keys)
    print(f"[Document {document_id}] Total pages: {total_pages}")
    
    batches = []
    batch_results = []
    
    for i in range(0, total_pages, BATCH_SIZE):
        batch_s3_keys = image_keys[i:i + BATCH_SIZE]
        batch_id = (i // BATCH_SIZE) + 1
        page_start = i + 1
        page_end = min(i + len(batch_s3_keys), total_pages)
        
        batches.append({
            "batch_id": batch_id,
            "doc_id": document_id,
            "s3_keys": batch_s3_keys,
            "total_pages": total_pages,
            "page_start": page_start,
            "page_end": page_end
        })
    
    total_batches = len(batches)
    print(f"[Document {document_id}] Split into {total_batches} batch(es)")
    print(f"[Document {document_id}] Starting parallel batch processing...")
    
    with ThreadPoolExecutor(max_workers=total_batches) as executor:
        future_to_batch = {
            executor.submit(process_single_batch, batch["s3_keys"], batch): batch
            for batch in batches
        }
        
        for future in as_completed(future_to_batch):
            try:
                batch_result = future.result()
                batch_results.append(batch_result)
            except Exception as e:
                batch = future_to_batch[future]
                print(f"  [Doc {document_id}] Batch {batch['batch_id']} FAILED: {e}")
                raise
    
    batch_results.sort(key=lambda x: x["batch_id"])
    
    combined_text = "\n\n".join([
        f"=== Pages {br['page_range']} ===\n{br['text']}"
        for br in batch_results
    ])
    
    doc_end_time = time.time()
    doc_duration = doc_end_time - doc_start_time
    
    total_input_tokens = sum(br["input_tokens"] for br in batch_results)
    total_output_tokens = sum(br["output_tokens"] for br in batch_results)
    total_cost = sum(br["cost_usd"] for br in batch_results)
    max_batch_duration = max(br["duration_seconds"] for br in batch_results)
    
    print(f"[Document {document_id}] ✓ Completed in {doc_duration:.2f}s")
    print(f"[Document {document_id}] Max batch duration: {max_batch_duration:.2f}s")
    print(f"[Document {document_id}] Total cost: ${total_cost:.6f}")
    
    return {
        "ocr_results": {
            document_id: combined_text
        },
        "timing_metrics": {
            document_id: {
                "document_duration_seconds": round(doc_duration, 2),
                "max_batch_duration_seconds": max_batch_duration,
                "total_batches": total_batches,
                "batch_details": [
                    {
                        "batch_id": br["batch_id"],
                        "page_range": br["page_range"],
                        "duration_seconds": br["duration_seconds"]
                    }
                    for br in batch_results
                ]
            }
        },
        "cost_metrics": {
            document_id: {
                "total_cost_usd": round(total_cost, 6),
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "batch_costs": [
                    {
                        "batch_id": br["batch_id"],
                        "page_range": br["page_range"],
                        "cost_usd": br["cost_usd"],
                        "input_tokens": br["input_tokens"],
                        "output_tokens": br["output_tokens"]
                    }
                    for br in batch_results
                ]
            }
        }
    }

@traceable(name="summarization_node")
def summarization_node(state: GraphState) -> Dict:
    """Combine all OCR results and produce final summary."""
    summary_start_time = time.time()
    print("\n[Summarization] Starting final summarization")
    
    ocr_results = state.get("ocr_results", {})
    additional_data = state.get("additional_data", "").strip()
    
    if not ocr_results and not additional_data:
        print("[Summarization] No OCR data and no additional text. Skipping summarization.")
        return {
            "final_summary": "no data",
            "intermediate_ocr_combined": "No OCR data extracted",
            "pipeline_end_time": time.time(),
            "timing_metrics": {
                "summarization": {
                    "duration_seconds": 0
                }
            },
            "cost_metrics": {
                "summarization": {
                    "cost_usd": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            }
        }
    
    print(f"[Summarization] Documents processed: {len(ocr_results)}")
    
    # Format extracted documents in order
    extracted_docs = []
    clinical_data = state.get("clinical_data", {})
    
    # Sort document names (preserve original order from S3)
    doc_names_ordered = sorted(clinical_data.keys())
    
    for doc_name in doc_names_ordered:
        if doc_name in ocr_results and ocr_results[doc_name].strip():
            extracted_docs.append(f"--- Document: {doc_name} ---\n{ocr_results[doc_name]}\n")
    
    extracted_documents_str = "\n".join(extracted_docs)
    
    intermediate_combined = f"""INTERMEDIATE OCR EXTRACTION
{'='*80}

EXTRACTED DOCUMENTS:
{extracted_documents_str}

{'='*80}

ADDITIONAL DATA:
{additional_data if additional_data else "(None provided)"}
"""
    
    prompt = SUMMARY_PROMPT.format(
        extracted_documents=extracted_documents_str,
        additional_data=additional_data
    )
    
    result = call_bedrock(prompt, image_paths=None, model_id=MODEL_ID_TEXT)
    
    summary_duration = time.time() - summary_start_time
    
    print(f"[Summarization] ✓ Completed in {summary_duration:.2f}s")
    print(f"[Summarization] Cost: ${result['cost']:.6f}")
    
    return {
        "final_summary": result["text"],
        "pipeline_end_time": time.time(),
        "intermediate_ocr_combined": intermediate_combined,
        "timing_metrics": {
            "summarization": {
                "duration_seconds": round(summary_duration, 2)
            }
        },
        "cost_metrics": {
            "summarization": {
                "cost_usd": result["cost"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"]
            }
        }
    }

# ============================================================================
# Dynamic Graph Construction
# ============================================================================
def build_parallel_ocr_graph(clinical_data: Dict[str, List[str]]) -> StateGraph:
    """Dynamically build a LangGraph with parallel OCR nodes."""
    graph = StateGraph(GraphState)
    
    graph.add_node("start", start_node)
    
    document_ids = [
        doc_id for doc_id, pages in clinical_data.items()
        if pages and len(pages) > 0
    ]
    
    for doc_id in document_ids:
        node_name = f"ocr_doc_{doc_id}"
        
        def create_ocr_node(document_id: str):
            @traceable(name=f"ocr_document_{document_id}")
            def node_func(state: GraphState) -> Dict:
                return ocr_document_node(state, document_id)
            return node_func
        
        graph.add_node(node_name, create_ocr_node(doc_id))
    
    graph.add_node("summarize", summarization_node)
    
    for doc_id in document_ids:
        graph.add_edge("start", f"ocr_doc_{doc_id}")
    
    for doc_id in document_ids:
        graph.add_edge(f"ocr_doc_{doc_id}", "summarize")
    
    graph.add_edge("summarize", END)
    graph.set_entry_point("start")
    
    return graph.compile()

# ============================================================================
# Main Execution Function
# ============================================================================
@traceable(name="parallel_ocr_pipeline")
def run_parallel_ocr_pipeline(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the parallel OCR pipeline with full metrics."""
    clinical_data = input_data.get("clinical_data", {})
    additional_data = input_data.get("additional_data", "").strip()
    request_id = input_data.get("request_id", "")
    case_id = input_data.get("case_id", "")
    
    print(f"\n{'='*80}")
    print("PARALLEL OCR PIPELINE STARTED")
    print(f"{'='*80}")
    print(f"Request ID: {request_id}")
    print(f"Case ID: {case_id}")
    print(f"Total Documents: {len(clinical_data)}")
    for doc_id, pages in clinical_data.items():
        print(f"  Document {doc_id}: {len(pages)} page(s)")
    print(f"Batch Size: {BATCH_SIZE} pages per batch")
    print(f"{'='*80}\n")
    
    has_any_images = any(
        pages and len(pages) > 0
        for pages in clinical_data.values()
    )
    
    if not has_any_images:
        if additional_data:
            print("[Pipeline] No images found. Running summarization-only pipeline.")
            summary_start = time.time()
            
            result = call_bedrock(
                SUMMARY_PROMPT.format(
                    extracted_documents="",
                    additional_data=additional_data
                ),
                image_paths=None,
                model_id=MODEL_ID_TEXT
            )
            
            summary_duration = time.time() - summary_start
            
            return {
                "final_summary": result["text"],
                "intermediate_ocr_combined": f"ADDITIONAL DATA:\n{additional_data}",
                "metadata": {
                    "timing": {
                        "total_pipeline_duration_seconds": round(summary_duration, 2),
                        "summarization_duration_seconds": round(summary_duration, 2)
                    },
                    "cost": {
                        "total_cost_usd": result["cost"],
                        "summarization_cost_usd": result["cost"],
                        "total_input_tokens": result["input_tokens"],
                        "total_output_tokens": result["output_tokens"]
                    },
                    "processing_info": {
                        "note": "Summarization-only run (no images provided)",
                        "total_documents": len(clinical_data),
                        "model_id_vl": MODEL_ID_VL,
                        "model_id_text": MODEL_ID_TEXT
                    }
                }
            }
        
        print("[Pipeline] No images and no additional text. Returning no data.")
        return {
            "final_summary": "no data",
            "intermediate_ocr_combined": "No data provided",
            "metadata": {
                "processing_info": {
                    "note": "No images and no additional text provided"
                }
            }
        }
    
    app = build_parallel_ocr_graph(clinical_data)
    
    initial_state = {
        "clinical_data": clinical_data,
        "additional_data": additional_data,
        "ocr_results": {},
        "timing_metrics": {},
        "cost_metrics": {},
        "final_summary": "",
        "pipeline_start_time": 0.0,
        "pipeline_end_time": 0.0,
        "intermediate_ocr_combined": "",
        "request_id": request_id,
        "case_id": case_id
    }
    
    final_state = app.invoke(initial_state)
    
    total_pipeline_duration = (
        final_state["pipeline_end_time"] - final_state["pipeline_start_time"]
    )
    
    doc_durations = [
        final_state["timing_metrics"][doc_id]["document_duration_seconds"]
        for doc_id, pages in clinical_data.items()
        if pages
    ]
    max_document_duration = max(doc_durations) if doc_durations else 0
    
    total_ocr_cost = sum(
        final_state["cost_metrics"][doc_id]["total_cost_usd"]
        for doc_id, pages in clinical_data.items()
        if pages
    )
    
    summary_cost = final_state["cost_metrics"]["summarization"]["cost_usd"]
    total_cost = total_ocr_cost + summary_cost
    
    total_input_tokens = (
        sum(
            final_state["cost_metrics"][doc_id]["input_tokens"]
            for doc_id, pages in clinical_data.items()
            if pages
        )
        + final_state["cost_metrics"]["summarization"]["input_tokens"]
    )
    
    total_output_tokens = (
        sum(
            final_state["cost_metrics"][doc_id]["output_tokens"]
            for doc_id, pages in clinical_data.items()
            if pages
        )
        + final_state["cost_metrics"]["summarization"]["output_tokens"]
    )
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED")
    print(f"{'='*80}")
    print(f"Total Pipeline Duration: {total_pipeline_duration:.2f}s")
    print(f"Max Document Duration: {max_document_duration:.2f}s")
    print(f"Total Cost: ${total_cost:.6f}")
    print(f"Total Tokens: {total_input_tokens} input / {total_output_tokens} output")
    print(f"{'='*80}\n")
    
    return {
        "final_summary": final_state["final_summary"],
        "intermediate_oc_combined": final_state.get("intermediate_ocr_combined", ""),
        "metadata": {
            "timing": {
                "total_pipeline_duration_seconds": round(total_pipeline_duration, 2),
                "max_document_duration_seconds": round(max_document_duration, 2),
                "summarization_duration_seconds": final_state["timing_metrics"]["summarization"]["duration_seconds"],
                "per_document": {
                    doc_id: final_state["timing_metrics"][doc_id]
                    for doc_id, pages in clinical_data.items()
                    if pages
                }
            },
            "cost": {
                "total_cost_usd": round(total_cost, 6),
                "ocr_cost_usd": round(total_ocr_cost, 6),
                "summarization_cost_usd": round(summary_cost, 6),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "per_document": {
                    doc_id: final_state["cost_metrics"][doc_id]
                    for doc_id, pages in clinical_data.items()
                    if pages
                },
                "summarization": final_state["cost_metrics"]["summarization"]
            },
            "processing_info": {
                "total_documents": len(clinical_data),
                "batch_size": BATCH_SIZE,
                "model_id_vl": MODEL_ID_VL,
                "model_id_text": MODEL_ID_TEXT,
                "parallelism": "Documents process in parallel; batches within documents process in parallel"
            }
        }
    }

# ============================================================================
# Lambda Handler
# ============================================================================
def lambda_handler(event, context):
    """AWS Lambda handler (API Gateway compatible)."""
    try:
        # =========================
        # Parse API Gateway input
        # =========================
        if "body" not in event or not event["body"]:
            raise ValueError("Request body is required")
        
        body = json.loads(event["body"])
        request_id = body["request_id"]
        case_id = body["case_id"]
        additional_data = body.get("additional_data", "")
        
        print(f"\n{'='*80}")
        print(f"Processing Request ID: {request_id}")
        print(f"Case ID: {case_id}")
        print(f"{'='*80}\n")
        
        # =========================
        # List documents from S3
        # =========================
        clinical_data = list_documents_from_s3(request_id)
        
        print(f"Found {len(clinical_data)} documents in S3:")
        for doc_name, pages in clinical_data.items():
            print(f"  - {doc_name}: {len(pages)} pages")
        
        # =========================
        # Run pipeline
        # =========================
        input_data = {
            "clinical_data": clinical_data,
            "additional_data": additional_data,
            "request_id": request_id,
            "case_id": case_id
        }
        
        result = run_parallel_ocr_pipeline(input_data)
        
        # =========================
        # Save intermediate step
        # =========================
        intermediate_text = result.get("intermediate_oc_combined", "")
        print(f"Intermediate OCR text length: {len(intermediate_text)}")
        
        intermediate_key = f"uploads/{request_id}/results/intermediate_step.json"
        s3.put_object(
            Bucket=BUCKET,
            Key=intermediate_key,
            Body=json.dumps(
                {"intermediate_ocr_combined": intermediate_text},
                indent=2
            ),
            ContentType="application/json"
        )
        print(f"✓ Saved intermediate step to s3://{BUCKET}/{intermediate_key}")
        
        # =========================
        # Save final summary
        # =========================
        final_key = f"uploads/{request_id}/results/final_summary.json"
        s3.put_object(
            Bucket=BUCKET,
            Key=final_key,
            Body=json.dumps({
                "final_summary": result["final_summary"],
                "metadata": result.get("metadata", {})
            }, indent=2),
            ContentType="application/json"
        )
        print(f"✓ Saved final summary to s3://{BUCKET}/{final_key}")
        
        # =========================
        # Update Supabase
        # =========================
        update_case_summary(case_id, result["final_summary"])
        print(f"✓ Updated Supabase case {case_id}")
        
        # =========================
        # Trigger anonymize API
        # =========================
        # anonymize_response = requests.post(
        #     ANONYMIZE_API,
        #     json={"request_id": request_id},
        #     headers={"Content-Type": "application/json"}
        # )
        # print(f"✓ Triggered anonymize API: {anonymize_response.status_code}")
        
        print(f"\n✓ Job {request_id} completed successfully")
        print(f"  Duration: {result['metadata']['timing']['total_pipeline_duration_seconds']}s")
        print(f"  Cost: ${result['metadata']['cost']['total_cost_usd']}")
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "started",
                "request_id": request_id
            })
        }
    
    except Exception as e:
        print(f"\n✗ Job failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if "request_id" in locals():
            error_key = f"uploads/{request_id}/results/error.json"
            s3.put_object(
                Bucket=BUCKET,
                Key=error_key,
                Body=json.dumps({
                    "error": str(e),
                    "error_type": type(e).__name__
                }, indent=2),
                ContentType="application/json"
            )
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }

