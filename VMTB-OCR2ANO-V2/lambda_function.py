import json
import io
import re
import os
import boto3
from datetime import datetime
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import create_client, Client

# =========================
# CONFIG
# =========================
AWS_REGION = "ap-south-1"
MODEL_ID = "qwen.qwen3-235b-a22b-2507-v1:0"
S3_BUCKET = "vmtb-bedrock-qwen-bucket-v2"
ANONYMIZED_PREFIX = "ANO_NNCMFAGSSS_22246_"

# Supabase config
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

# =========================
# CLIENTS
# =========================
s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# Initialize Supabase client
supabase: Client = None
if SUPABASE_URL and SUPABASE_ANON_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        print(f"[{datetime.utcnow().isoformat()}] Supabase client initialized successfully")
    except Exception as e:
        print(f"[{datetime.utcnow().isoformat()}] WARNING: Failed to initialize Supabase client: {str(e)}")

# =========================
# LOGGING
# =========================
def log(msg):
    print(f"[{datetime.utcnow().isoformat()}] {msg}")

# =========================
# SUPABASE UPDATE
# =========================
def update_report_status(request_id, status="unverified"):
    """
    Update report_status in Supabase cases table
    """
    if not supabase:
        log("ERROR: Supabase client not initialized")
        raise Exception("Supabase client not initialized. Check environment variables.")
    
    try:
        log(f"Updating Supabase: request_id={request_id}, report_status={status}")
        
        response = supabase.table("cases").update({
            "report_status": status
        }).eq("request_id", request_id).execute()
        
        if response.data:
            log(f"✓ Supabase updated successfully for request_id: {request_id}")
            return True
        else:
            log(f"WARNING: No rows updated in Supabase for request_id: {request_id}")
            return False
            
    except Exception as e:
        log(f"ERROR: Supabase update failed: {str(e)}")
        raise

# =========================
# HELPER: CHECK IF FOLDER IS ALREADY ANONYMIZED
# =========================
def is_already_anonymized(folder_name):
    """
    Check if folder name contains ANONYMIZED_PREFIX anywhere in the name (continuously)
    Returns True if already anonymized, False otherwise
    """
    return ANONYMIZED_PREFIX in folder_name

# =========================
# HELPER: CHECK IF IT'S A FILE (NOT A FOLDER)
# =========================
def is_existing_file(doc_name):
    """
    Check if the document name represents an existing file (e.g., ends with .pdf)
    Returns True if it's a file, False if it's a folder
    """
    # Check if it has a file extension like .pdf, .doc, .docx, etc.
    return doc_name.lower().endswith(('.pdf', '.doc', '.docx', '.txt', '.jpg', '.jpeg', '.png'))

# =========================
# HELPER: GET FOLDER NAME FOR OCR MATCHING
# =========================
def get_ocr_matching_name(folder_name):
    """
    Remove .pdf extension from folder name to match with OCR document keys
    OCR keys don't have .pdf extension
    """
    # Remove .pdf extension if present
    if folder_name.lower().endswith('.pdf'):
        return folder_name[:-4]
    return folder_name

# =========================
# HELPER: DELETE FOLDER FROM S3
# =========================
def delete_s3_folder(request_id, folder_name):
    """
    Delete an entire folder (all contents) from S3
    """
    folder_prefix = f"uploads/{request_id}/data/{folder_name}/"
    log(f"  Deleting folder from S3: {folder_prefix}")
    
    try:
        # List all objects in the folder
        continuation_token = None
        delete_count = 0
        
        while True:
            if continuation_token:
                response = s3.list_objects_v2(
                    Bucket=S3_BUCKET,
                    Prefix=folder_prefix,
                    ContinuationToken=continuation_token
                )
            else:
                response = s3.list_objects_v2(
                    Bucket=S3_BUCKET,
                    Prefix=folder_prefix
                )
            
            if 'Contents' in response:
                # Delete objects in batches
                objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
                
                if objects_to_delete:
                    s3.delete_objects(
                        Bucket=S3_BUCKET,
                        Delete={'Objects': objects_to_delete}
                    )
                    delete_count += len(objects_to_delete)
            
            # Check if there are more objects to list
            if response.get('IsTruncated'):
                continuation_token = response.get('NextContinuationToken')
            else:
                break
        
        log(f"  ✓ Deleted folder with {delete_count} objects")
        return True
    except Exception as e:
        log(f"  ERROR: Failed to delete folder: {str(e)}")
        return False

# =========================
# HELPER: LIST FOLDERS IN S3 DATA DIRECTORY
# =========================
def list_folders_in_data(request_id):
    """
    List all folders (and files) in the data directory
    Returns: list of folder/file names
    """
    prefix = f"uploads/{request_id}/data/"
    log(f"Listing contents in: s3://{S3_BUCKET}/{prefix}")
    
    try:
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix,
            Delimiter='/'
        )
        
        folders = []
        
        # Get folders (CommonPrefixes)
        if 'CommonPrefixes' in response:
            for prefix_obj in response['CommonPrefixes']:
                folder_path = prefix_obj['Prefix']
                # Extract folder name (remove prefix and trailing slash)
                folder_name = folder_path.replace(prefix, '').rstrip('/')
                folders.append(folder_name)
        
        # Get files in root data directory
        if 'Contents' in response:
            for obj in response['Contents']:
                file_path = obj['Key']
                # Skip the directory itself
                if file_path == prefix:
                    continue
                # Extract file name
                file_name = file_path.replace(prefix, '')
                # Only add if it's a direct child (no slashes)
                if '/' not in file_name:
                    folders.append(file_name)
        
        log(f"Found {len(folders)} items in data directory: {folders}")
        return folders
        
    except Exception as e:
        log(f"ERROR: Failed to list folders: {str(e)}")
        return []

# =========================
# LLM CALL WITH ROBUST ERROR HANDLING
# =========================
def call_llm(entities, max_retries=2):
    """
    Call Qwen model to identify PII in OCR text
    entities: [{id, text}]
    Returns: list of PII detections
    """
    if not entities:
        return []
    
    numbered_text = "\n".join(f"{e['id']}. {e['text']}" for e in entities)

    prompt = f"""You are a medical document privacy classifier.

Below is OCR text extracted from a single page. Each line has a number.

Rules:
- Identify ONLY PII (names, IDs, doctor names, hospitals, addresses, phone, email, URLs)
- DO NOT mark clinical information as PII (age, gender, diseases, medications, symptoms, treatments, test results, dates)

Return ONLY valid JSON array. No explanations.

Format:
[
  {{"id": 1, "pii_type": "FULL"}},
  {{"id": 2, "pii_type": "PARTIAL", "substrings": ["exact text"]}}
]

If no PII found, return: []

Text:
{numbered_text}
"""

    for attempt in range(max_retries):
        try:
            response = bedrock.converse(
                modelId=MODEL_ID,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "temperature": 0.1,
                    "maxTokens": 2048,
                    "topP": 0.9
                }
            )
            
            raw_text = response["output"]["message"]["content"][0]["text"].strip()
            
            # Clean markdown wrappers
            if raw_text.startswith("```"):
                raw_text = re.sub(r"```(?:json)?\s*\n?", "", raw_text)
                raw_text = re.sub(r"\n?```\s*$", "", raw_text)
            
            raw_text = raw_text.strip()
            
            # Try to parse JSON
            try:
                result = json.loads(raw_text)
                
                if not isinstance(result, list):
                    log(f"    WARNING: LLM returned non-list (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        continue
                    return []
                
                # Validate each item
                validated = []
                for item in result:
                    if isinstance(item, dict) and "id" in item and "pii_type" in item:
                        validated.append(item)
                
                return validated
                
            except json.JSONDecodeError as e:
                log(f"    JSON parse error (attempt {attempt + 1}): {str(e)}")
                log(f"    Raw response preview: {raw_text[:200]}...")
                
                # Try to fix common JSON issues
                fixed_text = fix_json_string(raw_text)
                if fixed_text != raw_text:
                    try:
                        result = json.loads(fixed_text)
                        if isinstance(result, list):
                            log(f"    Successfully fixed JSON!")
                            return result
                    except:
                        pass
                
                if attempt < max_retries - 1:
                    continue
                else:
                    log(f"    All parse attempts failed")
                    return []
        
        except Exception as e:
            log(f"    LLM API error (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                continue
            else:
                return []
    
    return []

def fix_json_string(text):
    """Attempt to fix common JSON issues"""
    # Remove trailing commas
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # Try to find JSON array boundaries
    start = text.find('[')
    end = text.rfind(']')
    
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    
    return text

# =========================
# CHUNKED LLM PROCESSING
# =========================
def call_llm_chunked(entities, chunk_size=30):
    """
    Process entities in chunks to avoid token limits and malformed JSON
    """
    all_results = []
    
    for i in range(0, len(entities), chunk_size):
        chunk = entities[i:i + chunk_size]
        log(f"    Processing entities {i+1}-{min(i+chunk_size, len(entities))} of {len(entities)}")
        
        chunk_results = call_llm(chunk)
        all_results.extend(chunk_results)
    
    return all_results

# =========================
# PARALLEL PAGE PROCESSING (WITH ANONYMIZATION)
# =========================
def process_single_page_with_anonymization(request_id, doc_name, page_name, entities, page_index, total_pages):
    """
    Process a single page: call LLM and anonymize PII
    Returns: (anonymized_image, pii_count, page_name)
    """
    log(f"  [{page_index}/{total_pages}] Processing page: {page_name} ({len(entities)} entities)")
    
    # Add .png extension to page_name for S3 path
    page_file_name = f"{page_name}.png"
    page_key = f"uploads/{request_id}/data/{doc_name}/{page_file_name}"
    
    # Load image from S3
    try:
        image_obj = s3.get_object(Bucket=S3_BUCKET, Key=page_key)
        image_bytes = image_obj["Body"].read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
        log(f"    Image loaded: {image.size[0]}x{image.size[1]} pixels")
    except Exception as e:
        log(f"    ERROR loading image from s3://{S3_BUCKET}/{page_key}: {str(e)}")
        # Use blank image as fallback
        image = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(image)

    # Call LLM to identify PII
    llm_input = [{"id": e["id"], "text": e["text"]} for e in entities]
    
    if llm_input:
        # Use chunking if more than 40 entities
        if len(llm_input) > 40:
            log(f"    Using chunked processing for {len(llm_input)} entities")
            pii_results = call_llm_chunked(llm_input, chunk_size=30)
        else:
            pii_results = call_llm(llm_input)
        
        log(f"    Found {len(pii_results)} PII entities")
    else:
        pii_results = []

    # Map id -> OCR entity
    entity_map = {e["id"]: e for e in entities}

    # Anonymize detected PII
    anonymized_count = 0
    for pii in pii_results:
        entity = entity_map.get(pii["id"])
        if not entity:
            log(f"    WARNING: PII id {pii['id']} not found in entities")
            continue

        bbox = polygon_to_bbox(entity["bbox"])

        if pii["pii_type"] == "FULL":
            draw.rectangle(bbox, fill="white")
            anonymized_count += 1
        
        elif pii["pii_type"] == "PARTIAL":
            for sub in pii.get("substrings", []):
                try:
                    sub_bbox = compute_partial_bbox(
                        entity["text"], sub, bbox
                    )
                    draw.rectangle(sub_bbox, fill="white")
                    anonymized_count += 1
                except Exception as e:
                    log(f"    WARNING: Partial bbox failed for '{sub}': {str(e)}")
    
    if anonymized_count > 0:
        log(f"    ✓ Anonymized {anonymized_count} regions on this page")
    else:
        log(f"    ✓ No anonymization needed for this page")

    return (image, len(pii_results), page_name)

# =========================
# PARALLEL PAGE PROCESSING (WITHOUT ANONYMIZATION - JUST LOAD)
# =========================
def process_single_page_without_anonymization(request_id, doc_name, page_file_name, page_index, total_pages):
    """
    Process a single page: just load the image without any LLM calls or anonymization
    page_file_name should already include .png extension
    Returns: (image, 0, page_file_name)
    """
    log(f"  [{page_index}/{total_pages}] Loading page: {page_file_name} (already anonymized, skipping LLM)")
    
    page_key = f"uploads/{request_id}/data/{doc_name}/{page_file_name}"
    
    # Load image from S3
    try:
        image_obj = s3.get_object(Bucket=S3_BUCKET, Key=page_key)
        image_bytes = image_obj["Body"].read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        log(f"    Image loaded: {image.size[0]}x{image.size[1]} pixels")
    except Exception as e:
        log(f"    ERROR loading image from s3://{S3_BUCKET}/{page_key}: {str(e)}")
        # Use blank image as fallback
        image = Image.new("RGB", (1000, 1000), "white")
    
    return (image, 0, page_file_name)

# =========================
# HELPER: LIST PNG FILES IN FOLDER
# =========================
def list_png_files_in_folder(request_id, folder_name):
    """
    List all PNG files in a specific folder
    Returns: sorted list of PNG filenames
    """
    folder_prefix = f"uploads/{request_id}/data/{folder_name}/"
    
    try:
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=folder_prefix
        )
        
        png_files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                file_path = obj['Key']
                file_name = file_path.replace(folder_prefix, '')
                if file_name.lower().endswith('.png'):
                    png_files.append(file_name)
        
        # Sort by page number
        png_files.sort(key=extract_page_number)
        
        log(f"  Found {len(png_files)} PNG files in folder '{folder_name}'")
        return png_files
        
    except Exception as e:
        log(f"  ERROR listing PNG files: {str(e)}")
        return []

# =========================
# GEOMETRY HELPERS
# =========================
def polygon_to_bbox(poly):
    """Convert polygon to bounding box [x1, y1, x2, y2]"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]

def compute_partial_bbox(full_text, substring, bbox):
    """Calculate bbox for a substring within full text"""
    if substring not in full_text:
        return bbox
    
    x1, y1, x2, y2 = bbox
    total_chars = max(len(full_text), 1)
    char_width = (x2 - x1) / total_chars

    start = full_text.index(substring)
    end = start + len(substring)

    return [
        int(x1 + start * char_width),
        y1,
        int(x1 + end * char_width),
        y2
    ]

# =========================
# PAGE SORTING
# =========================
def extract_page_number(page_name):
    """
    Extract page number from 'page_0.png', 'page_1.png', etc.
    Returns integer for sorting
    """
    match = re.search(r'page_(\d+)', page_name)
    if match:
        return int(match.group(1))
    return 0

# =========================
# MAIN HANDLER
# =========================
def lambda_handler(event, context):
    log("=" * 80)
    log("ANONYMIZATION LAMBDA STARTED")
    log("=" * 80)
    
    try:
        # Get request_id
        body = json.loads(event["body"])

        request_id = body["request_id"]
        if not request_id:
            log("ERROR: Missing request_id in event")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing request_id"})
            }
        
        log(f"Request ID: {request_id}")

        # Load OCR results
        ocr_key = f"uploads/{request_id}/results/ocr_full.json"
        log(f"Loading OCR results from: s3://{S3_BUCKET}/{ocr_key}")
        
        try:
            ocr_obj = s3.get_object(Bucket=S3_BUCKET, Key=ocr_key)
            ocr_data = json.loads(ocr_obj["Body"].read())
        except Exception as e:
            log(f"ERROR: Failed to load OCR results: {str(e)}")
            raise
        
        ocr_documents = ocr_data.get("documents", {})
        log(f"OCR contains {len(ocr_documents)} documents")
        log(f"OCR document keys: {list(ocr_documents.keys())}")

        # List actual folders/files in S3 data directory
        data_items = list_folders_in_data(request_id)
        
        if not data_items:
            log("WARNING: No folders or files found in data directory")
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "No folders or files to process",
                    "request_id": request_id
                })
            }

        documents_processed = 0
        total_pages_processed = 0
        total_pii_found = 0
        files_skipped = 0

        # Process EACH item in data directory
        for item_index, item_name in enumerate(data_items, 1):
            log("-" * 80)
            log(f"[{item_index}/{len(data_items)}] Processing item: '{item_name}'")
            
            # =========================
            # SKIP EXISTING FILES (ONLY PROCESS FOLDERS)
            # =========================
            if is_existing_file(item_name):
                log(f"  ⚠ SKIPPING: '{item_name}' is an existing file (not a folder)")
                log(f"  → This file will be left untouched")
                files_skipped += 1
                continue
            
            log(f"  ✓ '{item_name}' is a folder, proceeding with processing")
            
            # =========================
            # CHECK IF FOLDER IS ALREADY ANONYMIZED
            # =========================
            already_anonymized = is_already_anonymized(item_name)
            
            if already_anonymized:
                log(f"  ✓ Folder '{item_name}' is ALREADY ANONYMIZED (contains prefix)")
                log(f"  → Skipping LLM calls and OCR matching, will just merge pages into PDF")
                
                # List PNG files in the folder
                png_files = list_png_files_in_folder(request_id, item_name)
                
                if not png_files:
                    log(f"  WARNING: No PNG files found in folder '{item_name}', skipping")
                    continue
                
                # Process pages WITHOUT anonymization (just load images)
                log(f"  Loading {len(png_files)} pages without anonymization...")
                
                page_results = []
                max_workers = min(10, len(png_files))
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_page = {}
                    
                    for page_index, png_file in enumerate(png_files, 1):
                        future = executor.submit(
                            process_single_page_without_anonymization,
                            request_id,
                            item_name,
                            png_file,
                            page_index,
                            len(png_files)
                        )
                        future_to_page[future] = (page_index, png_file)
                    
                    for future in as_completed(future_to_page):
                        page_index, png_file = future_to_page[future]
                        try:
                            image, pii_count, _ = future.result()
                            page_results.append((page_index, image, pii_count))
                            log(f"  ✓ Page {page_index} loaded successfully")
                        except Exception as e:
                            log(f"  ERROR: Page {page_index} ({png_file}) failed: {str(e)}")
                            blank_image = Image.new("RGB", (1000, 1000), "white")
                            page_results.append((page_index, blank_image, 0))
                
                # Sort and extract images
                page_results.sort(key=lambda x: x[0])
                anonymized_images = [result[1] for result in page_results]
                total_pages_processed += len(anonymized_images)
                
                # Use existing name (already has prefix)
                anonymized_doc_name = item_name
                
            else:
                log(f"  → Folder '{item_name}' is NOT anonymized")
                log(f"  → Will perform LLM calls and anonymization")
                
                # Match folder name with OCR document keys
                ocr_matching_name = get_ocr_matching_name(item_name)
                log(f"  Matching folder name '{ocr_matching_name}' with OCR documents...")
                
                if ocr_matching_name not in ocr_documents:
                    log(f"  ERROR: No OCR data found for folder '{item_name}' (tried '{ocr_matching_name}')")
                    log(f"  Available OCR keys: {list(ocr_documents.keys())}")
                    continue
                
                pages = ocr_documents[ocr_matching_name]
                log(f"  ✓ Found OCR data with {len(pages)} pages")
                
                if not pages:
                    log(f"  WARNING: No pages in OCR data for '{item_name}', skipping")
                    continue
                
                # Sort pages by page number
                sorted_pages = sorted(pages.items(), key=lambda x: extract_page_number(x[0]))
                log(f"  Sorted page order: {[p[0] for p in sorted_pages]}")
                
                # Process pages WITH anonymization
                log(f"  Starting parallel processing of {len(sorted_pages)} pages with anonymization...")
                
                page_results = []
                max_workers = min(10, len(sorted_pages))
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_page = {}
                    
                    for page_index, (page_name, entities) in enumerate(sorted_pages, 1):
                        future = executor.submit(
                            process_single_page_with_anonymization,
                            request_id,
                            item_name,
                            page_name,
                            entities,
                            page_index,
                            len(sorted_pages)
                        )
                        future_to_page[future] = (page_index, page_name)
                    
                    for future in as_completed(future_to_page):
                        page_index, page_name = future_to_page[future]
                        try:
                            image, pii_count, _ = future.result()
                            page_results.append((page_index, image, pii_count))
                            total_pii_found += pii_count
                            log(f"  ✓ Page {page_index} completed with {pii_count} PII entities")
                        except Exception as e:
                            log(f"  ERROR: Page {page_index} ({page_name}) failed: {str(e)}")
                            blank_image = Image.new("RGB", (1000, 1000), "white")
                            page_results.append((page_index, blank_image, 0))
                
                # Sort and extract images
                page_results.sort(key=lambda x: x[0])
                anonymized_images = [result[1] for result in page_results]
                total_pages_processed += len(anonymized_images)
                
                # Add prefix to name
                anonymized_doc_name = f"{ANONYMIZED_PREFIX}{item_name}"

            # =========================
            # BUILD PDF FROM IMAGES
            # =========================
            if not anonymized_images:
                log(f"  ERROR: No images to convert for folder '{item_name}'")
                continue
            
            log(f"  Creating PDF from {len(anonymized_images)} pages...")
            
            try:
                pdf_buf = io.BytesIO()
                anonymized_images[0].save(
                    pdf_buf,
                    format="PDF",
                    save_all=True,
                    append_images=anonymized_images[1:] if len(anonymized_images) > 1 else []
                )
                pdf_buf.seek(0)
                
                log(f"  PDF created successfully ({pdf_buf.getbuffer().nbytes} bytes)")
            except Exception as e:
                log(f"  ERROR creating PDF: {str(e)}")
                continue

            # Upload PDF to S3
            pdf_key = f"uploads/{request_id}/data/{anonymized_doc_name}.pdf"
            log(f"  Uploading PDF to: s3://{S3_BUCKET}/{pdf_key}")
            
            try:
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key=pdf_key,
                    Body=pdf_buf,
                    ContentType="application/pdf"
                )
                log(f"  ✓ PDF uploaded successfully")
            except Exception as e:
                log(f"  ERROR uploading PDF: {str(e)}")
                continue

            # =========================
            # DELETE THE ENTIRE FOLDER
            # =========================
            log(f"  Deleting folder '{item_name}'...")
            delete_s3_folder(request_id, item_name)
            
            log(f"  ✓ Folder '{item_name}' processed successfully!")
            
            documents_processed += 1

        # =========================
        # UPDATE SUPABASE
        # =========================
        log("=" * 80)
        log("UPDATING SUPABASE DATABASE")
        log("=" * 80)
        
        try:
            update_report_status(request_id, "unverified")
            log("✓ Supabase database updated successfully")
        except Exception as e:
            log(f"ERROR: Failed to update Supabase: {str(e)}")
            return {
                "statusCode": 500,
                "body": json.dumps({
                    "error": f"Anonymization completed but Supabase update failed: {str(e)}",
                    "request_id": request_id
                })
            }

        # Final summary
        log("=" * 80)
        log("ANONYMIZATION COMPLETED SUCCESSFULLY")
        log(f"Folders processed: {documents_processed}")
        log(f"Files skipped (existing PDFs/files): {files_skipped}")
        log(f"Total pages processed: {total_pages_processed}")
        log(f"Total PII entities found: {total_pii_found}")
        log(f"Supabase report_status updated to: unverified")
        log("=" * 80)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Anonymization completed successfully",
                "request_id": request_id,
                "folders_processed": documents_processed,
                "files_skipped": files_skipped,
                "total_pages": total_pages_processed,
                "total_pii_found": total_pii_found,
                "supabase_updated": True,
                "report_status": "unverified"
            })
        }

    except Exception as e:
        log("=" * 80)
        log(f"FATAL ERROR: {str(e)}")
        import traceback
        log(traceback.format_exc())
        log("=" * 80)
        
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }