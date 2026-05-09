import json
import boto3
import io
import os
import subprocess
import tempfile
from PIL import Image
import fitz  # PyMuPDF
import requests
from typing import List, Dict


# Initialize AWS clients
s3_client = boto3.client('s3')

# Environment variables
BUCKET_NAME = 'vmtb-bedrock-qwen-bucket-v2'
EXTRACT_LAMBDA_API = 'https://gzgrswe52e.execute-api.ap-south-1.amazonaws.com/dev/extract'
ANONYMIZE_LAMBDA_API = 'https://trigger-ocr-service-622331214924.asia-south1.run.app/trigger'

def lambda_handler(event, context):
    """
    Main Lambda handler function
    """
    response = {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "status": "accepted",
            "message": "Request accepted and processing started"
        })
    }

    try:
        body = json.loads(event.get('body', '{}')) if isinstance(event.get('body'), str) else event.get('body', {})
        request_id = body.get('request_id')
        case_id = body.get('case_id')
        additional_data = body.get('additional_data')

        if not request_id:
            print("request_id missing")
        
        if not case_id:
            print("case_id missing")

        prefix = f"uploads/{request_id}/data/"
        print(f"Listing files from S3 path: {prefix}")

        s3_keys = list_s3_files(prefix)

        if not s3_keys:
            print("No files found in data folder")


        converted_structure = []
        failed_files = []

        for s3_key in s3_keys:
            try:
                result = process_document(s3_key)
                if result:
                    if 'text_content' in result:
                        if additional_data is None:
                            additional_data = {}

                        additional_data.setdefault('extracted_texts', []).append(
                            result['text_content']
                        )
                    else:
                        converted_structure.append(result)
                else:
                    failed_files.append(s3_key)
            except Exception as e:
                print(f"Error processing {s3_key}: {str(e)}")
                failed_files.append(s3_key)

        if not converted_structure and not additional_data:
            print("All files failed to process")

        trigger_lambda_api(
            EXTRACT_LAMBDA_API,
            request_id,
            case_id,
            additional_data,
            converted_structure
        )

        trigger_anonymize_api(
            ANONYMIZE_LAMBDA_API,
            request_id
        )

    except Exception as e:
        print(f"Lambda execution error: {str(e)}")

    return response


def list_s3_files(prefix: str) -> List[str]:
    """
    List all files under a given S3 prefix
    """
    keys = []
    paginator = s3_client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']

            if key.endswith('/'):
                continue

            filename = key.rsplit('/', 1)[-1]

            if filename.startswith("DELETE_NNCMFAGSSS_22246_"):
                print(f"Skipping deleted file: {key}")
                continue

            keys.append(key)

    print(f"Found {len(keys)} files")
    return keys


def process_document(s3_key: str) -> Dict:
    """
    Process a single document from S3
    """
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
    file_content = response['Body'].read()

    parent_path, filename = s3_key.rsplit('/', 1)
    file_name, file_ext = os.path.splitext(filename)
    file_ext = file_ext.lower()

    output_folder = f"{parent_path}/{file_name}"

    image_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'}
    doc_formats = {'.doc', '.docx', '.ppt', '.pptx', '.odt', '.rtf'}

    png_keys = []

    if file_ext == '.txt':
        try:
            text_content = file_content.decode('utf-8', errors='ignore')
            return {
                'original_key': s3_key,
                'text_content': text_content
            }
        except Exception as e:
            print(f"Error reading TXT file {s3_key}: {str(e)}")
            return None

    if file_ext in image_formats:
        png_keys = convert_image_to_png(file_content, output_folder)
    elif file_ext == '.pdf':
        png_keys = convert_pdf_to_png(file_content, output_folder)
    elif file_ext in doc_formats:
        png_keys = convert_docs_to_png(file_content, file_ext, output_folder)
    else:
        print(f"Unsupported file format: {file_ext}")
        return None

    if png_keys:
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        return {
            'original_key': s3_key,
            'output_folder': output_folder,
            'images': png_keys
        }

    return None


def convert_image_to_png(file_content: bytes, output_folder: str) -> List[str]:
    """
    Convert image to PNG format and upload to S3
    """
    try:
        image = Image.open(io.BytesIO(file_content))

        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        png_buffer = io.BytesIO()
        image.save(png_buffer, format='PNG')
        png_buffer.seek(0)

        output_key = f"{output_folder}/page_0.png"
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=output_key,
            Body=png_buffer.getvalue(),
            ContentType='image/png'
        )

        return [output_key]

    except Exception as e:
        print(f"Error converting image to PNG: {str(e)}")
        return []


def convert_pdf_to_png(file_content: bytes, output_folder: str) -> List[str]:
    """
    Convert PDF pages to PNG images and upload to S3
    """
    try:
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        png_keys = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")

            output_key = f"{output_folder}/page_{page_num}.png"
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=output_key,
                Body=img_data,
                ContentType='image/png'
            )

            png_keys.append(output_key)

        pdf_document.close()
        return png_keys

    except Exception as e:
        print(f"Error converting PDF to PNG: {str(e)}")
        return []


def convert_docs_to_png(file_content: bytes, file_ext: str, output_folder: str) -> List[str]:
    """
    Convert DOC/PPT files to PDF using LibreOffice and then PDF to PNG
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, f"input{file_ext}")
            pdf_path = os.path.join(tmpdir, "input.pdf")

            with open(input_path, "wb") as f:
                f.write(file_content)

            subprocess.run(
                [
                    "/usr/bin/soffice",
                    "--headless",
                    "--nologo",
                    "--nofirststartwizard",
                    "--nodefault",
                    "--norestore",
                    "-env:UserInstallation=file:///tmp/libreoffice-profile",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    tmpdir,
                    input_path
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60
            )

            if not os.path.exists(pdf_path):
                print("PDF conversion failed")
                return []

            with open(pdf_path, "rb") as pdf_file:
                return convert_pdf_to_png(pdf_file.read(), output_folder)

    except Exception as e:
        print(f"Error converting document to PNG: {str(e)}")
        return []


def trigger_lambda_api(api_url: str, request_id: str, case_id: str, additional_data: Dict, converted_structure: List[Dict]) -> Dict:
    """
    Trigger downstream Lambda via API Gateway
    """
    try:
        payload = {
            'request_id': request_id,
            'case_id': case_id,
            'additional_data': additional_data
        }

        response = requests.post(
            api_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )

        response.raise_for_status()

        return {
            'status_code': response.status_code,
            'response': response.json() if response.content else {}
        }

    except Exception as e:
        print(f"Error triggering {api_url}: {str(e)}")
        return {'error': str(e)}


def trigger_anonymize_api(api_url: str, request_id: str) -> Dict:
    """
    Trigger anonymize Lambda via API Gateway
    """
    try:
        payload = {
            'request_id': request_id
        }

        response = requests.post(
            api_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )

        response.raise_for_status()

        return {
            'status_code': response.status_code,
            'response': response.json() if response.content else {}
        }

    except Exception as e:
        print(f"Error triggering anonymize API {api_url}: {str(e)}")
        return {'error': str(e)}