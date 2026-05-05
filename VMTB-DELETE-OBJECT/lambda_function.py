import json
import boto3

s3 = boto3.client("s3")

BUCKET_NAME = "vmtb-bedrock-qwen-bucket-v2"
BASE_PREFIX = "uploads"
DELETE_PREFIX = "DELETE_NNCMFAGSSS_22246_"

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))

        request_id = body.get("request_id")
        files = body.get("delete_files", [])

        if not request_id or not isinstance(files, list) or not files:
            return response(400, "Invalid request body")

        renamed = []

        for filename in files:
            old_key = f"{BASE_PREFIX}/{request_id}/data/{filename}"
            new_key = f"{BASE_PREFIX}/{request_id}/data/{DELETE_PREFIX}{filename}"

            # Copy object
            s3.copy_object(
                Bucket=BUCKET_NAME,
                CopySource={"Bucket": BUCKET_NAME, "Key": old_key},
                Key=new_key
            )

            # Delete original
            s3.delete_object(
                Bucket=BUCKET_NAME,
                Key=old_key
            )

            renamed.append({
                "from": filename,
                "to": f"{DELETE_PREFIX}{filename}"
            })

        return response(200, {
            "message": "Files renamed successfully",
            "renamed": renamed
        })

    except Exception as e:
        return response(500, str(e))


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        },
        "body": json.dumps(body)
    }
