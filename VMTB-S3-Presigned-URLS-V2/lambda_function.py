import json
import boto3
import uuid

s3 = boto3.client("s3")

BUCKET_NAME = "vmtb-bedrock-qwen-bucket-v2"
URL_EXPIRATION_SECONDS = 900  # 15 minutes


def generate_presigned_post(bucket: str, prefix: str):
    return s3.generate_presigned_post(
        Bucket=bucket,
        Key=f"{prefix}${{filename}}",
        Fields={
            "acl": "private"
        },
        Conditions=[
            ["starts-with", "$key", prefix],
            ["content-length-range", 1, 50_000_000]  # 50 MB per file (optional)
        ],
        ExpiresIn=URL_EXPIRATION_SECONDS
    )


def lambda_handler(event, context):
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Folder-like prefix
        prefix = f"uploads/{request_id}/data/"

        presigned_post = generate_presigned_post(
            BUCKET_NAME,
            prefix
        )

        response = {
            "request_id": request_id,
            "upload": {
                "bucket": BUCKET_NAME,
                "prefix": prefix,
                "url": presigned_post["url"],
                "fields": presigned_post["fields"]
            }
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps(response)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }
