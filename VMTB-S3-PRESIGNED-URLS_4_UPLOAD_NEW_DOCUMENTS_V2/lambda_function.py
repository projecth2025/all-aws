import json
import boto3

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
            ["content-length-range", 1, 50_000_000]
        ],
        ExpiresIn=URL_EXPIRATION_SECONDS
    )


def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        request_id = body.get("request_id")

        if not request_id:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "request_id is required"})
            }

        prefix = f"uploads/{request_id}/data/"

        presigned_post = generate_presigned_post(
            BUCKET_NAME,
            prefix
        )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "request_id": request_id,
                "upload": {
                    "bucket": BUCKET_NAME,
                    "prefix": prefix,
                    "url": presigned_post["url"],
                    "fields": presigned_post["fields"]
                }
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
