import json
import boto3

s3 = boto3.client("s3")

BUCKET_NAME = "vmtb-bedrock-qwen-bucket-v2"
URL_EXPIRATION = 900  # 15 minutes
DELETE_PREFIX = "DELETE_NNCMFAGSSS_22246_"

def lambda_handler(event, context):
    try:
        params = event.get("queryStringParameters") or {}
        request_id = params.get("request_id")

        if not request_id:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "request_id is required"})
            }

        prefix = f"uploads/{request_id}/data/"

        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=prefix
        )

        if "Contents" not in response:
            return {
                "statusCode": 200,
                "body": json.dumps({"files": []})
            }

        files = []

        for obj in response["Contents"]:
            key = obj["Key"]

            if key.endswith("/"):
                continue

            filename = key.split("/")[-1]

            if filename.startswith(DELETE_PREFIX):
                continue

            url = s3.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": BUCKET_NAME,
                    "Key": key
                },
                ExpiresIn=URL_EXPIRATION
            )

            files.append({
                "filename": filename,
                "url": url
            })

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"files": files})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
