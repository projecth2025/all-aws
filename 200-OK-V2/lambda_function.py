import json
import os
import urllib.request
import urllib.error


def lambda_handler(event, context):
    try:
        # 1. Parse incoming request body
        body = event.get("body")

        if body:
            body = json.loads(body)
        else:
            body = {}

        request_id = body.get("request_id")
        case_id = body.get("case_id")
        additional_data = body.get("additional_data")

        # 2. Prepare payload for second Lambda (via API Gateway)
        payload = {
            "request_id": request_id,
            "case_id": case_id,
            "additional_data": additional_data
        }

        target_api_url = os.environ.get("TARGET_API_URL")

        if target_api_url:
            req = urllib.request.Request(
                target_api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            # 3. Trigger second Lambda (fire-and-forget style)
            try:
                urllib.request.urlopen(req, timeout=5)
            except Exception:
                # Intentionally ignore downstream failure
                pass

        # 4. Always return 200 OK
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "OK"})
        }

    except Exception:
        return {
            "statusCode": 500,
            "body": json.dumps({"message": Exception})
        }
