import json
from datetime import datetime, timezone
from decimal import Decimal

import boto3

from app.config import settings


class AwsClients:
    def __init__(self) -> None:
        self.rekognition = boto3.client("rekognition", region_name=settings.aws_region)
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        self.s3 = boto3.client("s3", region_name=settings.aws_region)
        self.dynamodb = boto3.resource("dynamodb", region_name=settings.aws_region)


clients = AwsClients()


def fetch_enrolled_face(employee_id: str) -> bytes:
    response = clients.s3.get_object(
        Bucket=settings.enrolled_faces_bucket,
        Key=f"{employee_id}.jpg",
    )
    return response["Body"].read()


def log_audit_event(
    session_id: str,
    employee_id: str,
    decision: str,
    confidence: float,
    latency_ms: int,
    identity_score: float,
    liveness_score: float,
    spoof_score: float,
    reasons: list[str],
) -> None:
    table = clients.dynamodb.Table(settings.ddb_table)
    table.put_item(
        Item={
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "employee_id": employee_id,
            "decision": decision,
            "confidence": Decimal(str(round(confidence, 4))),
            "latency_ms": latency_ms,
            "identity_score": Decimal(str(round(identity_score, 4))),
            "liveness_score": Decimal(str(round(liveness_score, 4))),
            "spoof_score": Decimal(str(round(spoof_score, 4))),
            "reasons": reasons,
        }
    )


def compare_faces(source_image_bytes: bytes, target_image_bytes: bytes) -> float:
    response = clients.rekognition.compare_faces(
        SourceImage={"Bytes": source_image_bytes},
        TargetImage={"Bytes": target_image_bytes},
        SimilarityThreshold=0,
    )
    matches = response.get("FaceMatches", [])
    if not matches:
        return 0.0
    return float(matches[0].get("Similarity", 0.0) / 100.0)


def detect_face_quality(image_bytes: bytes) -> float:
    response = clients.rekognition.detect_faces(Image={"Bytes": image_bytes}, Attributes=["ALL"])
    details = response.get("FaceDetails", [])
    if not details:
        return 0.0

    face = details[0]
    quality = face.get("Quality", {})
    brightness = float(quality.get("Brightness", 0.0) / 100.0)
    sharpness = float(quality.get("Sharpness", 0.0) / 100.0)
    eyes_open = 1.0 if face.get("EyesOpen", {}).get("Value", False) else 0.5
    return min(1.0, 0.45 * brightness + 0.45 * sharpness + 0.10 * eyes_open)


def bedrock_step_up_instruction(prior_reasons: list[str]) -> str:
    prompt = (
        "Generate one short liveness challenge requiring motion and asymmetry. "
        "Return only one sentence. Context: " + ", ".join(prior_reasons)
    )
    body = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"temperature": 0.2, "maxTokens": 60},
    }

    response = clients.bedrock_runtime.invoke_model(
        modelId=settings.bedrock_model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(response["body"].read())
    output = payload.get("output", {}).get("message", {}).get("content", [])
    if not output:
        return "Raise your right hand beside your face and blink once."

    text = output[0].get("text", "").strip()
    if not text:
        return "Turn slightly left and smile for one second."
    return text