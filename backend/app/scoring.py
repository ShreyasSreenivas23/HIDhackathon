from app.models import Decision


def decide(identity_score: float, liveness_score: float, spoof_score: float) -> tuple[Decision, float, list[str]]:
    # Lower spoof_score is better.
    confidence = max(0.0, min(1.0, 0.45 * identity_score + 0.45 * liveness_score + 0.10 * (1.0 - spoof_score)))
    reasons: list[str] = []

    if spoof_score > 0.60:
        reasons.append("High spoof artifact score")
    if identity_score < 0.70:
        reasons.append("Weak identity match")
    if liveness_score < 0.70:
        reasons.append("Weak liveness confidence")

    if confidence > 0.85 and identity_score > 0.80 and spoof_score < 0.40:
        return Decision.ALLOW, confidence, reasons or ["Strong identity and liveness"]

    if confidence < 0.55 or identity_score < 0.55:
        return Decision.DENY, confidence, reasons or ["Confidence below deny threshold"]

    return Decision.REQUEST_VERIFICATION, confidence, reasons or ["Borderline confidence requires step-up"]
