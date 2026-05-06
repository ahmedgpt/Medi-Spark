"""/api/predict — accepts symptom payload, publishes to Kafka, returns ack."""
from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request
from flask_login import current_user, login_required

from app.models.symptom_log import SymptomLog
from app.services.kafka_producer import get_producer

predict_bp = Blueprint("predict", __name__)
log = logging.getLogger(__name__)


@predict_bp.route("/predict", methods=["POST"])
@login_required
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()

    raw_text = (payload.get("symptoms_text") or "").strip()
    symptoms = payload.get("symptoms") or _split_text(raw_text)
    duration_days = int(payload.get("duration_days") or 1)
    age = int(payload["age"]) if str(payload.get("age", "")).isdigit() else current_user.age
    language = (payload.get("language") or "en").lower()

    if not symptoms:
        return jsonify({"error": "No symptoms supplied."}), 400

    log_id = SymptomLog.create(
        user_id=current_user.id,
        symptoms=symptoms,
        duration_days=duration_days,
        age=age,
        language=language,
        raw_text=raw_text,
    )

    event = {
        "log_id": log_id,
        "user_id": current_user.id,
        "symptoms": symptoms,
        "duration_days": duration_days,
        "age": age,
        "language": language,
        "raw_text": raw_text,
    }

    try:
        get_producer().send(
            current_app.config["TOPIC_SYMPTOM_INPUT"], value=event, key=current_user.id
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Kafka publish failed: %s", exc)
        return jsonify({"error": "Streaming layer unavailable.", "log_id": log_id}), 503

    # Week 1 returns an ack only. Week 2 will block until a prediction-result is ready.
    return jsonify(
        {
            "status": "queued",
            "log_id": log_id,
            "message": "Symptoms received. Prediction pipeline will populate results.",
        }
    )


def _split_text(text: str) -> list[str]:
    if not text:
        return []
    seps = [",", ";", "/", "\n"]
    for s in seps:
        text = text.replace(s, ",")
    return [t.strip().lower().replace(" ", "_") for t in text.split(",") if t.strip()]
