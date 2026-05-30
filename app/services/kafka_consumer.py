"""Kafka consumer worker.

Run as a standalone process:

    python -m app.services.kafka_consumer

Week 3 behaviour:
  * Consumes `symptom-input`  → runs full ML + RAG + risk pipeline
  * Consumes `retrain-trigger` → invokes continuous_learner
  * Mirrors events to `audit-log` for PySpark batch processor
"""
from __future__ import annotations

import json
import logging
import signal
import sys
from typing import Any

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from .kafka_producer import publish

try:
    from config.settings import Config
except ImportError:  # pragma: no cover
    from ...config.settings import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] consumer: %(message)s",
)
log = logging.getLogger(__name__)

_running = True


def _shutdown(_signum, _frame):
    global _running
    log.info("Shutting down consumer…")
    _running = False


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def handle_symptom_event(event: dict[str, Any]) -> None:
    log.info("symptom event received: log_id=%s symptoms=%s", event.get("log_id"), event.get("symptoms"))

    # Mirror to audit log (PySpark batch_processor reads from MongoDB)
    publish(Config.TOPIC_AUDIT_LOG, {"type": "symptom-input", "payload": event}, key=event.get("user_id"))

    # Week 3: run full ML + RAG pipeline
    try:
        from app.services.ml_predictor import predict_diseases
        from app.services.risk_assessor import assess_risk
        symptoms = event.get("symptoms", [])
        duration = int(event.get("duration_days", 1))
        age      = int(event.get("age", 30))
        predictions = predict_diseases(symptoms, top_n=3)
        risk        = assess_risk(symptoms, duration, age)
        result = {
            "user_id":     event.get("user_id"),
            "log_id":      event.get("log_id"),
            "predictions": predictions,
            "risk_level":  risk["risk_level"],
            "severity":    risk["severity_score"],
        }
        publish(Config.TOPIC_PREDICTION_RESULT, result, key=event.get("user_id"))
        log.info("ML pipeline complete: top=%s risk=%s",
                 predictions[0]["disease"] if predictions else "?",
                 risk["risk_level"])
    except Exception:  # noqa: BLE001
        log.exception("ML pipeline failed for event log_id=%s", event.get("log_id"))


def handle_retrain_trigger(event: dict[str, Any]) -> None:
    """Triggered by Kafka `retrain-trigger` topic — runs continuous learning pipeline."""
    force = event.get("force", False)
    log.info("retrain-trigger received (force=%s) — starting continuous learner …", force)
    try:
        from app.spark.continuous_learner import run_continuous_learning
        result = run_continuous_learning(force_deploy=force)
        log.info("Continuous learning result: %s", result)
    except Exception:  # noqa: BLE001
        log.exception("Continuous learning pipeline failed.")


def main() -> int:
    # Subscribe to all relevant topics
    topics = [
        Config.TOPIC_SYMPTOM_INPUT,
        Config.TOPIC_RETRAIN_TRIGGER,
    ]
    try:
        consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
            group_id="medispark-pipeline",
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            key_deserializer=lambda b: b.decode("utf-8") if b else None,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )
    except NoBrokersAvailable:
        log.error("Kafka broker unavailable at %s", Config.KAFKA_BOOTSTRAP_SERVERS)
        return 1

    log.info("Consumer subscribed to: %s", topics)
    while _running:
        records = consumer.poll(timeout_ms=1000)
        for tp, msgs in records.items():
            for msg in msgs:
                try:
                    if tp.topic == Config.TOPIC_SYMPTOM_INPUT:
                        handle_symptom_event(msg.value)
                    elif tp.topic == Config.TOPIC_RETRAIN_TRIGGER:
                        handle_retrain_trigger(msg.value)
                    else:
                        log.debug("Unhandled topic %s offset=%s", tp.topic, msg.offset)
                except Exception:  # noqa: BLE001
                    log.exception("Failed to process message topic=%s offset=%s",
                                  tp.topic, msg.offset)

    consumer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
