"""Kafka consumer worker.

Run as a standalone process:

    python -m app.services.kafka_consumer

Week 1 behaviour:
  * Consumes `symptom-input`
  * Logs the event
  * Mirrors it to `audit-log` (PySpark will read this in Week 3)

Week 2 will replace this with the full ML + RAG + risk pipeline.
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

    # Mirror to audit log
    publish(Config.TOPIC_AUDIT_LOG, {"type": "symptom-input", "payload": event}, key=event.get("user_id"))

    # Week 2 hook — placeholder so the wiring is visible
    # from app.services.ml_predictor import predict_disease
    # result = predict_disease(event["symptoms"], event["age"], event["duration_days"])
    # publish(Config.TOPIC_PREDICTION_RESULT, result, key=event.get("user_id"))


def main() -> int:
    try:
        consumer = KafkaConsumer(
            Config.TOPIC_SYMPTOM_INPUT,
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

    log.info("Consumer subscribed to %s", Config.TOPIC_SYMPTOM_INPUT)
    while _running:
        records = consumer.poll(timeout_ms=1000)
        for _tp, msgs in records.items():
            for msg in msgs:
                try:
                    handle_symptom_event(msg.value)
                except Exception:  # noqa: BLE001
                    log.exception("Failed to process message offset=%s", msg.offset)

    consumer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
