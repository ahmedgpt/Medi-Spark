"""Kafka producer singleton — JSON values, UTF-8 keys.
Degrades gracefully when kafka-python is not installed or no brokers are available.
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)
_producer = None


def get_producer():
    global _producer
    if _producer is not None:
        return _producer
    try:
        from kafka import KafkaProducer
        from kafka.errors import NoBrokersAvailable

        try:
            from config.settings import Config
        except ImportError:  # pragma: no cover
            from ...config.settings import Config

        _producer = KafkaProducer(
            bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
            client_id=Config.KAFKA_CLIENT_ID,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: (k or "").encode("utf-8"),
            acks="all",
            retries=3,
            linger_ms=20,
        )
        log.info("Kafka producer connected to %s", Config.KAFKA_BOOTSTRAP_SERVERS)
        return _producer
    except ImportError:
        log.warning("kafka-python not installed — Kafka publishing disabled.")
        return None
    except Exception as exc:
        log.warning("Kafka producer init failed: %s — publishing disabled.", exc)
        return None


def publish(topic: str, value: dict[str, Any], key: str | None = None) -> None:
    """Convenience wrapper used by services. No-op if Kafka is unavailable."""
    producer = get_producer()
    if producer is not None:
        try:
            producer.send(topic, value=value, key=key)
        except Exception as exc:
            log.warning("Kafka publish to '%s' failed: %s", topic, exc)