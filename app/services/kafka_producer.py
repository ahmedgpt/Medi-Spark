"""Kafka producer singleton — JSON values, UTF-8 keys."""
from __future__ import annotations

import json
import logging
from typing import Any

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

try:
    from config.settings import Config
except ImportError:  # pragma: no cover
    from ...config.settings import Config

log = logging.getLogger(__name__)
_producer: KafkaProducer | None = None


def get_producer() -> KafkaProducer:
    global _producer
    if _producer is not None:
        return _producer
    try:
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
    except NoBrokersAvailable as exc:
        log.error("No Kafka brokers available at %s", Config.KAFKA_BOOTSTRAP_SERVERS)
        raise exc
    return _producer


def publish(topic: str, value: dict[str, Any], key: str | None = None) -> None:
    """Convenience wrapper used by services."""
    get_producer().send(topic, value=value, key=key)