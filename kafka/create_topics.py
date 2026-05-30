"""Create all Kafka topics defined in `kafka/topics_config.yaml`.

Usage:
    python kafka/create_topics.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

# Make project root importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("create_topics")


TOPIC_SPECS = [
    (Config.TOPIC_SYMPTOM_INPUT,      3, 1),
    (Config.TOPIC_PREDICTION_RESULT,  3, 1),
    (Config.TOPIC_CHAT_MESSAGES,      3, 1),
    (Config.TOPIC_AUDIT_LOG,          3, 1),
    (Config.TOPIC_ALERT_HIGH_RISK,    1, 1),
    (Config.TOPIC_RETRAIN_TRIGGER,    1, 1),   # Week 3 — continuous learning
]


def main() -> int:
    admin = KafkaAdminClient(
        bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
        client_id="medispark-admin",
    )
    new_topics = [
        NewTopic(name=name, num_partitions=p, replication_factor=r)
        for name, p, r in TOPIC_SPECS
    ]
    for t in new_topics:
        try:
            admin.create_topics([t])
            log.info("Created topic: %s", t.name)
        except TopicAlreadyExistsError:
            log.info("Topic already exists: %s", t.name)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to create %s: %s", t.name, exc)
    admin.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
