"""Centralised configuration loaded from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))

    # Mongo
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/medispark")

    # Redis / sessions
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    SESSION_TYPE = "redis"
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_KEY_PREFIX = "medispark:"

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_CLIENT_ID = os.getenv("KAFKA_CLIENT_ID", "medispark-app")

    TOPIC_SYMPTOM_INPUT = os.getenv("TOPIC_SYMPTOM_INPUT", "symptom-input")
    TOPIC_PREDICTION_RESULT = os.getenv("TOPIC_PREDICTION_RESULT", "prediction-result")
    TOPIC_CHAT_MESSAGES = os.getenv("TOPIC_CHAT_MESSAGES", "chat-messages")
    TOPIC_AUDIT_LOG = os.getenv("TOPIC_AUDIT_LOG", "audit-log")
    TOPIC_ALERT_HIGH_RISK = os.getenv("TOPIC_ALERT_HIGH_RISK", "alert-high-risk")

    ALL_TOPICS = [
        TOPIC_SYMPTOM_INPUT,
        TOPIC_PREDICTION_RESULT,
        TOPIC_CHAT_MESSAGES,
        TOPIC_AUDIT_LOG,
        TOPIC_ALERT_HIGH_RISK,
    ]

    # ML
    MODEL_PATH = str(BASE_DIR / os.getenv("MODEL_PATH", "models/disease_classifier.pkl"))
    SYMPTOM_INDEX_PATH = str(
        BASE_DIR / os.getenv("SYMPTOM_INDEX_PATH", "models/symptom_index.json")
    )

    # LLM keys (used Week 2)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

    # Week 3 — Kafka topic for retraining trigger
    TOPIC_RETRAIN_TRIGGER = os.getenv("TOPIC_RETRAIN_TRIGGER", "retrain-trigger")

    # Week 3 — PySpark batch aggregation interval (hours)
    BATCH_INTERVAL_HOURS = int(os.getenv("BATCH_INTERVAL_HOURS", "6"))

    # Week 3 — Minimum accuracy improvement (%) to auto-deploy new model
    MODEL_AB_THRESHOLD = float(os.getenv("MODEL_AB_THRESHOLD", "2.0"))

    # Week 3 — Minimum new samples needed before triggering retraining
    MIN_RETRAIN_SAMPLES = int(os.getenv("MIN_RETRAIN_SAMPLES", "50"))
