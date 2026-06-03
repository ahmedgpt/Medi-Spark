"""Shared pytest fixtures for MediSpark tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017/medispark_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


@pytest.fixture
def app():
    """Create a Flask test app with testing config."""
    from app import create_app
    from config.settings import Config

    class TestConfig(Config):
        TESTING = True
        SECRET_KEY = "test-secret"
        WTF_CSRF_ENABLED = False

    application = create_app(TestConfig)
    return application


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def mock_mongo(monkeypatch):
    """Mock MongoDB for tests that don't need a real DB."""
    mock_db = MagicMock()
    mock_db.symptom_logs = MagicMock()
    mock_db.users = MagicMock()
    return mock_db
