"""Integration test for POST /api/predict endpoint."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def logged_in_client(app):
    """Return a Flask test client with a logged-in session."""
    client = app.test_client()

    # Register + login a test user
    with app.app_context():
        try:
            from app.extensions import mongo
            db = mongo.db
            if db is not None:
                db.users.delete_one({"email": "test@medispark.test"})
        except Exception:
            pass

    client.post("/auth/register", data={
        "name": "Test User",
        "email": "test@medispark.test",
        "password": "TestPass123!",
        "confirm_password": "TestPass123!",
        "age": "30",
    }, follow_redirects=True)

    client.post("/auth/login", data={
        "email": "test@medispark.test",
        "password": "TestPass123!",
    }, follow_redirects=True)

    return client


class TestPredictRoute:
    """Integration tests for the /api/predict endpoint."""

    def test_predict_returns_200(self, logged_in_client):
        resp = logged_in_client.post("/api/predict", json={
            "symptoms": ["cough", "fever"],
            "duration_days": 3,
            "age": 30,
        })
        assert resp.status_code == 200

    def test_predict_response_structure(self, logged_in_client):
        resp = logged_in_client.post("/api/predict", json={
            "symptoms": ["headache", "fatigue"],
            "duration_days": 1,
            "age": 25,
        })
        data = resp.get_json()
        assert "predictions" in data
        assert "risk" in data
        assert "medicines" in data

    def test_predict_empty_symptoms(self, logged_in_client):
        resp = logged_in_client.post("/api/predict", json={
            "symptoms": [],
            "duration_days": 1,
            "age": 30,
        })
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_predict_no_json(self, logged_in_client):
        resp = logged_in_client.post("/api/predict")
        assert resp.status_code == 400
