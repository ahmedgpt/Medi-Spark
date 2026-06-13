"""
MediSpark — Locust Load Test (Day 26)
======================================
Simulates 100 concurrent users hitting the three core API endpoints.

Usage (from project root):
    # Install locust (already in requirements.txt)
    pip install locust

    # Run against local dev server
    locust -f locustfile.py --host=http://localhost:5000 --users=100 --spawn-rate=10

    # Headless run (CI mode) — 100 users, 30 second ramp-up, 2 minute test
    locust -f locustfile.py --host=http://localhost:5000 \
           --users=100 --spawn-rate=10 --run-time=2m --headless \
           --html=load_test_report.html

Targets:
    POST /auth/login               — authenticate (warm up session)
    POST /api/predict              — disease prediction (heavy ML endpoint)
    POST /api/chat                 — chatbot message
    GET  /api/chat/history         — history fetch
    GET  /api/dashboard/stats      — dashboard stats
"""
import json
import random

from locust import HttpUser, between, task


# ── Sample payloads ───────────────────────────────────────────────────────────

SYMPTOM_SETS = [
    ["fever", "headache", "body aches", "fatigue"],
    ["cough", "breathlessness", "chest pain"],
    ["skin rash", "itching", "fever"],
    ["abdominal pain", "nausea", "jaundice"],
    ["excessive thirst", "frequent urination", "fatigue"],
    ["joint pain", "swelling", "stiffness"],
    ["high fever", "chills", "sweating", "headache"],
    ["runny nose", "sneezing", "sore throat"],
    ["shortness of breath", "wheezing", "cough"],
    ["dizziness", "loss of balance", "nausea"],
]

CHAT_MESSAGES = [
    "I have fever and headache, what should I do?",
    "What medicine for typhoid?",
    "Is malaria serious?",
    "I have been coughing for two weeks",
    "What are symptoms of diabetes?",
    "How to treat skin rash at home?",
    "mujhe bukhar hai aur sar dard hai",   # Roman Urdu
    "What is the risk level for asthma?",
]

TEST_USER_EMAIL    = "loadtest@medispark.test"
TEST_USER_PASSWORD = "LoadTest123!"


class MediSparkUser(HttpUser):
    """
    Simulates a typical MediSpark user:
      - Logs in once on start
      - Repeatedly submits symptoms, chats, and checks history/dashboard
    """

    wait_time = between(1, 3)   # 1–3 second think time between requests

    def on_start(self):
        """Called once per simulated user — register then log in."""
        # Try to register (will fail silently if user already exists)
        self.client.post(
            "/auth/register",
            data={
                "name":             "Load Tester",
                "email":            TEST_USER_EMAIL,
                "password":         TEST_USER_PASSWORD,
                "confirm_password": TEST_USER_PASSWORD,
                "age":              "30",
            },
        )
        # Log in
        resp = self.client.post(
            "/auth/login",
            data={
                "email":    TEST_USER_EMAIL,
                "password": TEST_USER_PASSWORD,
            },
            allow_redirects=True,
        )
        if resp.status_code not in (200, 302):
            self.environment.runner.quit()

    # ── Task weights (higher = called more often) ─────────────────────────────

    @task(5)
    def predict_symptoms(self):
        """POST /api/predict — core ML endpoint (most important to load test)."""
        symptoms = random.choice(SYMPTOM_SETS)
        self.client.post(
            "/api/predict",
            json={
                "symptoms":     symptoms,
                "duration_days": random.randint(1, 14),
                "age":          random.randint(18, 70),
            },
            name="/api/predict",
        )

    @task(3)
    def chat_message(self):
        """POST /api/chat — chatbot endpoint."""
        message = random.choice(CHAT_MESSAGES)
        self.client.post(
            "/api/chat",
            json={"message": message},
            name="/api/chat",
        )

    @task(2)
    def dashboard_stats(self):
        """GET /api/dashboard/stats — dashboard data."""
        self.client.get("/api/dashboard/stats", name="/api/dashboard/stats")

    @task(1)
    def chat_history(self):
        """GET /api/chat/history — conversation history."""
        self.client.get("/api/chat/history", name="/api/chat/history")

    @task(1)
    def history_page(self):
        """GET /api/history — symptom log history."""
        self.client.get("/api/history", name="/api/history")


class AdminUser(HttpUser):
    """
    Simulates a lighter admin-level browsing pattern (10% of users).
    Just hits the dashboard and history — no predictions or chat.
    """

    wait_time = between(2, 5)
    weight    = 10   # 10% of total simulated users

    def on_start(self):
        self.client.post(
            "/auth/login",
            data={"email": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD},
            allow_redirects=True,
        )

    @task(3)
    def dashboard(self):
        self.client.get("/api/dashboard/stats", name="/api/dashboard/stats [admin]")

    @task(1)
    def history(self):
        self.client.get("/api/history", name="/api/history [admin]")
