"""Smoke test for severity calculation."""
from app.services.severity_scorer import calculate_severity


def test_low_severity():
    assert calculate_severity(["cough"], duration_days=1, age=30) < 40


def test_high_severity():
    score = calculate_severity(
        ["chest_pain", "difficulty_breathing", "vomiting_blood"],
        duration_days=5,
        age=70,
    )
    assert score >= 70


def test_capped_at_100():
    score = calculate_severity(["chest_pain"] * 20, duration_days=30, age=80)
    assert score == 100
