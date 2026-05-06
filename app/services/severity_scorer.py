"""Week 2 — Severity scorer (0–100). Placeholder weights ready to use."""
from __future__ import annotations

SYMPTOM_WEIGHTS: dict[str, int] = {
    "chest_pain": 30,
    "difficulty_breathing": 28,
    "vomiting_blood": 35,
    "high_fever": 20,
    "severe_headache": 15,
    "fatigue": 8,
    "cough": 6,
}


def calculate_severity(symptoms: list[str], duration_days: int, age: int | None) -> int:
    base = sum(SYMPTOM_WEIGHTS.get(s, 5) for s in symptoms)
    duration_factor = min(1 + (duration_days * 0.1), 2.0)
    age_factor = 1.3 if (age is not None and (age > 60 or age < 5)) else 1.0
    return min(int(base * duration_factor * age_factor), 100)
