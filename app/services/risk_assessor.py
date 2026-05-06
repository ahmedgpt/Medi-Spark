"""Week 2 — Risk assessment from severity score."""
from __future__ import annotations


def assess(severity: int) -> dict:
    if severity >= 70:
        return {
            "level": "HIGH",
            "tests": ["CBC", "LFT", "Chest X-ray"],
            "advice": "Consult a specialist immediately. Prescription-level care recommended.",
        }
    if severity >= 40:
        return {
            "level": "MEDIUM",
            "tests": ["Basic blood test"],
            "advice": "OTC medication with monitoring. See a GP if symptoms persist.",
        }
    return {
        "level": "LOW",
        "tests": [],
        "advice": "Self-care and OTC remedies should suffice. Monitor for changes.",
    }
