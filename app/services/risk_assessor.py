"""
Day 10: Risk Assessor
Classifies cases as HIGH / MEDIUM / LOW with recommendations.
"""

from app.services.severity_scorer import calculate_severity


def assess_risk(symptoms: list, duration_days: int = 1, age: int = 30) -> dict:
    """
    Assess risk level based on severity score.
    Returns full risk assessment dict.
    """
    score = calculate_severity(symptoms, duration_days, age)

    if score >= 70:
        level = "HIGH"
        tests = [
            "Complete Blood Count (CBC)",
            "Liver Function Test (LFT)",
            "Chest X-Ray",
            "ECG",
            "Blood Culture"
        ]
        advice = "Seek immediate medical attention. Visit a specialist or emergency room."
        medicine_type = "prescription"

    elif score >= 40:
        level = "MEDIUM"
        tests = [
            "Basic Blood Test",
            "Urine Analysis"
        ]
        advice = "Consult a doctor within 24-48 hours. Monitor symptoms closely."
        medicine_type = "otc"

    else:
        level = "LOW"
        tests = []
        advice = "Rest and stay hydrated. Use OTC medicines if needed."
        medicine_type = "otc"

    return {
        "severity_score": score,
        "risk_level":     level,
        "recommended_tests": tests,
        "advice":         advice,
        "medicine_type":  medicine_type
    }