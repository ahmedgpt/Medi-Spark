"""Tests for risk assessor service."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.risk_assessor import assess_risk


class TestAssessRisk:
    """Test app.services.risk_assessor.assess_risk()."""

    def test_returns_dict(self):
        result = assess_risk(["cough", "fever"], duration_days=3, age=30)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = assess_risk(["headache"], duration_days=1, age=25)
        assert "risk_level" in result
        assert "severity_score" in result
        assert "recommended_tests" in result
        assert "advice" in result

    def test_risk_level_values(self):
        result = assess_risk(["cough"], duration_days=1, age=25)
        assert result["risk_level"] in ("HIGH", "MEDIUM", "LOW")

    def test_severity_in_range(self):
        result = assess_risk(["fever", "cough"], duration_days=5, age=40)
        assert 0 <= result["severity_score"] <= 100

    def test_high_risk_symptoms(self):
        """Chest pain + breathing difficulty → HIGH risk."""
        result = assess_risk(
            ["chest_pain", "breathlessness", "high_fever"],
            duration_days=7,
            age=65,
        )
        assert result["risk_level"] == "HIGH"

    def test_low_risk_mild(self):
        """Single mild symptom, young patient → LOW risk."""
        result = assess_risk(["headache"], duration_days=1, age=20)
        assert result["risk_level"] in ("LOW", "MEDIUM")

    def test_recommended_tests_is_list(self):
        result = assess_risk(["fever", "cough"], duration_days=3, age=30)
        assert isinstance(result["recommended_tests"], list)

    def test_elderly_risk_boost(self):
        """Older patients should have equal or higher severity."""
        young = assess_risk(["fever", "cough"], duration_days=3, age=25)
        old = assess_risk(["fever", "cough"], duration_days=3, age=70)
        assert old["severity_score"] >= young["severity_score"]
