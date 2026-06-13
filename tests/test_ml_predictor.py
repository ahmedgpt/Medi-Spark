"""Tests for ML predictor service."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestPredictDiseases:
    """Test app.services.ml_predictor.predict_diseases()."""

    def test_returns_list(self):
        from app.services.ml_predictor import predict_diseases
        result = predict_diseases(["cough", "fever"], top_n=3)
        assert isinstance(result, list)

    def test_top_n_respected(self):
        from app.services.ml_predictor import predict_diseases
        result = predict_diseases(["headache", "fatigue"], top_n=2)
        assert len(result) <= 2

    def test_each_prediction_has_keys(self):
        from app.services.ml_predictor import predict_diseases
        result = predict_diseases(["cough", "fever", "headache"], top_n=3)
        for pred in result:
            assert "disease" in pred
            assert "confidence" in pred
            assert isinstance(pred["confidence"], (int, float))

    def test_confidence_in_range(self):
        from app.services.ml_predictor import predict_diseases
        result = predict_diseases(["cough"], top_n=3)
        for pred in result:
            assert 0 <= pred["confidence"] <= 1

    def test_empty_symptoms(self):
        from app.services.ml_predictor import predict_diseases
        with pytest.raises(Exception):
            predict_diseases([], top_n=3)
