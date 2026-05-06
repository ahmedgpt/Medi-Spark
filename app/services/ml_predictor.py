"""ML predictor — Week 1 provides only a thin loader; Week 2 wires it into Kafka."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import joblib

from config.settings import Config

log = logging.getLogger(__name__)

_model = None
_symptom_index: dict[str, int] | None = None
_classes: list[str] | None = None


def _ensure_loaded() -> None:
    global _model, _symptom_index, _classes
    if _model is not None:
        return
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {Config.MODEL_PATH}. "
            "Run: python -m app.spark.model_trainer"
        )
    bundle = joblib.load(Config.MODEL_PATH)
    _model = bundle["model"]
    _symptom_index = bundle["symptom_index"]
    _classes = list(bundle["classes"])
    log.info("Model loaded: %d symptoms, %d classes", len(_symptom_index), len(_classes))


def predict_disease(symptoms: list[str], top_k: int = 3) -> dict[str, Any]:
    """Return top-k diseases with confidence scores."""
    _ensure_loaded()
    assert _model is not None and _symptom_index is not None and _classes is not None

    vector = [0] * len(_symptom_index)
    matched = 0
    for s in symptoms:
        idx = _symptom_index.get(s.strip().lower().replace(" ", "_"))
        if idx is not None:
            vector[idx] = 1
            matched += 1

    proba = _model.predict_proba([vector])[0]
    ranked = sorted(zip(_classes, proba), key=lambda x: x[1], reverse=True)[:top_k]
    return {
        "matched_symptoms": matched,
        "predictions": [
            {"disease": d, "confidence": round(float(p), 4)} for d, p in ranked
        ],
    }
