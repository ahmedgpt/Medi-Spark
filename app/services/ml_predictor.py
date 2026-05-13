"""Runtime ML predictor — loads trained model, returns top-3 disease predictions."""

import os
import joblib
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

_classifier    = None
_label_encoder = None
_feature_names = None   # list of 132 symptom column names


def _load():
    global _classifier, _label_encoder, _feature_names
    if _classifier is None:
        _classifier    = joblib.load(os.path.join(MODEL_DIR, "disease_classifier.pkl"))
        _label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        _feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))


def predict_diseases(symptoms: list, top_n: int = 3) -> list:
    """
    symptoms : list of raw strings e.g. ['itching', 'skin rash', 'nodal skin eruptions']
    Returns  : [{'disease': str, 'confidence': float}, ...]
    """
    try:
        _load()
    except Exception as e:
        raise RuntimeError(f"Failed to load ML model: {str(e)}") from e
    
    if _classifier is None or _label_encoder is None or _feature_names is None:
        raise RuntimeError("ML model failed to load. Ensure model files exist in models/ directory.")

    # Normalise input symptoms to match column-name format
    normalised = {s.strip().lower().replace(" ", "_") for s in symptoms}

    # Build binary feature vector (1 if symptom present, else 0)
    vec = np.array(
        [1 if feat in normalised else 0 for feat in _feature_names],
        dtype=int
    ).reshape(1, -1)

    proba       = _classifier.predict_proba(vec)[0]
    top_indices = np.argsort(proba)[::-1][:top_n]

    return [
        {
            "disease":    _label_encoder.classes_[i],
            "confidence": round(float(proba[i]), 4),
        }
        for i in top_indices
        if proba[i] > 0.0
    ]