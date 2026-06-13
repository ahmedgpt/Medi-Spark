"""
Runtime ML predictor — hybrid approach combining:
1. Symptom-overlap scoring (handles partial symptom input)
2. XGBoost ML model probabilities (for tiebreaking)

The Kaggle dataset trains each disease with a FIXED set of symptoms, so the
XGBoost model memorised exact patterns and fails on partial input.  The overlap
scorer fills that gap by ranking diseases by what fraction of the user's
symptoms match each disease's known symptom set.
"""

import os
import json
import joblib
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

_classifier       = None
_label_encoder    = None
_feature_names    = None   # list of 132 symptom column names
_disease_symptoms = None   # dict: disease -> list of symptom feature names


def _load():
    global _classifier, _label_encoder, _feature_names, _disease_symptoms
    if _classifier is None:
        _classifier    = joblib.load(os.path.join(MODEL_DIR, "disease_classifier.pkl"))
        _label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        _feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

        # Load disease-symptom map for overlap scoring
        ds_path = os.path.join(MODEL_DIR, "disease_symptoms.json")
        if os.path.exists(ds_path):
            with open(ds_path, "r") as f:
                _disease_symptoms = json.load(f)
        else:
            _disease_symptoms = {}
            print("[ML] WARNING: disease_symptoms.json not found - overlap scoring disabled")


def _overlap_scores(matched_features: set, top_n: int = 3) -> list:
    """
    Score each disease by how well the user's symptoms overlap with
    that disease's known symptom set.
    """
    if not _disease_symptoms or not matched_features:
        return []

    scores = []
    for disease, disease_syms in _disease_symptoms.items():
        disease_set = set(disease_syms)
        if not disease_set:
            continue

        overlap = matched_features & disease_set
        if not overlap:
            continue

        precision = len(overlap) / len(matched_features)
        recall = len(overlap) / len(disease_set)

        if precision + recall > 0:
            score = 2 * (precision * recall) / (precision + recall)
        else:
            score = 0.0

        scores.append({
            "disease": disease,
            "confidence": round(score, 4),
            "overlap": len(overlap),
            "total_disease_symptoms": len(disease_set),
        })

    # Sort by score descending, then by overlap count
    scores.sort(key=lambda x: (x["confidence"], x["overlap"]), reverse=True)
    return scores[:top_n]


def _ml_scores(matched_features: set, top_n: int = 3) -> list:
    """Original ML model prediction using feature vector."""
    vec = np.array(
        [1 if feat in matched_features else 0 for feat in _feature_names],
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

    # ── Map user symptoms to model feature names via synonym lookup ────────
    from app.services.symptom_mapper import map_symptoms
    matched_features = map_symptoms(symptoms, _feature_names)

    if not matched_features:
        print(f"[ML] WARNING: No symptoms matched any model features! Input: {symptoms}")
        return []

    active_count = len(matched_features)
    print(f"[ML] Feature vector: {active_count}/{len(_feature_names)} features active")

    # ── Get scores from both approaches ───────────────────────────────────
    ml_results      = _ml_scores(matched_features, top_n=10)
    overlap_results = _overlap_scores(matched_features, top_n=10)

    # ── Hybrid merge ──────────────────────────────────────────────────────
    # Combine overlap score (0-1) with ML probability (0-1).
    # Overlap scoring is weighted higher because it handles partial input
    # far better than the memorisation-prone ML model.
    OVERLAP_WEIGHT = 0.7
    ML_WEIGHT      = 0.3

    combined = {}

    for r in overlap_results:
        d = r["disease"]
        combined[d] = combined.get(d, 0.0) + r["confidence"] * OVERLAP_WEIGHT

    for r in ml_results:
        d = r["disease"]
        combined[d] = combined.get(d, 0.0) + r["confidence"] * ML_WEIGHT

    # Sort and return top N
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = [
        {"disease": d, "confidence": round(score, 4)}
        for d, score in ranked
        if score > 0.0
    ]

    if results:
        print(f"[ML] Top prediction: {results[0]['disease']} ({results[0]['confidence']*100:.1f}%)")

    return results