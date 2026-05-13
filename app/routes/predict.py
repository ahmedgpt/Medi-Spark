"""
Day 10-11: Predict route — wires ML + RAG + Severity + Risk + Medicine together.
Endpoint: POST /api/predict
"""

from flask import Blueprint, request, jsonify

from app.services.ml_predictor       import predict_diseases
from app.services.rag_engine         import retrieve
from app.services.severity_scorer    import calculate_severity
from app.services.risk_assessor      import assess_risk
from app.services.medicine_suggester import suggest_medicines
from app.services import kafka_producer

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    Full prediction pipeline:
    1. Receive symptoms + duration + age
    2. ML prediction (top 3 diseases)
    3. Severity score + risk assessment
    4. RAG retrieval (medical knowledge)
    5. Medicine suggestions
    6. Kafka event publish
    """
    data = request.get_json(silent=True) or {}

    # ── Input extraction ───────────────────────────────────────────────────────
    symptoms = data.get("symptoms", [])
    try:
        duration = int(data.get("duration_days", 1))
        age      = int(data.get("age", 30))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid duration_days or age - must be integers"}), 400

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    # ── Step 1: ML Prediction ─────────────────────────────────────────────────
    try:
        predictions = predict_diseases(symptoms, top_n=3)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    top_disease = predictions[0]["disease"] if predictions else "Unknown"

    # ── Step 2: Severity + Risk ───────────────────────────────────────────────
    try:
        risk = assess_risk(symptoms, duration, age)
    except Exception as e:
        return jsonify({"error": f"Risk assessment failed: {str(e)}"}), 500

    # ── Step 3: RAG Retrieval ─────────────────────────────────────────────────
    try:
        rag_query   = f"What are the symptoms and treatment of {top_disease}?"
        rag_results = retrieve(rag_query, k=3)
    except Exception as e:
        rag_results = []
        print(f"[WARN] RAG retrieval failed: {e}")

    # ── Step 4: Medicine Suggestions ─────────────────────────────────────────
    try:
        medicines = suggest_medicines(top_disease, risk.get("medicine_type", "otc"))
    except Exception as e:
        medicines = {"error": str(e)}

    # ── Step 5: Publish to Kafka ──────────────────────────────────────────────
    try:
        kafka_producer.publish("symptom-input", {
            "symptoms":    symptoms,
            "duration":    duration,
            "age":         age,
            "top_disease": top_disease,
            "risk_level":  risk["risk_level"],
            "severity":    risk["severity_score"]
        })
    except Exception:
        pass  # Don't fail the request if Kafka is down

    # ── Build response ────────────────────────────────────────────────────────
    response = {
        "predictions": predictions,
        "risk": {
            "severity_score":    risk.get("severity_score", 0),
            "risk_level":        risk.get("risk_level", "UNKNOWN"),
            "recommended_tests": risk.get("recommended_tests", []),
            "advice":            risk.get("advice", "")
        },
        "medicines":     medicines,
        "rag_knowledge": [
            {
                "disease": r.get("disease", "Unknown"),
                "content": r.get("content", "")[:300]
            }
            for r in rag_results
        ]
    }

    return jsonify(response), 200