"""
Predict route — wires ML + RAG + Severity + Risk + Medicine + LLM synthesis.
Endpoint: POST /api/predict
"""
from __future__ import annotations

import logging
import os
import threading

from flask import Blueprint, current_app, request, jsonify
from flask_login import current_user, login_required
from app.utils.auth_decorators import jwt_or_login_required

from app.services.ml_predictor       import predict_diseases
from app.services.severity_scorer    import calculate_severity
from app.services.risk_assessor      import assess_risk
from app.services.medicine_suggester import suggest_medicines
from app.services import kafka_producer

log = logging.getLogger(__name__)

predict_bp = Blueprint("predict", __name__)

_OLLAMA_URL     = os.getenv("OLLAMA_URL",     "http://localhost:11434")
_OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "gemma3:1b")
_LLM_TIMEOUT    = int(os.getenv("OLLAMA_TIMEOUT", "30"))


def _generate_llm_summary(
    top_disease: str,
    predictions: list,
    risk: dict,
    rag_snippets: list[str],
    medicines: dict,
) -> str:
    """
    Call Ollama (or cloud fallback) to synthesise ML output + RAG + risk into
    a human-readable, empathetic medical advisory paragraph.
    Returns empty string on any failure (non-fatal).
    """
    pred_lines = "\n".join(
        f"  {i+1}. {p['disease']} ({p['confidence']*100:.1f}%)"
        for i, p in enumerate(predictions[:3])
    )
    rag_context = "\n".join(s[:200] for s in rag_snippets[:2]) if rag_snippets else "No additional context."
    otc  = ", ".join(medicines.get("otc", [])[:3]) or "see a pharmacist"
    rx   = ", ".join(medicines.get("prescription", [])[:2]) or "consult a doctor"

    prompt = (
        f"You are MediSpark, an empathetic AI medical assistant.\n\n"
        f"A patient's symptom analysis produced these results:\n"
        f"Top predicted conditions:\n{pred_lines}\n\n"
        f"Risk level: {risk.get('risk_level', 'UNKNOWN')} "
        f"(severity score {risk.get('severity_score', 0)}/100)\n"
        f"Doctor's advice: {risk.get('advice', '')}\n\n"
        f"Relevant medical knowledge:\n{rag_context}\n\n"
        f"OTC medicines: {otc}\n"
        f"Prescription medicines: {rx}\n\n"
        f"Write a warm, clear 3-4 sentence summary for the patient explaining:\n"
        f"1. What condition is most likely and why.\n"
        f"2. The risk level and what they should do next.\n"
        f"3. End with a disclaimer to consult a qualified doctor.\n"
        f"Keep it under 120 words. Plain language, no bullet points."
    )

    # Try Ollama first
    try:
        import requests as req
        resp = req.post(
            f"{_OLLAMA_URL}/api/chat",
            json={
                "model":   _OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":  False,
                "options": {"temperature": 0.3, "num_predict": 180},
            },
            timeout=_LLM_TIMEOUT,
        )
        resp.raise_for_status()
        text = resp.json().get("message", {}).get("content", "").strip()
        if text:
            return text
    except Exception as exc:
        log.warning("[Predict] Ollama LLM summary failed: %s", exc)

    # Cloud fallback (Anthropic)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as exc:
            log.warning("[Predict] Anthropic LLM summary failed: %s", exc)

    return ""


@predict_bp.route("/predict", methods=["POST"])
@jwt_or_login_required
def predict():
    """
    Full prediction pipeline:
    1. Receive + sanitize symptoms + duration + age
    2. ML prediction (top 3 diseases)
    3. Severity score + risk assessment
    4. RAG retrieval (medical knowledge)
    5. Medicine suggestions
    6. Persist to MongoDB (SymptomLog)  ← Week 4 fix
    7. Kafka event publish
    """
    data = request.get_json(silent=True) or {}

    # ── Input extraction ───────────────────────────────────────────────────────
    symptoms = data.get("symptoms", [])
    if isinstance(symptoms, str):
        # Support comma-separated string as well as list
        symptoms = [s.strip() for s in symptoms.split(",") if s.strip()]

    try:
        duration = int(data.get("duration_days", 1))
        age      = int(data.get("age", 30))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid duration_days or age - must be integers"}), 400

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    # Input validation — max 20 symptoms, max 200 chars each
    if len(symptoms) > 20:
        return jsonify({"error": "Maximum 20 symptoms allowed."}), 400
    if any(len(str(s)) > 200 for s in symptoms):
        return jsonify({"error": "Each symptom must be under 200 characters."}), 400

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
        from app.services.rag_engine import retrieve
        rag_query   = f"What are the symptoms and treatment of {top_disease}?"
        rag_results = retrieve(rag_query, k=3)
    except Exception as e:
        rag_results = []
        log.warning("[Predict] RAG retrieval failed: %s", e)

    # ── Step 4: Medicine Suggestions ─────────────────────────────────────────
    try:
        medicines = suggest_medicines(top_disease, risk.get("medicine_type", "otc"))
    except Exception as e:
        medicines = {"error": str(e)}

    # ── Step 4b: LLM Advisory Summary ────────────────────────────────────────
    rag_snippets = [r.get("content", "") for r in rag_results]
    try:
        llm_summary = _generate_llm_summary(top_disease, predictions, risk, rag_snippets, medicines)
    except Exception as e:
        llm_summary = ""
        log.warning("[Predict] LLM summary skipped: %s", e)

    # ── Step 5: Persist to MongoDB ────────────────────────────────────────────
    log_id = None
    try:
        from app.models.symptom_log import SymptomLog
        log_id = SymptomLog.create_full(
            user_id       = str(current_user.id),
            symptoms      = symptoms,
            duration_days = duration,
            age           = age,
            language      = data.get("language", "english"),
            raw_text      = ", ".join(symptoms),
            top_disease   = top_disease,
            severity      = risk.get("severity_score", 0),
            risk_level    = risk.get("risk_level", "UNKNOWN"),
            predictions   = predictions,
        )
        log.info("[Predict] Saved log_id=%s for user=%s", log_id, current_user.id)
    except Exception as e:
        log.warning("[Predict] MongoDB persist failed (non-fatal): %s", e)

    # ── Step 5c: Audit Log (PostgreSQL) ──────────────────────────────────────
    try:
        from app.models.sql_models import AuditLog
        AuditLog.log(
            event_type="predict",
            user_id=str(current_user.id),
            payload={
                "symptoms": symptoms,
                "top_disease": top_disease,
                "risk_level": risk.get("risk_level"),
                "severity": risk.get("severity_score"),
            },
            ip=request.remote_addr,
        )
    except Exception as e:
        log.warning("[Predict] Audit log failed (non-fatal): %s", e)

    # ── Step 6: Publish to Kafka ──────────────────────────────────────────────
    try:
        kafka_producer.publish("symptom-input", {
            "symptoms":    symptoms,
            "duration":    duration,
            "age":         age,
            "top_disease": top_disease,
            "risk_level":  risk["risk_level"],
            "severity":    risk["severity_score"],
            "log_id":      log_id,
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
        "llm_summary":   llm_summary,
        "rag_knowledge": [
            {
                "disease": r.get("disease", "Unknown"),
                "content": r.get("content", "")[:300]
            }
            for r in rag_results
        ]
    }

    return jsonify(response), 200