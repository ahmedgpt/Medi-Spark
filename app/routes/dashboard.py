"""Dashboard / landing pages + stats API."""
from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, jsonify, redirect, render_template, url_for
from flask_login import current_user, login_required

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def index():
    if not current_user.is_authenticated:
        return redirect(url_for("auth.login"))
    return render_template("index.html")


@dashboard_bp.route("/predict")
@login_required
def predict_page():
    return render_template("predict.html")


@dashboard_bp.route("/chat")
@login_required
def chat_page():
    return render_template("chat.html")


@dashboard_bp.route("/history")
@login_required
def history_page():
    return render_template("history.html")


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard Stats API — powers Chart.js widgets
# ══════════════════════════════════════════════════════════════════════════════

@dashboard_bp.route("/api/dashboard/stats")
@login_required
def dashboard_stats():
    """
    Return aggregated stats for the logged-in user's dashboard.
    Works with or without MongoDB — returns sensible defaults.
    """
    stats = {
        "total_consultations": 0,
        "risk_distribution": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
        "recent_severity": 0,
        "recent_disease": "—",
        "recent_risk": "UNKNOWN",
        "disease_trend": [],        # [{disease, count}]
        "severity_history": [],     # [{date, severity}]
    }

    try:
        from app.extensions import mongo
        db = mongo.db
        if db is None:
            return jsonify(stats)

        user_id = str(current_user.id)

        # Total consultations for this user
        total = db.symptom_logs.count_documents({"user_id": user_id})
        stats["total_consultations"] = total

        # Risk distribution
        pipeline_risk = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}},
        ]
        for doc in db.symptom_logs.aggregate(pipeline_risk):
            level = doc["_id"]
            if level in stats["risk_distribution"]:
                stats["risk_distribution"][level] = doc["count"]

        # Most recent prediction
        recent = db.symptom_logs.find_one(
            {"user_id": user_id},
            sort=[("created_at", -1)],
        )
        if recent:
            stats["recent_severity"] = recent.get("severity", 0)
            stats["recent_disease"] = recent.get("top_disease", "—")
            stats["recent_risk"] = recent.get("risk_level", "UNKNOWN")

        # Top-5 disease trends (from symptom_trends or symptom_logs)
        pipeline_disease = [
            {"$group": {"_id": "$top_disease", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5},
        ]
        stats["disease_trend"] = [
            {"disease": d["_id"] or "Unknown", "count": d["count"]}
            for d in db.symptom_logs.aggregate(pipeline_disease)
        ]

        # Severity history (last 10 entries)
        severity_docs = list(
            db.symptom_logs.find(
                {"user_id": user_id, "severity": {"$exists": True}},
                {"severity": 1, "created_at": 1, "_id": 0},
            ).sort("created_at", -1).limit(10)
        )
        stats["severity_history"] = [
            {
                "date": d.get("created_at", datetime.now(timezone.utc)).strftime("%m/%d %H:%M"),
                "severity": d.get("severity", 0),
            }
            for d in reversed(severity_docs)
        ]

    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Dashboard stats error: %s", exc)

    return jsonify(stats)
