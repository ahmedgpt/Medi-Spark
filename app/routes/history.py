"""/api/history — list past symptom logs for the current user."""
from __future__ import annotations

from flask import Blueprint, jsonify
from flask_login import current_user, login_required

from app.models.symptom_log import SymptomLog

history_bp = Blueprint("history", __name__)


@history_bp.route("/history", methods=["GET"])
@login_required
def history():
    items = SymptomLog.list_for_user(current_user.id)
    # ObjectId / datetime → JSON-safe strings
    for it in items:
        if "created_at" in it and it["created_at"]:
            it["created_at"] = it["created_at"].isoformat()
        if it.get("updated_at"):
            it["updated_at"] = it["updated_at"].isoformat()
    return jsonify({"items": items})
