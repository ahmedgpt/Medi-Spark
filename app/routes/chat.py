"""/api/chat — Week 1 stub (echoes message and publishes to chat-messages)."""
from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request
from flask_login import current_user, login_required

from app.services.kafka_producer import get_producer

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message."}), 400

    event = {"user_id": current_user.id, "message": message}
    try:
        get_producer().send(
            current_app.config["TOPIC_CHAT_MESSAGES"], value=event, key=current_user.id
        )
    except Exception:  # noqa: BLE001
        pass  # Week 1 — non-fatal

    # Week 1 placeholder reply. Week 3 wires this up to LangChain.
    return jsonify(
        {
            "reply": (
                "Thanks — I've received your message. The conversational AI "
                "pipeline is enabled in Week 3."
            )
        }
    )
