"""
/api/chat — Week 3 full implementation.

Endpoints
---------
POST   /api/chat                 — send a message, get AI reply
GET    /api/chat/history         — fetch current user's chat history
DELETE /api/chat/reset           — clear conversation memory
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required

from app.services.chatbot import chat as chatbot_chat, get_history, reset_memory

chat_bp = Blueprint("chat", __name__)


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/chat
# Body: { "message": "I have fever and cough" }
# ──────────────────────────────────────────────────────────────────────────────
@chat_bp.route("/chat", methods=["POST"])
@login_required
def chat():
    """
    Multi-turn medical chatbot endpoint.

    Accepts messages in English, Native Urdu, or Roman Urdu.
    Returns an AI-generated reply with RAG context.
    """
    data    = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Empty message."}), 400

    if len(message) > 1000:
        return jsonify({"error": "Message too long. Maximum 1000 characters."}), 400

    try:
        result = chatbot_chat(
            user_id=str(current_user.id),
            message=message,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Chat service error: {exc}"}), 500

    return jsonify({
        "reply":         result["reply"],
        "detected_lang": result["detected_lang"],
        "translated":    result.get("translated"),
        "sources":       result.get("sources", []),
    })


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/chat/history
# ──────────────────────────────────────────────────────────────────────────────
@chat_bp.route("/chat/history", methods=["GET"])
@login_required
def chat_history():
    """Return the current user's conversation history."""
    history = get_history(str(current_user.id))
    return jsonify({"history": history})


# ──────────────────────────────────────────────────────────────────────────────
# DELETE /api/chat/reset
# ──────────────────────────────────────────────────────────────────────────────
@chat_bp.route("/chat/reset", methods=["DELETE"])
@login_required
def chat_reset():
    """Clear the current user's conversation memory."""
    reset_memory(str(current_user.id))
    return jsonify({"status": "ok", "message": "Conversation memory cleared."})
