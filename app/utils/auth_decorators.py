"""
Dual-auth decorator: accepts either a Flask-Login session cookie
OR a Flask-JWT-Extended Bearer token.

Usage
-----
    from app.utils.auth_decorators import jwt_or_login_required

    @bp.route("/api/something")
    @jwt_or_login_required
    def something():
        # current_user is set for session auth
        # get_jwt_identity() returns user_id for JWT auth
        ...
"""
from __future__ import annotations

from functools import wraps

from flask import jsonify, request
from flask_login import current_user


def jwt_or_login_required(fn):
    """Allow access via session (Flask-Login) OR JWT Bearer token."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # 1. Check for Bearer token in Authorization header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
                from app.models.user import User
                import flask_login

                verify_jwt_in_request()
                user_id = get_jwt_identity()
                user = User.get_by_id(user_id)
                if user:
                    flask_login.login_user(user)   # populate current_user for the duration
                return fn(*args, **kwargs)
            except Exception as exc:
                return jsonify({"error": f"Invalid token: {exc}"}), 401

        # 2. Fall back to session-based Flask-Login
        if not current_user.is_authenticated:
            return jsonify({"error": "Authentication required. Provide a session cookie or Bearer token."}), 401

        return fn(*args, **kwargs)

    return wrapper
