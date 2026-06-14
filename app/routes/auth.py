"""Authentication blueprint — register, login, logout, JWT token endpoints.

Endpoints
---------
POST /auth/register          — create account (web form)
POST /auth/login             — login (web form, sets session cookie)
GET  /auth/logout            — logout (clears session)
POST /auth/token             — issue JWT access + refresh tokens (API clients)
POST /auth/token/refresh     — exchange refresh token for new access token
"""
from __future__ import annotations

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for
from flask_login import login_required, login_user, logout_user
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity

from app.models.user import User

auth_bp = Blueprint("auth", __name__)


def _limiter():
    """Return the app-level Flask-Limiter instance, or None if not configured."""
    return getattr(current_app, "limiter", None)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Rate limit: 10 registration attempts per minute per IP
        lim = _limiter()
        if lim:
            lim.limit("10 per minute")(lambda: None)()

        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        name     = request.form.get("name", "").strip()
        age_raw  = request.form.get("age", "").strip()

        if not email or not password:
            flash("Email and password are required.", "danger")
            return render_template("register.html")

        if User.get_by_email(email):
            flash("An account with that email already exists.", "warning")
            return render_template("register.html")

        age  = int(age_raw) if age_raw.isdigit() else None
        user = User.create(email=email, password=password, name=name, age=age)
        login_user(user)
        try:
            from app.models.sql_models import AuditLog
            AuditLog.log("register", user_id=str(user.id), ip=request.remote_addr)
        except Exception:
            pass
        flash("Welcome to MediSpark!", "success")
        return redirect(url_for("dashboard.index"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Rate limit: 5 login attempts per minute per IP  (brute-force protection)
        lim = _limiter()
        if lim:
            try:
                lim.limit("5 per minute")(lambda: None)()
            except Exception:
                flash("Too many login attempts. Please wait a minute and try again.", "danger")
                return render_template("login.html"), 429

        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user     = User.get_by_email(email)
        if not user or not user.check_password(password):
            flash("Invalid credentials.", "danger")
            return render_template("login.html")

        login_user(user, remember=True)
        try:
            from app.models.sql_models import AuditLog
            AuditLog.log("login", user_id=str(user.id), ip=request.remote_addr)
        except Exception:
            pass
        return redirect(url_for("dashboard.index"))

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Signed out.", "info")
    return redirect(url_for("auth.login"))


# ── JWT Token endpoints (for REST / mobile API clients) ───────────────────────

@auth_bp.route("/token", methods=["POST"])
def issue_token():
    """
    Issue JWT access + refresh tokens.
    Accepts JSON: { "email": "...", "password": "..." }
    Returns:      { "access_token": "...", "refresh_token": "..." }
    """
    data     = request.get_json(silent=True) or {}
    email    = (data.get("email") or "").strip()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "email and password required"}), 400

    user = User.get_by_email(email)
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid credentials"}), 401

    access_token  = create_access_token(identity=str(user.id))
    refresh_token = create_refresh_token(identity=str(user.id))

    try:
        from app.models.sql_models import AuditLog
        AuditLog.log("jwt_login", user_id=str(user.id), ip=request.remote_addr)
    except Exception:
        pass

    return jsonify({
        "access_token":  access_token,
        "refresh_token": refresh_token,
        "user_id":       str(user.id),
        "email":         user.email,
    }), 200


@auth_bp.route("/token/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh_token():
    """Exchange a valid refresh token for a new access token."""
    identity     = get_jwt_identity()
    access_token = create_access_token(identity=identity)
    return jsonify({"access_token": access_token}), 200
