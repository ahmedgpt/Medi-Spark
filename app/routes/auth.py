"""Authentication blueprint — register, login, logout.
Week 4: Flask-Limiter applied to login (5/min) and register (10/min) to
prevent brute-force and spam account creation.
"""
from __future__ import annotations

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import login_required, login_user, logout_user

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
        return redirect(url_for("dashboard.index"))

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Signed out.", "info")
    return redirect(url_for("auth.login"))
