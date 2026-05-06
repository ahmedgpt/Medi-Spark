"""Authentication blueprint — register, login, logout."""
from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required, login_user, logout_user

from app.models.user import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        name = request.form.get("name", "").strip()
        age_raw = request.form.get("age", "").strip()

        if not email or not password:
            flash("Email and password are required.", "danger")
            return render_template("register.html")

        if User.get_by_email(email):
            flash("An account with that email already exists.", "warning")
            return render_template("register.html")

        age = int(age_raw) if age_raw.isdigit() else None
        user = User.create(email=email, password=password, name=name, age=age)
        login_user(user)
        flash("Welcome to MediSpark!", "success")
        return redirect(url_for("dashboard.index"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = User.get_by_email(email)
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
