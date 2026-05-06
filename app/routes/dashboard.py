"""Dashboard / landing pages."""
from __future__ import annotations

from flask import Blueprint, redirect, render_template, url_for
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


@dashboard_bp.route("/history")
@login_required
def history_page():
    return render_template("history.html")
