"""Flask application factory — Week 4: security hardened."""
from __future__ import annotations

import logging
import re

from flask import Flask, jsonify, request

try:
    from config.settings import Config
except ImportError:  # pragma: no cover
    from ..config.settings import Config

from .extensions import init_redis, login_manager, mongo, server_session


def create_app(config_class: type[Config] = Config) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(config_class)

    logging.basicConfig(
        level=logging.DEBUG if config_class.DEBUG else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Redis-backed Flask sessions
    app.config["SESSION_REDIS"] = init_redis(config_class.REDIS_URL)
    server_session.init_app(app)

    # Mongo
    mongo.init_app(app, uri=config_class.MONGO_URI)

    # Login
    login_manager.init_app(app)

    from .models.user import User  # noqa: WPS433  (local import avoids cycles)

    @login_manager.user_loader
    def load_user(user_id: str):
        return User.get_by_id(user_id)

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address

        limiter = Limiter(
            get_remote_address,
            app=app,
            default_limits=["60 per minute"],
            storage_uri=config_class.REDIS_URL,
        )
        # Stricter limits on API endpoints
        limiter.limit("30 per minute")(lambda: None)  # will be applied via decorators
        app.limiter = limiter
    except ImportError:
        logging.getLogger(__name__).warning(
            "flask-limiter not installed — rate limiting disabled."
        )

    # ── Security Headers ──────────────────────────────────────────────────────
    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    # ── Input Sanitization (strip HTML/script tags from JSON bodies) ──────────
    _TAG_RE = re.compile(r"<[^>]+>")

    def _sanitize(value):
        if isinstance(value, str):
            return _TAG_RE.sub("", value).strip()
        if isinstance(value, dict):
            return {k: _sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitize(v) for v in value]
        return value

    @app.before_request
    def sanitize_json_input():
        if request.is_json and request.data:
            try:
                raw = request.get_json(silent=True)
                if raw:
                    sanitized = _sanitize(raw)
                    # Store sanitized data for route handlers
                    request._sanitized_json = sanitized
            except Exception:
                pass

    # ── Blueprints ────────────────────────────────────────────────────────────
    from .routes.auth import auth_bp
    from .routes.chat import chat_bp
    from .routes.dashboard import dashboard_bp
    from .routes.history import history_bp
    from .routes.predict import predict_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(chat_bp, url_prefix="/api")
    app.register_blueprint(history_bp, url_prefix="/api")

    @app.context_processor
    def inject_globals():
        return {"app_name": "MediSpark"}

    return app
