"""Flask application factory."""
from __future__ import annotations

import logging

from flask import Flask

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

    # Blueprints
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
