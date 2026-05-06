"""Flask extension singletons (initialised in the app factory)."""
from __future__ import annotations

import redis
from flask_login import LoginManager
from flask_pymongo import PyMongo
from flask_session import Session

mongo = PyMongo()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message_category = "info"

server_session = Session()

# Raw redis client (used by Flask-Session and ad-hoc cache reads)
redis_client: "redis.Redis | None" = None


def init_redis(url: str) -> "redis.Redis":
    global redis_client
    redis_client = redis.from_url(url, decode_responses=False)
    return redis_client
