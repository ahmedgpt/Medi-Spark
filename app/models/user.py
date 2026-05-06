"""User model — thin wrapper around the Mongo `users` collection."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from bson import ObjectId
from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from app.extensions import mongo


class User(UserMixin):
    def __init__(self, doc: dict[str, Any]):
        self._doc = doc
        self.id = str(doc["_id"])
        self.email = doc["email"]
        self.name = doc.get("name", "")
        self.age = doc.get("age")
        self.password_hash = doc.get("password_hash", "")

    # ---- Flask-Login ----
    def get_id(self) -> str:  # type: ignore[override]
        return self.id

    # ---- Auth helpers ----
    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    # ---- Lookups ----
    @classmethod
    def collection(cls):
        return mongo.db.users

    @classmethod
    def get_by_id(cls, user_id: str) -> "User | None":
        try:
            doc = cls.collection().find_one({"_id": ObjectId(user_id)})
        except Exception:
            return None
        return cls(doc) if doc else None

    @classmethod
    def get_by_email(cls, email: str) -> "User | None":
        doc = cls.collection().find_one({"email": email.lower().strip()})
        return cls(doc) if doc else None

    @classmethod
    def create(cls, email: str, password: str, name: str = "", age: int | None = None) -> "User":
        doc = {
            "email": email.lower().strip(),
            "name": name.strip(),
            "age": age,
            "password_hash": generate_password_hash(password),
            "created_at": datetime.utcnow(),
        }
        result = cls.collection().insert_one(doc)
        doc["_id"] = result.inserted_id
        return cls(doc)
