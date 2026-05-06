"""Symptom log — every prediction request is recorded here for analytics & history."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from bson import ObjectId

from app.extensions import mongo


class SymptomLog:
    @staticmethod
    def collection():
        return mongo.db.symptom_logs

    @classmethod
    def create(
        cls,
        user_id: str,
        symptoms: list[str],
        duration_days: int,
        age: int | None,
        language: str,
        raw_text: str,
    ) -> str:
        doc = {
            "user_id": user_id,
            "symptoms": symptoms,
            "duration_days": duration_days,
            "age": age,
            "language": language,
            "raw_text": raw_text,
            "prediction": None,  # filled by consumer / week 2
            "severity_score": None,
            "risk_level": None,
            "created_at": datetime.utcnow(),
        }
        return str(cls.collection().insert_one(doc).inserted_id)

    @classmethod
    def list_for_user(cls, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        cursor = (
            cls.collection()
            .find({"user_id": user_id})
            .sort("created_at", -1)
            .limit(limit)
        )
        out = []
        for d in cursor:
            d["_id"] = str(d["_id"])
            out.append(d)
        return out

    @classmethod
    def update_prediction(cls, log_id: str, prediction: dict[str, Any]) -> None:
        cls.collection().update_one(
            {"_id": ObjectId(log_id)},
            {"$set": {"prediction": prediction, "updated_at": datetime.utcnow()}},
        )
