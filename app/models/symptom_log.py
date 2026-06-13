"""Symptom log — every prediction request is recorded here for analytics & history."""
from __future__ import annotations

from datetime import datetime, timezone
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
        """Legacy create — stores initial symptom data only (kept for backwards compat)."""
        doc = {
            "user_id":       user_id,
            "symptoms":      symptoms,
            "duration_days": duration_days,
            "age":           age,
            "language":      language,
            "raw_text":      raw_text,
            "prediction":    None,
            "severity":      None,
            "risk_level":    None,
            "created_at":    datetime.now(timezone.utc),
        }
        return str(cls.collection().insert_one(doc).inserted_id)

    @classmethod
    def create_full(
        cls,
        user_id: str,
        symptoms: list[str],
        duration_days: int,
        age: int | None,
        language: str,
        raw_text: str,
        top_disease: str,
        severity: float,
        risk_level: str,
        predictions: list[dict],
    ) -> str:
        """
        Week 4: Persist a complete prediction result in one insert.
        Stores everything the dashboard charts and history page need:
          - top_disease / predictions list       -> disease_trend aggregation
          - severity / severity_score            -> severity_history aggregation
          - risk_level                           -> risk_distribution aggregation
          - created_at                           -> timeline sorting
        """
        doc = {
            "user_id":        user_id,
            "symptoms":       symptoms,
            "duration_days":  duration_days,
            "age":            age,
            "language":       language,
            "raw_text":       raw_text,
            # prediction results
            "top_disease":    top_disease,
            "prediction":     predictions[0] if predictions else None,
            "predictions":    predictions,
            "severity":       severity,
            "severity_score": severity,   # alias used by /api/dashboard/stats query
            "risk_level":     risk_level,
            "created_at":     datetime.now(timezone.utc),
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
            if "created_at" in d and d["created_at"]:
                d["created_at"] = d["created_at"].isoformat()
            if d.get("updated_at"):
                d["updated_at"] = d["updated_at"].isoformat()
            out.append(d)
        return out

    @classmethod
    def update_prediction(cls, log_id: str, prediction: dict[str, Any]) -> None:
        cls.collection().update_one(
            {"_id": ObjectId(log_id)},
            {"$set": {"prediction": prediction, "updated_at": datetime.now(timezone.utc)}},
        )
