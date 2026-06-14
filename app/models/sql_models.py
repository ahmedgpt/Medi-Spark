"""
PostgreSQL models via SQLAlchemy.
Tables:
  - audit_logs   : structured event log (all predict/chat events)
  - drug_database: relational drug-disease mapping (DrugBank replacement)
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.extensions import db


class AuditLog(db.Model):
    __tablename__ = "audit_logs"

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.String(64), index=True, nullable=True)
    event_type = db.Column(db.String(64), nullable=False)   # 'predict' | 'chat' | 'login' | 'register'
    payload    = db.Column(db.JSON,       nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True),
                           default=lambda: datetime.now(timezone.utc))

    @classmethod
    def log(cls, event_type: str, user_id: str | None = None,
            payload: dict | None = None, ip: str | None = None) -> None:
        """Insert an audit record. Silently swallows DB errors (non-fatal)."""
        try:
            entry = cls(
                event_type=event_type,
                user_id=user_id,
                payload=payload or {},
                ip_address=ip,
            )
            db.session.add(entry)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            import logging
            logging.getLogger(__name__).warning("AuditLog insert failed: %s", exc)

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "user_id":    self.user_id,
            "event_type": self.event_type,
            "payload":    self.payload,
            "ip_address": self.ip_address,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DrugEntry(db.Model):
    """
    Relational drug-disease mapping — replaces the hardcoded MEDICINE_DB dict.
    Populated by scripts/populate_drug_db.py (OpenFDA / enhanced static data).
    """
    __tablename__ = "drug_database"

    id           = db.Column(db.Integer, primary_key=True)
    disease      = db.Column(db.String(120), nullable=False, index=True)
    drug_name    = db.Column(db.String(200), nullable=False)
    drug_type    = db.Column(db.String(20),  nullable=False)   # 'otc' | 'prescription'
    description  = db.Column(db.Text,        nullable=True)
    source       = db.Column(db.String(64),  nullable=True)    # 'OpenFDA' | 'RxNorm' | 'static'
    created_at   = db.Column(db.DateTime(timezone=True),
                             default=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "disease":     self.disease,
            "drug_name":   self.drug_name,
            "drug_type":   self.drug_type,
            "description": self.description,
            "source":      self.source,
        }
