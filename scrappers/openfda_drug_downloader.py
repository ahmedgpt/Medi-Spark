"""
OpenFDA Drug Database Downloader — DrugBank replacement.
=========================================================
Queries the FDA drug labeling API (https://api.fda.gov/drug/label.json)
for each disease in MEDICINE_DB and enriches the drug entries with
official FDA indications, warnings, and drug names.

Also saves results to:
  data/drug_database/drug_entries.json  — flat JSON for offline use
  PostgreSQL drug_database table        — via SQLAlchemy (optional)

Run
---
    python scrappers/openfda_drug_downloader.py

No API key required. Rate limit: 240 requests/minute.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_PATH = BASE_DIR / "data" / "drug_database" / "drug_entries.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

import requests

FDA_BASE = "https://api.fda.gov/drug/label.json"

# Map our disease names → search terms the FDA API understands
DISEASE_SEARCH_MAP: dict[str, list[str]] = {
    "Fungal infection":         ["fungal infection", "antifungal"],
    "Allergy":                  ["allergic rhinitis", "allergy", "antihistamine"],
    "GERD":                     ["gastroesophageal reflux", "GERD", "acid reflux"],
    "Chronic cholestasis":      ["cholestasis", "bile acid"],
    "Drug Reaction":            ["drug hypersensitivity", "drug reaction"],
    "Peptic ulcer diseae":      ["peptic ulcer", "H. pylori"],
    "AIDS":                     ["HIV", "antiretroviral"],
    "Diabetes":                 ["diabetes mellitus", "insulin", "metformin"],
    "Gastroenteritis":          ["gastroenteritis", "diarrhea"],
    "Bronchial Asthma":         ["asthma", "bronchodilator"],
    "Hypertension":             ["hypertension", "antihypertensive"],
    "Migraine":                 ["migraine", "triptan"],
    "Cervical spondylosis":     ["cervical spondylosis", "neck pain"],
    "Jaundice":                 ["jaundice", "liver disease"],
    "Malaria":                  ["malaria", "chloroquine", "artemether"],
    "Chicken pox":              ["varicella", "chickenpox", "acyclovir"],
    "Dengue":                   ["dengue fever", "dengue"],
    "Typhoid":                  ["typhoid fever", "Salmonella typhi"],
    "hepatitis A":              ["hepatitis A"],
    "Hepatitis B":              ["hepatitis B", "tenofovir"],
    "Hepatitis C":              ["hepatitis C", "sofosbuvir"],
    "Tuberculosis":             ["tuberculosis", "isoniazid", "rifampin"],
    "Common Cold":              ["common cold", "rhinovirus"],
    "Pneumonia":                ["pneumonia", "amoxicillin"],
    "Heart attack":             ["myocardial infarction", "heart attack", "aspirin"],
    "Hypertension":             ["hypertension", "amlodipine"],
    "Hypothyroidism":           ["hypothyroidism", "levothyroxine"],
    "Hyperthyroidism":          ["hyperthyroidism", "methimazole"],
    "Diabetes":                 ["diabetes", "metformin", "insulin"],
    "Urinary tract infection":  ["urinary tract infection", "UTI", "nitrofurantoin"],
    "Arthritis":                ["rheumatoid arthritis", "methotrexate"],
    "Acne":                     ["acne vulgaris", "tretinoin", "clindamycin"],
    "Psoriasis":                ["psoriasis", "methotrexate", "cyclosporine"],
    "Migraine":                 ["migraine", "sumatriptan"],
}


def query_fda(search_term: str, limit: int = 5) -> list[dict]:
    """Query FDA drug label API for a given indication/drug name."""
    try:
        resp = requests.get(
            FDA_BASE,
            params={
                "search": f'indications_and_usage:"{search_term}"',
                "limit":  limit,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("results", [])
        if resp.status_code == 404:
            return []
        log.warning("FDA API returned %d for '%s'", resp.status_code, search_term)
        return []
    except Exception as exc:
        log.warning("FDA query failed for '%s': %s", search_term, exc)
        return []


def extract_drug_names(fda_result: dict) -> list[str]:
    """Pull clean drug names out of an FDA label result."""
    names = set()
    openfda = fda_result.get("openfda", {})
    for field in ("brand_name", "generic_name", "substance_name"):
        for n in openfda.get(field, []):
            names.add(n.strip().title())
    return sorted(names)


def build_drug_entries() -> list[dict]:
    """
    Query OpenFDA for each disease and produce a flat list of drug entries.
    Each entry: {disease, drug_name, drug_type, description, source}
    """
    from app.services.medicine_suggester import MEDICINE_DB

    entries: list[dict] = []
    seen: set[tuple] = set()  # (disease, drug_name) dedup

    # 1. Seed from existing static MEDICINE_DB (always included as baseline)
    for disease, meds in MEDICINE_DB.items():
        for drug in meds.get("otc", []):
            key = (disease, drug)
            if key not in seen:
                seen.add(key)
                entries.append({
                    "disease":     disease,
                    "drug_name":   drug,
                    "drug_type":   "otc",
                    "description": "",
                    "source":      "static",
                })
        for drug in meds.get("prescription", []):
            key = (disease, drug)
            if key not in seen:
                seen.add(key)
                entries.append({
                    "disease":     disease,
                    "drug_name":   drug,
                    "drug_type":   "prescription",
                    "description": "",
                    "source":      "static",
                })

    # 2. Enrich from OpenFDA
    log.info("Querying OpenFDA for %d diseases …", len(DISEASE_SEARCH_MAP))
    for disease, search_terms in DISEASE_SEARCH_MAP.items():
        for term in search_terms[:2]:   # limit to first 2 terms per disease
            results = query_fda(term, limit=3)
            for res in results:
                drug_names = extract_drug_names(res)
                indication = (res.get("indications_and_usage") or [""])[0][:200]
                for dname in drug_names[:4]:
                    key = (disease, dname)
                    if key not in seen:
                        seen.add(key)
                        entries.append({
                            "disease":     disease,
                            "drug_name":   dname,
                            "drug_type":   "prescription",
                            "description": indication,
                            "source":      "OpenFDA",
                        })
            time.sleep(0.26)   # stay under 240 req/min FDA rate limit

    log.info("Total drug entries built: %d", len(entries))
    return entries


def save_json(entries: list[dict]) -> None:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    log.info("Saved %d entries to %s", len(entries), OUTPUT_PATH)


def seed_postgres(entries: list[dict]) -> None:
    """Upsert entries into the PostgreSQL drug_database table."""
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
        from flask import Flask
        from config.settings import Config
        from app.extensions import db
        from app.models.sql_models import DrugEntry

        app = Flask(__name__)
        app.config["SQLALCHEMY_DATABASE_URI"] = Config.POSTGRES_URI
        app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
        db.init_app(app)

        with app.app_context():
            db.create_all()
            added = 0
            for e in entries:
                exists = DrugEntry.query.filter_by(
                    disease=e["disease"], drug_name=e["drug_name"]
                ).first()
                if not exists:
                    db.session.add(DrugEntry(
                        disease=e["disease"],
                        drug_name=e["drug_name"],
                        drug_type=e["drug_type"],
                        description=e["description"],
                        source=e["source"],
                    ))
                    added += 1
            db.session.commit()
            log.info("PostgreSQL: inserted %d new drug entries.", added)
    except Exception as exc:
        log.warning("PostgreSQL seed skipped (not required): %s", exc)


if __name__ == "__main__":
    log.info("=== MediSpark Drug Database Builder (OpenFDA) ===")
    entries = build_drug_entries()
    save_json(entries)
    seed_postgres(entries)
    log.info("Done. Run the Flask app to serve enriched drug data.")
