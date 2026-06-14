"""
Medicine Suggester — DrugBank replacement via OpenFDA + static fallback.

Resolution order:
  1. PostgreSQL drug_database table (populated by scrappers/openfda_drug_downloader.py)
  2. data/drug_database/drug_entries.json  (offline cache from the same downloader)
  3. Hardcoded MEDICINE_DB (always-available baseline)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_DATA_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "drug_database" / "drug_entries.json"

# ── Static baseline (always available, no external dependencies) ──────────────
MEDICINE_DB = {
    "Fungal infection":     {"otc": ["Clotrimazole cream", "Fluconazole 150mg"],       "prescription": ["Itraconazole", "Terbinafine"]},
    "Allergy":              {"otc": ["Cetirizine 10mg", "Loratadine"],                  "prescription": ["Montelukast", "Fexofenadine"]},
    "GERD":                 {"otc": ["Omeprazole 20mg", "Antacids"],                    "prescription": ["Pantoprazole", "Esomeprazole"]},
    "Chronic cholestasis":  {"otc": ["Vitamin supplements"],                            "prescription": ["Ursodeoxycholic acid", "Cholestyramine"]},
    "Drug Reaction":        {"otc": ["Antihistamines", "Hydrocortisone cream"],         "prescription": ["Prednisolone", "Epinephrine (severe)"]},
    "Peptic ulcer diseae":  {"otc": ["Antacids", "Omeprazole 20mg"],                   "prescription": ["Clarithromycin", "Amoxicillin", "Pantoprazole"]},
    "AIDS":                 {"otc": ["Multivitamins", "Probiotics"],                    "prescription": ["Antiretroviral therapy (ART) — consult specialist"]},
    "Diabetes":             {"otc": ["Glucose monitoring kit"],                         "prescription": ["Metformin", "Insulin", "Glipizide"]},
    "Gastroenteritis":      {"otc": ["ORS sachets", "Loperamide", "Zinc tablets"],     "prescription": ["Ciprofloxacin", "Metronidazole"]},
    "Bronchial Asthma":     {"otc": ["Salbutamol inhaler"],                             "prescription": ["Budesonide inhaler", "Montelukast", "Prednisolone"]},
    "Hypertension":         {"otc": ["Lifestyle changes", "Low-sodium diet"],           "prescription": ["Amlodipine", "Lisinopril", "Losartan"]},
    "Migraine":             {"otc": ["Ibuprofen 400mg", "Paracetamol"],                 "prescription": ["Sumatriptan", "Topiramate", "Amitriptyline"]},
    "Cervical spondylosis": {"otc": ["Ibuprofen gel", "Diclofenac gel"],               "prescription": ["Gabapentin", "Muscle relaxants", "Physiotherapy"]},
    "Paralysis (brain hemorrhage)": {"otc": [],                                         "prescription": ["Emergency neurology — hospitalisation required"]},
    "Jaundice":             {"otc": ["Rest", "Hydration", "Avoid alcohol"],             "prescription": ["Ursodeoxycholic acid", "Liver support medications"]},
    "Malaria":              {"otc": ["Paracetamol for fever"],                          "prescription": ["Chloroquine", "Artemether-Lumefantrine", "Primaquine"]},
    "Chicken pox":          {"otc": ["Calamine lotion", "Paracetamol", "Antihistamines"], "prescription": ["Acyclovir", "Valacyclovir"]},
    "Dengue":               {"otc": ["Paracetamol", "ORS", "Rest"],                    "prescription": ["Hospital monitoring — no specific antiviral"]},
    "Typhoid":              {"otc": ["ORS", "Paracetamol"],                             "prescription": ["Ciprofloxacin", "Azithromycin", "Ceftriaxone"]},
    "hepatitis A":          {"otc": ["Rest", "Hydration", "Avoid alcohol"],             "prescription": ["Supportive care — no specific antiviral"]},
    "Hepatitis B":          {"otc": ["Rest", "Hydration"],                              "prescription": ["Tenofovir", "Entecavir", "Interferon alfa"]},
    "Hepatitis C":          {"otc": ["Rest", "Hydration"],                              "prescription": ["Sofosbuvir", "Daclatasvir", "Ribavirin"]},
    "Hepatitis D":          {"otc": ["Rest", "Hydration"],                              "prescription": ["Pegylated interferon alfa"]},
    "Hepatitis E":          {"otc": ["Rest", "Hydration", "Avoid alcohol"],             "prescription": ["Supportive care"]},
    "Alcoholic hepatitis":  {"otc": ["Vitamins B1, B6, B12"],                          "prescription": ["Prednisolone", "Pentoxifylline", "Liver transplant evaluation"]},
    "Tuberculosis":         {"otc": ["Vitamin B6 supplement"],                          "prescription": ["Isoniazid", "Rifampicin", "Pyrazinamide", "Ethambutol"]},
    "Common Cold":          {"otc": ["Paracetamol", "Antihistamines", "Decongestants"], "prescription": ["Antibiotics only if bacterial secondary infection"]},
    "Pneumonia":            {"otc": ["Paracetamol", "Rest", "Hydration"],               "prescription": ["Amoxicillin", "Azithromycin", "Ceftriaxone"]},
    "Dimorphic hemmorhoids(piles)": {"otc": ["Sitz bath", "Fiber supplements", "Witch hazel cream"], "prescription": ["Hydrocortisone suppositories", "Surgical referral"]},
    "Heart attack":         {"otc": ["Aspirin 325mg (immediately)"],                   "prescription": ["Emergency hospitalisation — call ambulance NOW"]},
    "Varicose veins":       {"otc": ["Compression stockings", "Elevation of legs"],    "prescription": ["Sclerotherapy", "Surgical referral"]},
    "Hypothyroidism":       {"otc": ["Selenium supplements"],                           "prescription": ["Levothyroxine"]},
    "Hyperthyroidism":      {"otc": ["Avoid caffeine and iodine"],                      "prescription": ["Methimazole", "Propylthiouracil", "Beta-blockers"]},
    "Hypoglycemia":         {"otc": ["Glucose tablets", "Fruit juice", "Sugar"],        "prescription": ["Glucagon injection (severe)", "IV dextrose"]},
    "Osteoarthristis":      {"otc": ["Ibuprofen", "Paracetamol", "Glucosamine"],       "prescription": ["Celecoxib", "Physiotherapy", "Joint injection"]},
    "Arthritis":            {"otc": ["Ibuprofen", "Diclofenac gel"],                   "prescription": ["Methotrexate", "Hydroxychloroquine", "Biologics"]},
    "Acne":                 {"otc": ["Benzoyl peroxide", "Salicylic acid wash"],        "prescription": ["Tretinoin", "Clindamycin", "Isotretinoin"]},
    "Urinary tract infection": {"otc": ["Cranberry juice", "Increase water intake"],   "prescription": ["Nitrofurantoin", "Trimethoprim", "Ciprofloxacin"]},
    "Psoriasis":            {"otc": ["Moisturisers", "Coal tar shampoo"],               "prescription": ["Methotrexate", "Cyclosporine", "Biologics"]},
    "Impetigo":             {"otc": ["Antiseptic wash"],                                "prescription": ["Fusidic acid cream", "Flucloxacillin"]},
    "(vertigo) Paroymsal  Positional Vertigo": {"otc": ["Meclizine", "Antihistamines"], "prescription": ["Betahistine", "Epley manoeuvre by specialist"]},
}

DEFAULT_MEDICINES = {
    "otc":          ["Paracetamol 500mg", "ORS sachets", "Multivitamins"],
    "prescription": ["Consult a doctor for prescription medicines"]
}


def _query_postgres(disease: str) -> dict | None:
    """
    Try to fetch drug entries from the PostgreSQL drug_database table.
    Returns {"otc": [...], "prescription": [...]} or None if unavailable.
    """
    try:
        from app.models.sql_models import DrugEntry
        rows = DrugEntry.query.filter_by(disease=disease).all()
        if not rows:
            return None
        otc  = [r.drug_name for r in rows if r.drug_type == "otc"]
        rx   = [r.drug_name for r in rows if r.drug_type == "prescription"]
        if otc or rx:
            return {"otc": otc, "prescription": rx}
    except Exception as exc:
        log.debug("[MedicineSuggester] PostgreSQL query skipped: %s", exc)
    return None


def _query_json_cache(disease: str) -> dict | None:
    """
    Fallback: read drug_entries.json built by openfda_drug_downloader.py.
    Returns {"otc": [...], "prescription": [...]} or None.
    """
    if not _DATA_FILE.exists():
        return None
    try:
        with open(_DATA_FILE, encoding="utf-8") as f:
            entries: list[dict] = json.load(f)
        otc = [e["drug_name"] for e in entries if e.get("disease") == disease and e.get("drug_type") == "otc"]
        rx  = [e["drug_name"] for e in entries if e.get("disease") == disease and e.get("drug_type") == "prescription"]
        if otc or rx:
            return {"otc": otc, "prescription": rx}
    except Exception as exc:
        log.debug("[MedicineSuggester] JSON cache read failed: %s", exc)
    return None


def suggest_medicines(disease: str, medicine_type: str = "otc") -> dict:
    """
    Suggest medicines for a given disease and risk level.

    Resolution order: PostgreSQL → JSON cache → static MEDICINE_DB → defaults.

    Args:
        disease      : predicted disease name
        medicine_type: 'otc' | 'prescription'
    Returns:
        dict with otc, prescription, recommended, source keys
    """
    source = "static"

    # 1. PostgreSQL
    entry = _query_postgres(disease)
    if entry:
        source = "postgresql"
    else:
        # 2. JSON cache (OpenFDA offline data)
        entry = _query_json_cache(disease)
        if entry:
            source = "openfda_cache"
        else:
            # 3. Static baseline
            entry = MEDICINE_DB.get(disease, DEFAULT_MEDICINES)

    otc = entry.get("otc", DEFAULT_MEDICINES["otc"])
    rx  = entry.get("prescription", DEFAULT_MEDICINES["prescription"])

    return {
        "disease":      disease,
        "otc":          otc,
        "prescription": rx,
        "recommended":  otc if medicine_type == "otc" else rx,
        "source":       source,
    }
