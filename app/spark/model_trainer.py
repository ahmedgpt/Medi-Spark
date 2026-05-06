"""Train the baseline disease classifier.

Usage:
    python -m app.spark.model_trainer --data data/symptom_dataset/dataset.csv

Dataset format expected (Kaggle Disease Symptom Prediction):
    Disease, Symptom_1, Symptom_2, ..., Symptom_17

If no dataset exists, a tiny synthetic dataset is generated so the rest of the
Week 1 pipeline can still be smoke-tested end-to-end.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from config.settings import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("trainer")


# ---------- Data loading ----------

def _load_kaggle_format(csv_path: Path) -> tuple[list[list[str]], list[str]]:
    df = pd.read_csv(csv_path)
    label_col = next(
        (c for c in df.columns if c.lower() in {"disease", "prognosis", "label"}),
        df.columns[0],
    )
    symptom_cols = [c for c in df.columns if c != label_col]

    samples: list[list[str]] = []
    labels: list[str] = []
    for _, row in df.iterrows():
        syms = [
            str(row[c]).strip().lower().replace(" ", "_")
            for c in symptom_cols
            if pd.notna(row[c]) and str(row[c]).strip()
        ]
        if not syms:
            continue
        samples.append(syms)
        labels.append(str(row[label_col]).strip())
    return samples, labels


def _synthetic_dataset() -> tuple[list[list[str]], list[str]]:
    """Tiny demo dataset so Week 1 can run without downloading anything."""
    log.warning("No dataset supplied — generating a synthetic 6-disease toy set.")
    data = {
        "Common Cold": [["cough", "runny_nose", "sore_throat"], ["cough", "fatigue", "sneezing"]],
        "Influenza": [["high_fever", "fatigue", "cough", "body_ache"], ["high_fever", "chills", "headache"]],
        "Migraine": [["severe_headache", "nausea", "light_sensitivity"], ["severe_headache", "vomiting"]],
        "Gastroenteritis": [["vomiting", "diarrhoea", "abdominal_pain"], ["nausea", "diarrhoea", "fatigue"]],
        "Pneumonia": [["high_fever", "cough", "difficulty_breathing"], ["chest_pain", "cough", "fatigue"]],
        "Hypertension": [["severe_headache", "dizziness", "fatigue"], ["chest_pain", "blurred_vision"]],
    }
    samples, labels = [], []
    for disease, examples in data.items():
        for ex in examples * 25:  # blow it up so RF has something to chew on
            samples.append(ex)
            labels.append(disease)
    return samples, labels


# ---------- Vectorisation ----------

def _build_index(samples: list[list[str]]) -> dict[str, int]:
    vocab = sorted({s for sample in samples for s in sample})
    return {s: i for i, s in enumerate(vocab)}


def _vectorise(samples: list[list[str]], index: dict[str, int]) -> np.ndarray:
    X = np.zeros((len(samples), len(index)), dtype=np.uint8)
    for r, sample in enumerate(samples):
        for s in sample:
            if s in index:
                X[r, index[s]] = 1
    return X


# ---------- Train ----------

def train(csv_path: str | None) -> None:
    if csv_path and Path(csv_path).exists():
        samples, labels = _load_kaggle_format(Path(csv_path))
        log.info("Loaded %d samples from %s", len(samples), csv_path)
    else:
        samples, labels = _synthetic_dataset()

    symptom_index = _build_index(samples)
    X = _vectorise(samples, symptom_index)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    log.info("Validation accuracy: %.3f", acc)
    log.info("\n%s", classification_report(y_test, preds, zero_division=0))

    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    joblib.dump(
        {"model": model, "symptom_index": symptom_index, "classes": list(model.classes_)},
        Config.MODEL_PATH,
    )
    with open(Config.SYMPTOM_INDEX_PATH, "w", encoding="utf-8") as fh:
        json.dump(symptom_index, fh, indent=2)
    log.info("Saved model → %s", Config.MODEL_PATH)
    log.info("Saved symptom index → %s", Config.SYMPTOM_INDEX_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MediSpark baseline classifier.")
    parser.add_argument("--data", default="data/symptom_dataset/dataset.csv")
    args = parser.parse_args()
    train(args.data)


if __name__ == "__main__":
    main()
