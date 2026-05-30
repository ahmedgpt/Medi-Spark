"""
Days 20-21: Continuous Learning Pipeline
==========================================
Collects new labeled symptom-disease data from MongoDB, retrains the
XGBoost classifier (via PySpark MLlib or sklearn fallback), runs an A/B
evaluation, and auto-deploys the new model if it outperforms the current
one by at least MODEL_AB_THRESHOLD percent.

Usage
-----
    python -m app.spark.continuous_learner              # run once
    python -m app.spark.continuous_learner --force      # skip A/B gate, always deploy

Triggered automatically when Kafka receives a `retrain-trigger` event
(published by kafka_consumer once MIN_RETRAIN_SAMPLES new logs accumulate).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] continuous_learner: %(message)s",
)

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

MONGO_URI      = os.getenv("MONGO_URI",          "mongodb://localhost:27017/medispark")
MONGO_DB       = os.getenv("MONGO_DB",           "medispark")
MODEL_DIR      = BASE_DIR / "models"
AB_THRESHOLD   = float(os.getenv("MODEL_AB_THRESHOLD", "2.0"))   # minimum % improvement
MIN_SAMPLES    = int(os.getenv("MIN_RETRAIN_SAMPLES",  "50"))


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def collect_new_samples() -> list[dict]:
    """
    Pull symptom-disease records from MongoDB `symptom_logs` that have been
    user-confirmed (labeled=True) and not yet used for training.
    Returns list of dicts with keys: symptoms (list), disease (str).
    """
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db     = client[MONGO_DB]

        # Fetch confirmed logs (user validated the predicted disease)
        docs = list(
            db["symptom_logs"].find(
                {"labeled": True, "used_for_training": {"$ne": True}},
                {"_id": 1, "symptoms": 1, "confirmed_disease": 1}
            )
        )
        log.info("Found %d new labeled sample(s).", len(docs))
        return docs
    except Exception as exc:  # noqa: BLE001
        log.error("Could not fetch new samples: %s", exc)
        return []


def mark_as_used(doc_ids: list) -> None:
    """Mark MongoDB documents as consumed for training."""
    try:
        from pymongo import MongoClient
        from bson import ObjectId
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db     = client[MONGO_DB]
        db["symptom_logs"].update_many(
            {"_id": {"$in": [ObjectId(str(d)) for d in doc_ids]}},
            {"$set": {"used_for_training": True, "trained_at": datetime.now(timezone.utc)}}
        )
        log.info("Marked %d sample(s) as used.", len(doc_ids))
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not mark samples as used: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD ORIGINAL TRAINING DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_original_data() -> tuple:
    """
    Load the original symptom dataset (132-feature binary matrix) used in Week 1.
    Returns (X, y, feature_names, label_encoder).
    """
    import joblib
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

    # Try to load from CSV if available
    csv_candidates = [
        BASE_DIR / "data" / "symptom_dataset" / "dataset.csv",
        BASE_DIR / "data" / "symptom_dataset" / "Training.csv",
        BASE_DIR / "data" / "symptom_dataset" / "training.csv",
    ]
    for p in csv_candidates:
        if p.exists():
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            prognosis_col = next(
                (c for c in df.columns if "prognosis" in c or "disease" in c), None
            )
            if prognosis_col:
                X_df = df.drop(columns=[prognosis_col])
                X_df = X_df.reindex(columns=feature_names, fill_value=0)
                y_raw = df[prognosis_col].astype(str)
                y = label_encoder.transform(y_raw)
                log.info("Loaded %d original training rows from %s.", len(df), p)
                return X_df.values, y, feature_names, label_encoder

    # Fallback: return empty arrays (new samples only)
    log.warning("Original CSV not found — training on new samples only.")
    return None, None, feature_names, label_encoder


# ══════════════════════════════════════════════════════════════════════════════
# 3. MERGE + FEATURISE NEW SAMPLES
# ══════════════════════════════════════════════════════════════════════════════

def featurise_new_samples(
    new_docs: list[dict],
    feature_names: list[str],
    label_encoder,
) -> tuple:
    """
    Convert raw symptom lists + confirmed disease labels into (X, y) arrays.
    Skips samples with diseases not in the existing label encoder vocabulary.
    """
    import numpy as np

    X_rows, y_rows = [], []
    skipped = 0
    known_classes = set(label_encoder.classes_)

    for doc in new_docs:
        disease  = doc.get("confirmed_disease", "")
        symptoms = doc.get("symptoms", [])

        if disease not in known_classes:
            skipped += 1
            continue

        normalised = {s.strip().lower().replace(" ", "_") for s in symptoms}
        vec = np.array(
            [1 if feat in normalised else 0 for feat in feature_names],
            dtype=int
        )
        X_rows.append(vec)
        y_rows.append(label_encoder.transform([disease])[0])

    if skipped:
        log.info("Skipped %d sample(s) with unseen disease labels.", skipped)

    if not X_rows:
        return None, None

    import numpy as np
    return np.array(X_rows), np.array(y_rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4. RETRAIN
# ══════════════════════════════════════════════════════════════════════════════

def retrain(X_train, y_train) -> object:
    """
    Train a new XGBoost classifier.
    Tries PySpark MLlib first; falls back to sklearn XGBoost.
    Returns the trained sklearn model (joblib-serialisable).
    """
    import numpy as np

    log.info("Training new model on %d samples …", len(y_train))

    # ── Try PySpark MLlib ─────────────────────────────────────────────────────
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.classification import GBTClassifier
        from pyspark.ml import Pipeline
        import pandas as pd

        spark = (
            SparkSession.builder
            .appName("MediSpark-ContinuousLearner")
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

        # Build a pandas df → Spark df
        cols = [f"f{i}" for i in range(X_train.shape[1])]
        pdf  = pd.DataFrame(X_train, columns=cols)
        pdf["label"] = y_train.astype(float)
        sdf  = spark.createDataFrame(pdf)

        assembler = VectorAssembler(inputCols=cols, outputCol="features")
        gbt = GBTClassifier(
            labelCol="label", featuresCol="features",
            maxIter=50, maxDepth=5, seed=42
        )
        pipeline = Pipeline(stages=[assembler, gbt])
        model    = pipeline.fit(sdf)
        spark.stop()
        log.info("PySpark GBT model trained successfully.")

        # Wrap as sklearn-compatible predictor for A/B testing
        class SparkModelWrapper:
            def __init__(self, pipeline_model, feature_cols):
                self._model = pipeline_model
                self._cols  = feature_cols
                self._spark = None

            def predict_proba(self, X):
                """sklearn-like predict_proba interface."""
                import pandas as pd
                from pyspark.sql import SparkSession
                spark = SparkSession.builder.master("local[*]").getOrCreate()
                pdf = pd.DataFrame(X, columns=self._cols)
                sdf = spark.createDataFrame(pdf)
                preds = self._model.transform(sdf)
                proba = [[r["probability"][i] for i in range(len(r["probability"]))]
                         for r in preds.select("probability").collect()]
                spark.stop()
                import numpy as np
                return np.array(proba)

            def predict(self, X):
                proba = self.predict_proba(X)
                return proba.argmax(axis=1)

        return SparkModelWrapper(model, cols)

    except ImportError:
        log.info("PySpark not available — using sklearn XGBoost.")
    except Exception as exc:  # noqa: BLE001
        log.warning("PySpark training failed (%s) — using sklearn XGBoost.", exc)

    # ── sklearn XGBoost fallback ──────────────────────────────────────────────
    from xgboost import XGBClassifier
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    log.info("sklearn XGBoost model trained successfully.")
    return clf


# ══════════════════════════════════════════════════════════════════════════════
# 5. A/B EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test) -> float:
    """Return accuracy (0–100) of *model* on the hold-out test set."""
    import numpy as np
    from sklearn.metrics import accuracy_score

    try:
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds) * 100
        return round(acc, 2)
    except Exception as exc:  # noqa: BLE001
        log.error("Evaluation failed: %s", exc)
        return 0.0


def ab_test(new_model, X_test, y_test) -> tuple[float, float, bool]:
    """
    Compare new model against current deployed model.

    Returns (old_acc, new_acc, should_deploy).
    """
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Load current model
    try:
        current_model = joblib.load(MODEL_DIR / "disease_classifier.pkl")
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load current model for A/B test: %s", exc)
        return 0.0, 0.0, True  # force deploy if no baseline

    old_acc = evaluate(current_model, X_test, y_test)
    new_acc = evaluate(new_model,     X_test, y_test)

    log.info("A/B Test — Current: %.2f%% | New: %.2f%% | Threshold: +%.1f%%",
             old_acc, new_acc, AB_THRESHOLD)

    should_deploy = (new_acc - old_acc) >= AB_THRESHOLD
    return old_acc, new_acc, should_deploy


# ══════════════════════════════════════════════════════════════════════════════
# 6. DEPLOY
# ══════════════════════════════════════════════════════════════════════════════

def deploy_model(new_model, old_acc: float, new_acc: float) -> None:
    """Back up the current model and deploy the new one."""
    import joblib

    ts      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup  = MODEL_DIR / f"disease_classifier_backup_{ts}.pkl"
    current = MODEL_DIR / "disease_classifier.pkl"

    # Backup old model
    if current.exists():
        shutil.copy(current, backup)
        log.info("Backed up current model to: %s", backup)

    # Save new model
    joblib.dump(new_model, current)
    log.info("New model deployed: %.2f%% → %.2f%%", old_acc, new_acc)

    # Write deployment log to MongoDB
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db     = client[MONGO_DB]
        db["model_deployments"].insert_one({
            "deployed_at":  datetime.now(timezone.utc),
            "old_accuracy": old_acc,
            "new_accuracy": new_acc,
            "improvement":  round(new_acc - old_acc, 2),
            "backup_path":  str(backup),
        })
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not log deployment to MongoDB: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# 7. TOP-LEVEL RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_continuous_learning(force_deploy: bool = False) -> dict:
    """
    Full pipeline:
      collect → check threshold → load data → merge → retrain → A/B → deploy?

    Returns a result summary dict.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    log.info("=" * 55)
    log.info("MediSpark Continuous Learner — force_deploy=%s", force_deploy)
    log.info("=" * 55)

    # 1. Collect new samples
    new_docs = collect_new_samples()
    if len(new_docs) < MIN_SAMPLES and not force_deploy:
        msg = (f"Only {len(new_docs)} new sample(s); need {MIN_SAMPLES} before retraining. "
               "Skipping.")
        log.info(msg)
        return {"status": "skipped", "reason": msg, "new_samples": len(new_docs)}

    # 2. Load original data
    X_orig, y_orig, feature_names, label_encoder = load_original_data()

    # 3. Featurise new samples
    X_new, y_new = featurise_new_samples(new_docs, feature_names, label_encoder)

    # 4. Merge
    if X_orig is not None and X_new is not None:
        X = np.vstack([X_orig, X_new])
        y = np.concatenate([y_orig, y_new])
    elif X_orig is not None:
        X, y = X_orig, y_orig
    elif X_new is not None:
        X, y = X_new, y_new
    else:
        log.error("No training data available at all — aborting.")
        return {"status": "error", "reason": "no training data"}

    if len(np.unique(y)) < 2:
        log.error("Need at least 2 classes in training data.")
        return {"status": "error", "reason": "insufficient class diversity"}

    # 5. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Retrain
    new_model = retrain(X_train, y_train)

    # 7. A/B test
    old_acc, new_acc, should_deploy = ab_test(new_model, X_test, y_test)

    if force_deploy:
        should_deploy = True
        log.info("Force-deploy flag set — skipping A/B gate.")

    # 8. Deploy (or skip)
    if should_deploy:
        deploy_model(new_model, old_acc, new_acc)
        # Mark new samples as consumed
        doc_ids = [d["_id"] for d in new_docs if "_id" in d]
        mark_as_used(doc_ids)
        return {
            "status":       "deployed",
            "old_accuracy": old_acc,
            "new_accuracy": new_acc,
            "new_samples":  len(new_docs),
            "total_train":  len(y_train),
        }
    else:
        log.info(
            "New model (%.2f%%) does not improve over current (%.2f%%) by %.1f%% — not deployed.",
            new_acc, old_acc, AB_THRESHOLD,
        )
        return {
            "status":       "not_deployed",
            "old_accuracy": old_acc,
            "new_accuracy": new_acc,
            "new_samples":  len(new_docs),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 8. CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediSpark Continuous Learning Pipeline.")
    parser.add_argument(
        "--force", action="store_true",
        help="Skip A/B gate and always deploy the new model.",
    )
    args   = parser.parse_args()
    result = run_continuous_learning(force_deploy=args.force)
    print("\n" + "=" * 55)
    print("Result:", json.dumps(result, indent=2))
