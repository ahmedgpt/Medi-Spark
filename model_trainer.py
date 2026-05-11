"""
Day 7: Real dataset trainer for MediSpark
Dataset : C:\\Users\\This pc\\Desktop\\med_spark material\\datase\\
Run     : python -m app.spark.model_trainer
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# ── EXACT PATHS ON YOUR MACHINE ────────────────────────────────────────────────
TRAIN_PATH = r"C:\Users\This pc\Desktop\med_spark material\datase\Training.csv"
TEST_PATH  = r"C:\Users\This pc\Desktop\med_spark material\datase\Testing.csv"

# Models saved inside your project
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_clean():
    """Load Training.csv and Testing.csv and clean them."""

    # ── Guard: check files exist ───────────────────────────────────────────────
    for path, name in [(TRAIN_PATH, "Training.csv"), (TEST_PATH, "Testing.csv")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n❌  Cannot find: {path}\n"
                f"    Make sure the file is named exactly '{name}'\n"
                f"    Current folder contents should show Training and Testing\n"
            )

    print(f"[DATA] Loading Training.csv ...")
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"       Shape: {train_df.shape}")

    print(f"[DATA] Loading Testing.csv ...")
    test_df = pd.read_csv(TEST_PATH)
    print(f"       Shape: {test_df.shape}")

    # ── Standardise column names ───────────────────────────────────────────────
    for df in [train_df, test_df]:
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

    # ── Drop Kaggle's unnamed garbage column ───────────────────────────────────
    for df in [train_df, test_df]:
        bad = [c for c in df.columns if "unnamed" in c]
        if bad:
            df.drop(columns=bad, inplace=True)
            print(f"[DATA] Dropped: {bad}")

    # ── Find target column ─────────────────────────────────────────────────────
    if "prognosis" in train_df.columns:
        target_col = "prognosis"
    elif "disease" in train_df.columns:
        target_col = "disease"
    else:
        target_col = train_df.columns[-1]
        print(f"[DATA] ⚠️  Using last column as target: '{target_col}'")

    symptom_cols = [c for c in train_df.columns if c != target_col]

    # ── Clean values ───────────────────────────────────────────────────────────
    for df in [train_df, test_df]:
        df[symptom_cols] = df[symptom_cols].fillna(0).astype(int)
        df[target_col]   = df[target_col].str.strip()

    print(f"\n[DATA] Target column    : '{target_col}'")
    print(f"[DATA] Symptom features : {len(symptom_cols)}")
    print(f"[DATA] Unique diseases  : {train_df[target_col].nunique()}")
    print(f"[DATA] Train samples    : {len(train_df)}")
    print(f"[DATA] Test  samples    : {len(test_df)}")
    print(f"\n[DATA] Diseases found:")
    for d in sorted(train_df[target_col].unique()):
        print(f"       • {d}")

    return train_df, test_df, symptom_cols, target_col


def build_features(train_df, test_df, symptom_cols, target_col):
    """Encode labels into numeric arrays."""
    le = LabelEncoder()
    le.fit(train_df[target_col])

    X_train = train_df[symptom_cols].values
    y_train = le.transform(train_df[target_col])
    X_test  = test_df[symptom_cols].values
    y_test  = le.transform(test_df[target_col])

    print(f"\n[FEAT] X_train : {X_train.shape}")
    print(f"[FEAT] X_test  : {X_test.shape}")
    print(f"[FEAT] Classes : {len(le.classes_)}")

    return X_train, y_train, X_test, y_test, le


def train_and_evaluate(X_train, y_train, X_test, y_test, le):
    """Train both models, compare, return winner."""

    # ── Random Forest ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[RF] Training Random Forest (200 trees) ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_cv       = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"[RF] Test  accuracy : {rf_test_acc:.4f}")
    print(f"[RF] 5-fold CV      : {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

    # ── XGBoost ────────────────────────────────────────────────────────────────
    print("\n[XGB] Training XGBoost (300 trees) ...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    xgb_test_acc = accuracy_score(y_test, xgb.predict(X_test))
    xgb_cv       = cross_val_score(xgb, X_train, y_train, cv=5, scoring="accuracy")
    print(f"[XGB] Test  accuracy : {xgb_test_acc:.4f}")
    print(f"[XGB] 5-fold CV      : {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")

    # ── Pick winner ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    if xgb_cv.mean() >= rf_cv.mean():
        best, best_name = xgb, "xgboost"
        print("✅  XGBoost wins — saving as primary model.")
    else:
        best, best_name = rf, "random_forest"
        print("✅  Random Forest wins — saving as primary model.")

    print(f"\n[REPORT] Classification report ({best_name}):")
    print(classification_report(
        y_test,
        best.predict(X_test),
        target_names=le.classes_
    ))

    return best, best_name


def save_artifacts(model, le, symptom_cols, model_name):
    """Save the 4 files that ml_predictor.py loads at runtime."""
    joblib.dump(model,         os.path.join(MODEL_DIR, "disease_classifier.pkl"))
    joblib.dump(le,            os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(symptom_cols,  os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump({
        "model_type" : model_name,
        "n_features" : len(symptom_cols),
        "n_classes"  : len(le.classes_),
        "diseases"   : le.classes_.tolist(),
    }, os.path.join(MODEL_DIR, "model_meta.pkl"))

    print(f"\n💾  Saved to: {MODEL_DIR}")
    print(f"    ✔ disease_classifier.pkl  ({model_name})")
    print(f"    ✔ label_encoder.pkl       ({len(le.classes_)} diseases)")
    print(f"    ✔ feature_names.pkl       ({len(symptom_cols)} symptoms)")
    print(f"    ✔ model_meta.pkl")


if __name__ == "__main__":
    train_df, test_df, symptom_cols, target_col   = load_and_clean()
    X_train, y_train, X_test, y_test, le          = build_features(
        train_df, test_df, symptom_cols, target_col
    )
    best_model, best_name                          = train_and_evaluate(
        X_train, y_train, X_test, y_test, le
    )
    save_artifacts(best_model, le, symptom_cols, best_name)
    print("\n🎉  Day 7 COMPLETE — model ready for Week 2 RAG integration!")