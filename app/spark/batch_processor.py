"""
Day 19: PySpark Batch Processor
================================
Reads raw symptom events from MongoDB `symptom_logs` collection,
aggregates them into 6-hour windows, and writes trend summaries to
MongoDB `symptom_trends` collection.

Run standalone
--------------
    python -m app.spark.batch_processor
    python -m app.spark.batch_processor --hours 24   # custom lookback window

Scheduled via cron / APScheduler every 6 hours in production.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] batch_processor: %(message)s",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# ── Environment ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

MONGO_URI   = os.getenv("MONGO_URI",   "mongodb://localhost:27017/medispark")
MONGO_DB    = os.getenv("MONGO_DB",    "medispark")


# ══════════════════════════════════════════════════════════════════════════════
# 1. FETCH RAW LOGS FROM MONGODB (Pandas — no Spark driver needed in dev)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_symptom_logs(lookback_hours: int = 6) -> list[dict]:
    """Fetch raw symptom log documents from the last *lookback_hours* hours."""
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db     = client[MONGO_DB]

        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        docs   = list(
            db["symptom_logs"].find(
                {"created_at": {"$gte": cutoff}},
                {"_id": 0, "symptoms": 1, "top_disease": 1,
                 "risk_level": 1, "severity": 1, "created_at": 1}
            )
        )
        log.info("Fetched %d symptom log(s) from last %dh.", len(docs), lookback_hours)
        return docs
    except Exception as exc:  # noqa: BLE001
        log.error("MongoDB fetch failed: %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 2. PYSPARK AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def run_spark_aggregation(docs: list[dict], window_hours: int = 6) -> list[dict]:
    """
    Use PySpark to aggregate symptom logs into disease-level trend records.

    For each unique top_disease in the window:
      - count of cases
      - average severity score
      - HIGH/MED/LOW risk counts
      - most common individual symptoms
    """
    if not docs:
        log.info("No documents to aggregate.")
        return []

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType, ArrayType

        spark = (
            SparkSession.builder
            .appName("MediSpark-BatchProcessor")
            .master("local[*]")
            .config("spark.driver.memory", "1g")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

        # ── Define schema ───────────────────────────────────────────────────
        schema = StructType([
            StructField("top_disease", StringType(), True),
            StructField("risk_level",  StringType(), True),
            StructField("severity",    FloatType(),  True),
            StructField("created_at",  TimestampType(), True),
        ])

        # Flatten for Spark (drop symptoms list for now)
        flat_docs = []
        for d in docs:
            flat_docs.append({
                "top_disease": d.get("top_disease", "Unknown"),
                "risk_level":  d.get("risk_level",  "UNKNOWN"),
                "severity":    float(d.get("severity", 0.0)),
                "created_at":  d.get("created_at"),
            })

        df = spark.createDataFrame(flat_docs, schema=schema)

        # ── Time-window aggregation ──────────────────────────────────────────
        agg_df = df.groupBy("top_disease").agg(
            F.count("*").alias("case_count"),
            F.round(F.avg("severity"), 2).alias("avg_severity"),
            F.sum(F.when(F.col("risk_level") == "HIGH",   1).otherwise(0)).alias("high_risk_count"),
            F.sum(F.when(F.col("risk_level") == "MEDIUM", 1).otherwise(0)).alias("medium_risk_count"),
            F.sum(F.when(F.col("risk_level") == "LOW",    1).otherwise(0)).alias("low_risk_count"),
        ).orderBy(F.desc("case_count"))

        results = [row.asDict() for row in agg_df.collect()]
        spark.stop()

        log.info("Aggregation complete: %d disease trend(s) computed.", len(results))
        return results

    except ImportError:
        log.warning("PySpark not installed — falling back to pandas aggregation.")
        return _pandas_aggregation(docs)
    except Exception as exc:  # noqa: BLE001
        log.error("Spark aggregation failed: %s — falling back to pandas.", exc)
        return _pandas_aggregation(docs)


def _pandas_aggregation(docs: list[dict]) -> list[dict]:
    """Pandas fallback when PySpark is unavailable (e.g. local dev without Java)."""
    import pandas as pd

    flat = [
        {
            "top_disease": d.get("top_disease", "Unknown"),
            "risk_level":  d.get("risk_level",  "UNKNOWN"),
            "severity":    float(d.get("severity", 0.0)),
        }
        for d in docs
    ]
    df = pd.DataFrame(flat)
    if df.empty:
        return []

    grouped = df.groupby("top_disease").agg(
        case_count    = ("top_disease", "count"),
        avg_severity  = ("severity",    "mean"),
        high_risk_count   = ("risk_level", lambda x: (x == "HIGH").sum()),
        medium_risk_count = ("risk_level", lambda x: (x == "MEDIUM").sum()),
        low_risk_count    = ("risk_level", lambda x: (x == "LOW").sum()),
    ).reset_index()

    grouped["avg_severity"] = grouped["avg_severity"].round(2)
    return grouped.to_dict(orient="records")


# ══════════════════════════════════════════════════════════════════════════════
# 3. WRITE TRENDS TO MONGODB
# ══════════════════════════════════════════════════════════════════════════════

def write_trends(trends: list[dict], window_hours: int = 6) -> None:
    """Upsert aggregated trends into MongoDB `symptom_trends` collection."""
    if not trends:
        log.info("No trends to write.")
        return
    try:
        from pymongo import MongoClient, UpdateOne
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db     = client[MONGO_DB]

        now      = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=window_hours)

        ops = []
        for trend in trends:
            ops.append(
                UpdateOne(
                    {
                        "top_disease":    trend["top_disease"],
                        "window_start":   window_start,
                    },
                    {
                        "$set": {
                            **trend,
                            "window_start": window_start,
                            "window_end":   now,
                            "updated_at":   now,
                        }
                    },
                    upsert=True,
                )
            )

        result = db["symptom_trends"].bulk_write(ops)
        log.info(
            "Trends written: %d upserted, %d modified.",
            result.upserted_count,
            result.modified_count,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to write trends to MongoDB: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TOP-LEVEL RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_batch(lookback_hours: int = 6) -> list[dict]:
    """
    Full batch pipeline:
      fetch → aggregate → write → return trend list.
    Safe to call from APScheduler, Airflow, or manually.
    """
    log.info("=" * 55)
    log.info("MediSpark Batch Processor — lookback=%dh", lookback_hours)
    log.info("=" * 55)

    docs   = fetch_symptom_logs(lookback_hours)
    trends = run_spark_aggregation(docs, window_hours=lookback_hours)
    write_trends(trends, window_hours=lookback_hours)

    log.info("Batch job complete. %d trend record(s) produced.", len(trends))
    return trends


# ══════════════════════════════════════════════════════════════════════════════
# 5. CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediSpark PySpark batch aggregation job.")
    parser.add_argument(
        "--hours", type=int, default=6,
        help="Lookback window in hours (default: 6)."
    )
    args = parser.parse_args()
    run_batch(lookback_hours=args.hours)
