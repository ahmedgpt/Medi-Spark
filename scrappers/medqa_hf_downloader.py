"""
HuggingFace Medical QA Downloader — Columbia Medical Corpus substitute.
=======================================================================
Downloads `medalpaca/medical_meadow_medical_flashcards` from HuggingFace
(33 k+ clinical Q&A cards; no authentication required) and saves a clean
JSON file at data/medical_knowledge/medqa_hf.json.

That file is read by rag_engine.load_medqa_hf() and added as a 4th RAG source
alongside MedQuAD, WHO Fact Sheets, and PubMedQA.

Run
---
    pip install datasets
    python scrappers/medqa_hf_downloader.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_PATH = BASE_DIR / "data" / "medical_knowledge" / "medqa_hf.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

HF_DATASET  = "medalpaca/medical_meadow_medical_flashcards"
HF_SPLIT    = "train"
MAX_RECORDS = 20_000   # cap to keep ChromaDB index manageable


def download() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets library not installed. Run: pip install datasets")
        sys.exit(1)

    log.info("Downloading HuggingFace dataset: %s …", HF_DATASET)
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    log.info("Downloaded %d records. Capping to %d.", len(ds), MAX_RECORDS)

    records: list[dict] = []
    for row in ds.select(range(min(MAX_RECORDS, len(ds)))):
        instruction = (row.get("instruction") or "").strip()
        output      = (row.get("output")      or "").strip()
        if not output:
            continue
        content = f"Question: {instruction}\nAnswer: {output}" if instruction else output
        records.append({
            "source":  HF_DATASET,
            "disease": "General Medical",
            "content": content,
        })

    log.info("Processed records: %d", len(records))
    return records


def save(records: list[dict]) -> None:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info("Saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    records = download()
    save(records)
    log.info("Done. Add to RAG by running: python -m app.services.rag_engine --rebuild")
