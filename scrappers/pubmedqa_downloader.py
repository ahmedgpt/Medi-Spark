"""
PubMedQA Downloader
====================
Downloads the PubMedQA dataset from HuggingFace and saves it to:
    data/medical_knowledge/pubmedqa.json

The RAG engine (app/services/rag_engine.py) automatically ingests this file
alongside MedQuAD and WHO Fact Sheets when building the ChromaDB vector store.

Subsets available:
  - pqa_labeled    :   1,000 expert-annotated Q&A (highest quality)
  - pqa_unlabeled  :  61,249 automatically labelled Q&A
  - pqa_artificial : 211,269 artificially generated Q&A

By default only pqa_labeled + pqa_unlabeled are downloaded (good balance of
quality vs. size). Pass --full to also include pqa_artificial (~211k rows, slow).

Usage
-----
    # From the project root:
    python scrappers/pubmedqa_downloader.py

    # Include the artificial subset too (slower, ~211k extra rows):
    python scrappers/pubmedqa_downloader.py --full

    # Force re-download even if output already exists:
    python scrappers/pubmedqa_downloader.py --force

Requirements
------------
    pip install datasets        (HuggingFace datasets library)
    (already added to requirements.txt)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data" / "medical_knowledge"
OUTPUT_FILE  = OUTPUT_DIR / "pubmedqa.json"

# ── Dataset config ─────────────────────────────────────────────────────────────
DATASET_ID = "qiaojin/PubMedQA"

# Subsets to download by default (skip pqa_artificial unless --full)
DEFAULT_SUBSETS = ["pqa_labeled", "pqa_unlabeled"]
ALL_SUBSETS     = ["pqa_labeled", "pqa_unlabeled", "pqa_artificial"]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _flatten_context(context_field) -> str:
    """
    PubMedQA context is a dict with keys 'contexts' (list of str) and
    'labels' / 'meshes'. Flatten the sentences into a single paragraph.
    """
    if isinstance(context_field, dict):
        sentences = context_field.get("contexts", [])
        return " ".join(str(s).strip() for s in sentences if s)
    if isinstance(context_field, list):
        return " ".join(str(s).strip() for s in context_field if s)
    return str(context_field).strip()


def _convert_row(row: dict) -> dict | None:
    """
    Convert one PubMedQA row into the standard MediSpark RAG document format:
        {title, question, answer, context, source, decision}
    Returns None if the row is missing essential content.
    """
    question    = str(row.get("question") or "").strip()
    long_answer = str(row.get("long_answer") or "").strip()
    context_raw = row.get("context", {})
    context_txt = _flatten_context(context_raw)
    decision    = str(row.get("final_decision") or "").strip()
    pubid       = str(row.get("pubid") or "")

    if not question or not long_answer:
        return None

    # Build a rich content string: question + abstract + conclusion
    content_parts = [f"Question: {question}"]
    if context_txt:
        content_parts.append(f"Abstract: {context_txt}")
    content_parts.append(f"Conclusion: {long_answer}")
    if decision:
        content_parts.append(f"Decision: {decision}")

    return {
        "title":    question[:120],           # short title for metadata
        "question": question,
        "answer":   long_answer,
        "content":  "\n".join(content_parts),
        "decision": decision,
        "pubid":    pubid,
        "source":   "PubMedQA",
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run(full: bool = False, force: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded
    if OUTPUT_FILE.exists() and not force:
        existing = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        print(
            f"[PubMedQA] Output already exists with {len(existing):,} documents.\n"
            f"           Use --force to re-download.\n"
            f"           Path: {OUTPUT_FILE}"
        )
        return

    # Import here so the error message is clear if not installed
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌  HuggingFace `datasets` library not found.")
        print("    Install it with:  pip install datasets")
        sys.exit(1)

    subsets = ALL_SUBSETS if full else DEFAULT_SUBSETS
    all_docs: list[dict] = []
    skipped = 0

    for subset in subsets:
        print(f"\n[PubMedQA] Downloading subset: {subset} ...")
        try:
            ds = load_dataset(DATASET_ID, subset, split="train", trust_remote_code=True)
        except Exception as exc:
            print(f"  [WARN] Failed to load {subset}: {exc}")
            continue

        print(f"[PubMedQA] {subset}: {len(ds):,} rows — converting ...")
        for row in ds:
            doc = _convert_row(row)
            if doc:
                all_docs.append(doc)
            else:
                skipped += 1

        print(f"[PubMedQA] {subset}: converted {len(all_docs):,} docs so far.")

    if not all_docs:
        print("❌  No documents were converted. Check your internet connection.")
        sys.exit(1)

    # Save
    OUTPUT_FILE.write_text(
        json.dumps(all_docs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"✅  PubMedQA download complete!")
    print(f"    Subsets downloaded : {', '.join(subsets)}")
    print(f"    Documents saved    : {len(all_docs):,}")
    print(f"    Skipped (empty)    : {skipped:,}")
    print(f"    Output file        : {OUTPUT_FILE}")
    print("=" * 60)
    print("\nNext step: rebuild the RAG vector store to include this data:")
    print("    python -m app.services.rag_engine --rebuild")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PubMedQA for MediSpark RAG.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also download pqa_artificial (~211k rows, slower).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output file already exists.",
    )
    args = parser.parse_args()
    run(full=args.full, force=args.force)
