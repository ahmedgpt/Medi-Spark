"""
Day 8-9: RAG Engine — LangChain + ChromaDB
Sources : 1) MedQuAD CSV       (16,413 Q&A pairs)
          2) WHO Fact Sheets JSON (scraped by scrappers/who_scraper.py)
          3) PubMedQA JSON       (downloaded by scrappers/pubmedqa_downloader.py)
Run     : python -m app.services.rag_engine           (load existing)
          python -m app.services.rag_engine --rebuild  (force full rebuild)
"""

import os
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MEDQUAD_PATH  = os.getenv("MEDQUAD_PATH",  os.path.join(BASE_DIR, "data", "medical_knowledge", "medquad.csv"))
WHO_PATH      = os.path.join(BASE_DIR, "data", "medical_knowledge", "who_factsheets.json")
PUBMEDQA_PATH = os.path.join(BASE_DIR, "data", "medical_knowledge", "pubmedqa.json")
CHROMA_DIR    = os.path.join(BASE_DIR, "data", "chromadb")
os.makedirs(CHROMA_DIR, exist_ok=True)

# ── EMBEDDING MODEL (free, runs locally) ───────────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"

# ── Singleton vector store ─────────────────────────────────────────────────────
_vectorstore = None


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN MEDQUAD CSV
# ══════════════════════════════════════════════════════════════════════════════
def load_medquad() -> list[Document]:
    """Load MedQuAD CSV and convert to LangChain Documents."""
    print(f"[RAG] Loading MedQuAD from:\n      {MEDQUAD_PATH}")

    if not os.path.exists(MEDQUAD_PATH):
        raise FileNotFoundError(
            f"\n❌  MedQuAD CSV not found at:\n    {MEDQUAD_PATH}\n"
        )

    df = pd.read_csv(MEDQUAD_PATH)
    print(f"[RAG] Raw shape: {df.shape}")

    # ── Standardise column names ───────────────────────────────────────────────
    df.columns = df.columns.str.strip().str.lower()

    # Detect question / answer / disease columns flexibly
    q_col  = next((c for c in df.columns if "question" in c), None)
    a_col  = next((c for c in df.columns if "answer"   in c), None)
    d_col  = next((c for c in df.columns if "disease"  in c or "focus" in c), None)

    print(f"[RAG] Columns detected: question='{q_col}' answer='{a_col}' disease='{d_col}'")

    # Drop rows with missing answers (useless for RAG)
    df.dropna(subset=[a_col], inplace=True)
    df[a_col] = df[a_col].astype(str).str.strip()
    df         = df[df[a_col] != ""]
    print(f"[RAG] Clean rows: {len(df)}")

    # ── Convert to LangChain Documents ────────────────────────────────────────
    documents = []
    for _, row in df.iterrows():
        question = str(row[q_col]).strip() if q_col else ""
        answer   = str(row[a_col]).strip()
        disease  = str(row[d_col]).strip() if d_col else "General"

        # Combine Q+A as the document content so retrieval matches questions too
        content  = f"Disease: {disease}\nQuestion: {question}\nAnswer: {answer}"

        documents.append(Document(
            page_content=content,
            metadata={
                "disease":  disease,
                "question": question,
                "source":   "MedQuAD"
            }
        ))

    print(f"[RAG] Documents created: {len(documents)}")
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD PUBMEDQA JSON
# ══════════════════════════════════════════════════════════════════════════════
def load_pubmedqa() -> list[Document]:
    """Load PubMedQA JSON produced by scrappers/pubmedqa_downloader.py.
    Returns an empty list (with a warning) if the file does not exist yet.
    """
    import json

    path = os.getenv("PUBMEDQA_PATH", PUBMEDQA_PATH)

    if not os.path.exists(path):
        print(f"[RAG] PubMedQA not found at {path} — skipping.")
        print("      Run: python scrappers/pubmedqa_downloader.py  to generate this file.")
        return []

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    documents = []
    for rec in records:
        content = rec.get("content", "").strip()
        if not content:
            continue
        documents.append(Document(
            page_content=content,
            metadata={
                "disease": rec.get("title", "General")[:120],
                "source":  "PubMedQA",
                "pubid":   rec.get("pubid", ""),
            }
        ))

    print(f"[RAG] PubMedQA documents loaded: {len(documents):,}")
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOAD WHO FACT SHEETS JSON
# ══════════════════════════════════════════════════════════════════════════════
def load_who_factsheets() -> list[Document]:
    """Load WHO fact sheets JSON produced by scrappers/who_scraper.py.
    Returns an empty list (with a warning) if the file does not exist yet.
    """
    import json

    path = os.getenv("WHO_PATH", WHO_PATH)

    if not os.path.exists(path):
        print(f"[RAG] WHO fact sheets not found at {path} - skipping.")
        print("      Run: python scrappers/who_scraper.py  to generate this file.")
        return []

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    documents = []
    for rec in records:
        content = rec.get("content", "").strip()
        if not content:
            continue
        documents.append(Document(
            page_content=content,
            metadata={
                "disease": rec.get("title", "General"),
                "source":  "WHO Fact Sheets",
                "url":     rec.get("url", ""),
            }
        ))

    print(f"[RAG] WHO documents loaded: {len(documents):,}")
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD / LOAD VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════
def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """
    Build ChromaDB vector store from MedQuAD + WHO Fact Sheets + PubMedQA.
    Persists to disk so it only needs to be built once.
    Pass force_rebuild=True (or use --rebuild flag) to re-embed everything.
    """
    global _vectorstore
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

    # ── Load from disk if already built ───────────────────────────────────────
    if not force_rebuild and os.path.exists(os.path.join(CHROMA_DIR, "chroma.sqlite3")):
        print("[RAG] Loading existing ChromaDB from disk ...")
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        print(f"[RAG] Loaded {_vectorstore._collection.count()} vectors.")
        return _vectorstore

    # ── Build fresh — merge ALL sources ──────────────────────────────────────
    print("[RAG] Building vector store from scratch ...")
    documents  = load_medquad()
    documents += load_who_factsheets()
    documents += load_pubmedqa()
    print(f"[RAG] Total documents across all sources: {len(documents):,}")

    # Split long answers into chunks
    splitter  = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks    = splitter.split_documents(documents)
    print(f"[RAG] Total chunks after splitting: {len(chunks)}")

    # Embed and persist
    print("[RAG] Embedding chunks (this may take several minutes) ...")
    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"[RAG] Done. Vector store built and saved to: {CHROMA_DIR}")
    return _vectorstore


# ══════════════════════════════════════════════════════════════════════════════
# 4. RETRIEVE — called by ml_predictor and Flask routes
# ══════════════════════════════════════════════════════════════════════════════
def retrieve(query: str, k: int = 3) -> list[dict]:
    """
    Retrieve top-k relevant medical Q&A chunks for a given query.

    Args:
        query : e.g. "What are the symptoms of Diabetes?"
        k     : number of results to return

    Returns:
        [{"content": str, "disease": str, "source": str, "score": float}, ...]
    """
    global _vectorstore
    if _vectorstore is None:
        build_vectorstore()

    results = _vectorstore.similarity_search_with_score(query, k=k)

    return [
        {
            "content": doc.page_content,
            "disease": doc.metadata.get("disease", "Unknown"),
            "source":  doc.metadata.get("source",  "MedQuAD"),
            "score":   round(float(score), 4)
        }
        for doc, score in results
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN — run to build or rebuild the vector store
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build MediSpark RAG vector store.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force full rebuild even if ChromaDB already exists.")
    args = parser.parse_args()

    print("=" * 60)
    print("MediSpark RAG Engine — Multi-Source Build")
    print("=" * 60)

    vs = build_vectorstore(force_rebuild=args.rebuild)

    print("\n[TEST] Running test query ...")
    test_query   = "What are the symptoms of Diabetes?"
    test_results = retrieve(test_query, k=3)

    print(f"\n[TEST] Query: '{test_query}'")
    print(f"[TEST] Top {len(test_results)} results:\n")
    for i, r in enumerate(test_results, 1):
        print(f"  Result {i}:")
        print(f"  Source  : {r['source']}")
        print(f"  Disease : {r['disease']}")
        print(f"  Score   : {r['score']}")
        print(f"  Content : {r['content'][:200]}...")
        print()

    print("\n✅  RAG engine ready! Sources: MedQuAD + WHO Fact Sheets + PubMedQA")