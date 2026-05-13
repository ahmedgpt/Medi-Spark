"""
Day 8-9: RAG Engine — LangChain + ChromaDB
Sources : MedQuAD CSV (16,413 Q&A pairs)
Run     : python -m app.services.rag_engine  (to build the vector store)
"""

import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# ── PATHS ──────────────────────────────────────────────────────────────────────
MEDQUAD_PATH = r"C:\Users\This pc\Desktop\med_spark material\dataset_rag\medquad.csv"

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DIR   = os.path.join(BASE_DIR, "data", "chromadb")
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

    print(f"[RAG] Columns detected → question='{q_col}' answer='{a_col}' disease='{d_col}'")

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
# 2. BUILD / LOAD VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════
def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """
    Build ChromaDB vector store from MedQuAD documents.
    Persists to disk so it only needs to be built once.
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

    # ── Build fresh ───────────────────────────────────────────────────────────
    print("[RAG] Building vector store from scratch (first time only) ...")
    documents = load_medquad()

    # Split long answers into chunks
    splitter  = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks    = splitter.split_documents(documents)
    print(f"[RAG] Total chunks after splitting: {len(chunks)}")

    # Embed in batches and persist
    print("[RAG] Embedding chunks (this takes 2-5 mins first time) ...")
    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    _vectorstore.persist()
    print(f"[RAG] ✅ Vector store built and saved to: {CHROMA_DIR}")
    return _vectorstore


# ══════════════════════════════════════════════════════════════════════════════
# 3. RETRIEVE — called by ml_predictor and Flask routes
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
# 4. MAIN — run this once to build the vector store
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("MediSpark RAG Engine — Day 8 Setup")
    print("=" * 60)

    # Build the vector store
    vs = build_vectorstore(force_rebuild=False)

    # Quick test retrieval
    print("\n[TEST] Running test query ...")
    test_query   = "What are the symptoms of Diabetes?"
    test_results = retrieve(test_query, k=3)

    print(f"\n[TEST] Query: '{test_query}'")
    print(f"[TEST] Top {len(test_results)} results:\n")
    for i, r in enumerate(test_results, 1):
        print(f"  Result {i}:")
        print(f"  Disease : {r['disease']}")
        print(f"  Score   : {r['score']}")
        print(f"  Content : {r['content'][:200]}...")
        print()

    print("🎉  Day 8 COMPLETE — RAG engine ready!")