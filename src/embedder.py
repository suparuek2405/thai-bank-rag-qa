"""
embedder.py — Embedding and ChromaDB vector store for Thai Bank RAG system
==========================================================================
Embeds text chunks using sentence-transformers/all-MiniLM-L6-v2,
stores them in ChromaDB with bank_name metadata for filtering.

Usage (in Colab):
    from src.embedder import build_vectorstore, load_vectorstore, query
"""

import os
import json
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION   = "thai_banks"
CHROMA_DIR   = "/content/chroma_db"   # local Colab storage (fast); copy to Drive after building


# ---------------------------------------------------------------------------
# Embedding function
# ---------------------------------------------------------------------------

def get_embedding_fn(model_name: str = EMBED_MODEL):
    """
    Return a ChromaDB-compatible embedding function using sentence-transformers.

    Model: all-MiniLM-L6-v2
      - Output: 384-dimensional vectors
      - Max input: 256 tokens (~1024 chars)
      - Speed: ~14k sentences/sec on CPU
      - Free, runs entirely locally — no API key needed
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


# ---------------------------------------------------------------------------
# Build ChromaDB vector store
# ---------------------------------------------------------------------------

def build_vectorstore(
    chunks: list[dict],
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION,
    batch_size: int = 500,
    reset: bool = False
) -> chromadb.Collection:
    """
    Embed all chunks and store them in ChromaDB.

    Args:
        chunks:          Flat list of chunk dicts from loader.load_chunks()
        chroma_dir:      Directory to persist ChromaDB (local Colab path)
        collection_name: Name of the ChromaDB collection
        batch_size:      Chunks per embedding batch (controls memory usage)
        reset:           If True, delete and rebuild the collection from scratch

    Returns:
        ChromaDB Collection object ready for querying

    ChromaDB stores per chunk:
        - id:        unique string ID
        - document:  the chunk text
        - embedding: 384-d vector from all-MiniLM-L6-v2
        - metadata:  {bank_name, source_file, page_number, chunk_index}
    """
    client = chromadb.PersistentClient(path=chroma_dir)
    embed_fn = get_embedding_fn()

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for text
    )

    existing = collection.count()
    if existing > 0 and not reset:
        print(f"Collection already has {existing:,} chunks. Use reset=True to rebuild.")
        return collection

    print(f"Embedding {len(chunks):,} chunks in batches of {batch_size}...")
    print(f"Model: {EMBED_MODEL}")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        ids       = [f"{c['bank_name']}_{c['source_file']}_{c['page_number']}_{c['chunk_index']}"
                     for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [{
            "bank_name":   c["bank_name"],
            "source_file": c["source_file"],
            "page_number": c["page_number"],
            "chunk_index": c["chunk_index"]
        } for c in batch]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        done = min(i + batch_size, len(chunks))
        print(f"  {done:,} / {len(chunks):,} chunks embedded", end="\r")

    print(f"\nDone. Collection '{collection_name}' has {collection.count():,} chunks.")
    return collection


# ---------------------------------------------------------------------------
# Load existing vector store
# ---------------------------------------------------------------------------

def load_vectorstore(
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION
) -> chromadb.Collection:
    """Load an existing ChromaDB collection (no re-embedding needed)."""
    client = chromadb.PersistentClient(path=chroma_dir)
    embed_fn = get_embedding_fn()
    collection = client.get_collection(
        name=collection_name,
        embedding_function=embed_fn
    )
    print(f"Loaded collection '{collection_name}' with {collection.count():,} chunks.")
    return collection


# ---------------------------------------------------------------------------
# Query / retrieval
# ---------------------------------------------------------------------------

def query(
    collection: chromadb.Collection,
    question: str,
    bank_name: Optional[str] = None,
    top_k: int = 5
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a question.

    Args:
        collection: ChromaDB collection
        question:   Natural language question
        bank_name:  If set, restrict search to this bank only (single-bank mode)
                    If None, search all banks (cross-bank mode)
        top_k:      Number of chunks to return

    Returns:
        List of result dicts:
            {
                "text":      str,
                "bank_name": str,
                "page":      int,
                "score":     float   # cosine distance (lower = more similar)
            }
    """
    where = {"bank_name": bank_name} if bank_name else None

    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "text":      doc,
            "bank_name": meta["bank_name"],
            "page":      meta["page_number"],
            "score":     round(dist, 4)
        }
        for doc, meta, dist in zip(docs, metas, distances)
    ]


def print_results(results: list[dict], question: str) -> None:
    """Pretty-print retrieval results for inspection."""
    print(f"Query: {question!r}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['bank_name']} p.{r['page']}  (score: {r['score']})")
        print(f"     {r['text'][:200].strip()!r}")
        print()
