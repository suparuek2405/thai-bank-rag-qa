"""
loader.py — PDF loading and chunking for Thai Bank RAG system
=============================================================
Loads 56-1 One Report PDFs from Google Drive, extracts text per page,
splits into overlapping chunks, and attaches metadata for filtering.

Usage (in Colab):
    from src.loader import load_bank_pdf, chunk_documents, process_all_banks
"""

import os
import re
import json
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANKS = ["BBL", "KBANK", "KTB", "SCBX", "TTB", "TISCO", "BAY", "LHFG", "CREDIT", "KKP"]

# Default chunking config — tuned later in NB04
DEFAULT_CHUNK_SIZE = 512       # characters (not tokens; see note below)
DEFAULT_OVERLAP    = 100       # characters of overlap between consecutive chunks

# NOTE on "tokens" vs "characters":
#   all-MiniLM-L6-v2 has a 256-token limit (~1024 characters for Thai/English mixed text).
#   We chunk by character count, which is simpler and language-agnostic.
#   512 chars ≈ 128 tokens — well within the embedding model's limit.


# ---------------------------------------------------------------------------
# Step 1: Load a single PDF
# ---------------------------------------------------------------------------

def load_bank_pdf(bank_name: str, pdf_dir: str) -> list[dict]:
    """
    Load one bank's 56-1 PDF and extract text page by page.

    Args:
        bank_name: Bank ticker, e.g. "KBANK"
        pdf_dir:   Directory containing PDFs, e.g. "/content/drive/MyDrive/.../data/raw"

    Returns:
        List of page dicts:
            {
                "bank_name":    "KBANK",
                "source_file":  "KBANK_56-1_2025.pdf",
                "page_number":  int,         # 1-indexed
                "text":         str,         # raw extracted text for this page
                "char_count":   int
            }
    """
    filename = f"{bank_name}_56-1_2025.pdf"
    pdf_path = os.path.join(pdf_dir, filename)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            f"Make sure Google Drive is mounted and {filename} is in {pdf_dir}"
        )

    pages = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # plain text extraction

        # Skip pages with very little text (likely images/covers)
        if len(text.strip()) < 50:
            continue

        pages.append({
            "bank_name":   bank_name,
            "source_file": filename,
            "page_number": page_num + 1,   # 1-indexed for human readability
            "text":        text,
            "char_count":  len(text)
        })

    doc.close()
    return pages


def load_all_banks(pdf_dir: str, banks: list[str] = BANKS) -> dict[str, list[dict]]:
    """
    Load PDFs for all banks. Returns dict keyed by bank ticker.

    Args:
        pdf_dir: Path to directory containing all PDFs
        banks:   List of bank tickers to load

    Returns:
        {"KBANK": [page_dict, ...], "BBL": [...], ...}
    """
    all_pages = {}
    failed = []

    for bank in banks:
        try:
            pages = load_bank_pdf(bank, pdf_dir)
            all_pages[bank] = pages
            total_chars = sum(p["char_count"] for p in pages)
            print(f"  [{bank}] {len(pages)} pages | {total_chars:,} chars")
        except FileNotFoundError as e:
            print(f"  [ERROR] {bank}: {e}")
            failed.append(bank)

    if failed:
        print(f"\nWarning: {len(failed)} bank(s) failed to load: {failed}")
    else:
        print(f"\nAll {len(banks)} banks loaded successfully.")

    return all_pages


# ---------------------------------------------------------------------------
# Step 2: Inspect extraction quality
# ---------------------------------------------------------------------------

def inspect_page(page: dict, n_chars: int = 500) -> None:
    """Print a preview of a page to check text extraction quality."""
    print(f"Bank: {page['bank_name']} | Page: {page['page_number']} | "
          f"Chars: {page['char_count']:,}")
    print("-" * 60)
    print(page["text"][:n_chars])
    print("...")


def extraction_stats(all_pages: dict) -> dict:
    """
    Summarise extraction quality across all banks.

    Returns per-bank stats:
        {bank: {"total_pages": int, "total_chars": int, "avg_chars_per_page": float}}
    """
    stats = {}
    for bank, pages in all_pages.items():
        total_chars = sum(p["char_count"] for p in pages)
        stats[bank] = {
            "total_pages":       len(pages),
            "total_chars":       total_chars,
            "avg_chars_per_page": round(total_chars / len(pages), 0) if pages else 0
        }
    return stats


# ---------------------------------------------------------------------------
# Step 3: Chunking
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Basic cleaning: collapse excessive whitespace, normalise newlines."""
    # Replace multiple blank lines with a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace non-breaking spaces and other odd whitespace
    text = text.replace("\xa0", " ").replace("\t", " ")
    # Collapse multiple spaces into one
    text = re.sub(r"  +", " ", text)
    return text.strip()


def chunk_page(
    page: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> list[dict]:
    """
    Split one page's text into overlapping chunks.

    Each chunk carries full metadata so it can be stored independently
    in the vector store.

    Args:
        page:       Page dict from load_bank_pdf()
        chunk_size: Target chunk length in characters
        overlap:    Number of characters to repeat at chunk boundaries

    Returns:
        List of chunk dicts:
            {
                "bank_name":   str,
                "source_file": str,
                "page_number": int,
                "chunk_index": int,   # position within this page
                "text":        str,
                "char_count":  int
            }
    """
    text = _clean_text(page["text"])
    if not text:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # Try to end at a sentence boundary to avoid cutting mid-sentence
        if end < len(text):
            # Look for ". " or ".\n" within the last 100 chars of the window
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + 1  # include the period

        chunk_text = text[start:end].strip()

        if len(chunk_text) > 20:  # ignore tiny trailing chunks
            chunks.append({
                "bank_name":   page["bank_name"],
                "source_file": page["source_file"],
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "text":        chunk_text,
                "char_count":  len(chunk_text)
            })
            chunk_index += 1

        start = end - overlap

    return chunks


def chunk_documents(
    all_pages: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> list[dict]:
    """
    Chunk all pages across all banks into a flat list of chunk dicts.

    Args:
        all_pages:  Output of load_all_banks()
        chunk_size: Characters per chunk
        overlap:    Characters of overlap

    Returns:
        Flat list of all chunk dicts, ready for embedding.
    """
    all_chunks = []

    for bank, pages in all_pages.items():
        bank_chunks = []
        for page in pages:
            bank_chunks.extend(chunk_page(page, chunk_size, overlap))

        print(f"  [{bank}] {len(pages)} pages → {len(bank_chunks)} chunks "
              f"(avg {len(bank_chunks) // len(pages) if pages else 0} chunks/page)")
        all_chunks.extend(bank_chunks)

    print(f"\nTotal: {len(all_chunks)} chunks across {len(all_pages)} banks")
    return all_chunks


# ---------------------------------------------------------------------------
# Step 4: Save / Load processed chunks
# ---------------------------------------------------------------------------

def save_chunks(
    chunks: list[dict],
    output_dir: str,
    chunk_size: int,
    overlap: int
) -> str:
    """
    Save chunks to JSON with config embedded in filename.

    Args:
        chunks:     List of chunk dicts
        output_dir: Directory to save into (e.g. "data/processed")
        chunk_size: Used in filename for traceability
        overlap:    Used in filename for traceability

    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"chunks_c{chunk_size}_o{overlap}.json"
    output_path = os.path.join(output_dir, filename)

    payload = {
        "config": {
            "chunk_size": chunk_size,
            "overlap":    overlap,
            "total_chunks": len(chunks),
            "banks": sorted(set(c["bank_name"] for c in chunks))
        },
        "chunks": chunks
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(output_path) / 1_000_000
    print(f"Saved {len(chunks):,} chunks → {output_path} ({size_mb:.1f} MB)")
    return output_path


def load_chunks(filepath: str) -> tuple[list[dict], dict]:
    """
    Load chunks from a previously saved JSON file.

    Returns:
        (chunks, config) tuple
    """
    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["chunks"], payload["config"]


# ---------------------------------------------------------------------------
# Convenience: process everything in one call
# ---------------------------------------------------------------------------

def process_all_banks(
    pdf_dir: str,
    output_dir: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    banks: list[str] = BANKS
) -> tuple[list[dict], str]:
    """
    Full pipeline: load PDFs → chunk → save.

    Args:
        pdf_dir:    Path to PDFs (Google Drive)
        output_dir: Path to save chunks JSON
        chunk_size: Characters per chunk
        overlap:    Characters of overlap
        banks:      List of bank tickers

    Returns:
        (chunks, saved_filepath)
    """
    print(f"Loading PDFs from: {pdf_dir}")
    print(f"Config: chunk_size={chunk_size}, overlap={overlap}\n")

    print("--- Loading ---")
    all_pages = load_all_banks(pdf_dir, banks)

    print("\n--- Chunking ---")
    chunks = chunk_documents(all_pages, chunk_size, overlap)

    print("\n--- Saving ---")
    filepath = save_chunks(chunks, output_dir, chunk_size, overlap)

    return chunks, filepath
