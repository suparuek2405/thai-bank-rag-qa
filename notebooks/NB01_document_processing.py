# =============================================================================
# NB01 — Document Processing
# Thai Bank Financial Q&A System
# =============================================================================
# Copy each cell block into a separate Colab cell.
# Run them in order top-to-bottom.
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
# Run time: ~30 seconds

get_ipython().system('pip install pymupdf langchain langchain-community -q')

import fitz
import langchain
print(f"PyMuPDF:   {fitz.__version__}")
print(f"LangChain: {langchain.__version__}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Mount Google Drive, clone repo, set paths
# ─────────────────────────────────────────────────────────────────────────────
# PDFs live on Drive (too large for GitHub). Repo code is cloned from GitHub.

from google.colab import drive
drive.mount('/content/drive')

import sys, os

# ── Paths ──────────────────────────────────────────────────────────────────
DRIVE_ROOT    = "/content/drive/MyDrive/Github experiment/thai-bank-rag-qa"
PDF_DIR       = f"{DRIVE_ROOT}/data/raw"
PROCESSED_DIR = f"{DRIVE_ROOT}/data/processed"   # save processed files to Drive so they persist

REPO_DIR = "/content/thai-bank-rag-qa"

# ── Clone repo (or pull latest if already cloned) ─────────────────────────
if not os.path.exists(REPO_DIR):
    get_ipython().system(
        "git clone https://github.com/suparuek2405/thai-bank-rag-qa.git /content/thai-bank-rag-qa"
    )
else:
    get_ipython().system("cd /content/thai-bank-rag-qa && git pull --rebase origin main")

sys.path.insert(0, REPO_DIR)

# ── Quick sanity check: list PDFs on Drive ────────────────────────────────
pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
print(f"Found {len(pdf_files)} PDFs in Drive:")
for f in pdf_files:
    size_mb = os.path.getsize(os.path.join(PDF_DIR, f)) / 1_000_000
    print(f"  {f}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Load one PDF and inspect extraction quality
# ─────────────────────────────────────────────────────────────────────────────
# Goal: see what raw text looks like before chunking.
# Try this for 2-3 banks to check quality — especially CREDIT (was 57 MB).

from src.loader import load_bank_pdf, inspect_page, extraction_stats

# Pick any bank to inspect first
TEST_BANK = "KBANK"

pages = load_bank_pdf(TEST_BANK, PDF_DIR)
print(f"\n{TEST_BANK}: {len(pages)} text-bearing pages")

# Preview page 1 and a middle page
print("\n=== Page 1 ===")
inspect_page(pages[0])

print("\n=== Page 10 ===")
inspect_page(pages[9])


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Load ALL 10 banks and print extraction stats
# ─────────────────────────────────────────────────────────────────────────────

from src.loader import load_all_banks

BANKS = ["BBL", "KBANK", "KTB", "SCBX", "TTB", "TISCO", "BAY", "LHFG", "CREDIT", "KKP"]

print("Loading all 10 bank PDFs...\n")
all_pages = load_all_banks(PDF_DIR, BANKS)

# Summary table
stats = extraction_stats(all_pages)
print(f"\n{'Bank':<8} {'Pages':>6} {'Total chars':>14} {'Avg chars/page':>15}")
print("-" * 47)
for bank, s in stats.items():
    print(f"{bank:<8} {s['total_pages']:>6,} {s['total_chars']:>14,} {s['avg_chars_per_page']:>15,.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Inspect a page with financial table content
# ─────────────────────────────────────────────────────────────────────────────
# Find pages that likely contain financial tables (look for "%" or "million")

def find_financial_pages(pages, keywords=["NPL", "NIM", "ROE", "Capital", "ล้านบาท"], n=3):
    """Return up to n pages that contain any of the keywords."""
    hits = []
    for p in pages:
        if any(kw in p["text"] for kw in keywords):
            hits.append(p)
        if len(hits) >= n:
            break
    return hits

for bank in ["KBANK", "KTB", "CREDIT"]:
    hits = find_financial_pages(all_pages[bank])
    if hits:
        print(f"\n=== {bank} — Financial page sample ===")
        inspect_page(hits[0], n_chars=600)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Explore chunking: compare 3 chunk sizes on one page
# ─────────────────────────────────────────────────────────────────────────────
# This is a visual inspection step — no right/wrong yet.
# Goal: see which size keeps financial sentences intact.

from src.loader import chunk_page

# Use a financial page from KBANK for the comparison
sample_page = find_financial_pages(all_pages["KBANK"])[0]

print(f"Source page: {sample_page['bank_name']} p.{sample_page['page_number']} "
      f"({sample_page['char_count']} chars)\n")

for chunk_size, overlap in [(256, 50), (512, 100), (1024, 100)]:
    chunks = chunk_page(sample_page, chunk_size=chunk_size, overlap=overlap)
    print(f"chunk_size={chunk_size:4d}, overlap={overlap:3d}  →  {len(chunks)} chunks")

    # Show the first chunk so we can compare content boundaries
    print(f"  [First chunk preview] {chunks[0]['text'][:200]!r}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Chunk all banks with 3 configs and count totals
# ─────────────────────────────────────────────────────────────────────────────
# We run all 3 configs so NB04 can compare them.
# We'll choose the best one based on RAGAS scores — not guessing now.

from src.loader import chunk_documents, save_chunks

CONFIGS = [
    {"chunk_size": 256,  "overlap": 50},
    {"chunk_size": 512,  "overlap": 100},
    {"chunk_size": 1024, "overlap": 100},
]

chunk_sets = {}

for cfg in CONFIGS:
    cs, ov = cfg["chunk_size"], cfg["overlap"]
    print(f"\n── Config: chunk_size={cs}, overlap={ov} ──")
    chunks = chunk_documents(all_pages, chunk_size=cs, overlap=ov)
    chunk_sets[(cs, ov)] = chunks
    save_chunks(chunks, PROCESSED_DIR, chunk_size=cs, overlap=ov)

print("\nAll configs saved to", PROCESSED_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Verify metadata on a sample chunk
# ─────────────────────────────────────────────────────────────────────────────
# Metadata is what enables bank-level filtering later.
# Every chunk MUST have: bank_name, source_file, page_number, chunk_index.

sample_chunks = chunk_sets[(512, 100)]

print("Sample chunk (first 3 fields shown as metadata, text truncated):\n")
for chunk in sample_chunks[:3]:
    print({k: v for k, v in chunk.items() if k != "text"})
    print(f"  text preview: {chunk['text'][:100]!r}")
    print()

# Verify all banks present
banks_in_chunks = set(c["bank_name"] for c in sample_chunks)
print(f"Banks in chunk set: {sorted(banks_in_chunks)}")
assert banks_in_chunks == set(BANKS), f"Missing banks: {set(BANKS) - banks_in_chunks}"
print("✓ All 10 banks present in chunk set")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Summary stats per bank (512-char config)
# ─────────────────────────────────────────────────────────────────────────────

from collections import Counter

chunks_512 = chunk_sets[(512, 100)]
per_bank = Counter(c["bank_name"] for c in chunks_512)

print(f"Chunk distribution (chunk_size=512, overlap=100):\n")
print(f"{'Bank':<8} {'Chunks':>8} {'Share':>8}")
print("-" * 26)
total = sum(per_bank.values())
for bank in BANKS:
    n = per_bank.get(bank, 0)
    print(f"{bank:<8} {n:>8,} {n/total:>7.1%}")
print(f"{'TOTAL':<8} {total:>8,}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Push src/loader.py to GitHub
# ─────────────────────────────────────────────────────────────────────────────
# Replace YOUR_TOKEN with your actual GitHub Personal Access Token.
# Go to GitHub > Settings > Developer settings > Personal access tokens > Tokens (classic)
# Scope needed: repo

import getpass

TOKEN = getpass.getpass("GitHub Personal Access Token: ")
USER  = "suparuek2405"
REPO  = "thai-bank-rag-qa"

get_ipython().system(f"""
    cd {REPO_DIR} && \
    git config user.email "suparuek2405@gmail.com" && \
    git config user.name "Suparuek Wattananupan" && \
    git remote set-url origin https://{USER}:{TOKEN}@github.com/{USER}/{REPO}.git && \
    git pull --rebase origin main && \
    git add src/loader.py .gitignore && \
    git commit -m "[NB01] loader.py: PDF extraction + chunking for 10 banks" && \
    git push origin main
""")

print("\nDone! Check https://github.com/suparuek2405/thai-bank-rag-qa")
