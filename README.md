# Thai Bank Financial Q&A 🏦
> RAG pipeline over 10 Thai banks' FY2025 annual filings: 51,674 chunks, metadata-filtered retrieval, and RAGAS evaluation tracked with MLflow

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--db-orange)](https://trychroma.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-0.4.x-purple)](https://docs.ragas.io)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)](https://mlflow.org)
[![Gemini](https://img.shields.io/badge/Gemini-2.5--Flash-red?logo=google)](https://ai.google.dev)

---

## Project Summary

I built a RAG pipeline over FY2025 56-1 One Reports for 10 Thai commercial banks, covering 3,282 pages and 9.5 million characters of financial disclosures. The system routes questions to either a bank-filtered or cross-bank retrieval path, passes the top chunks to Gemini 2.5 Flash, and returns answers with source page references. Four notebooks cover document processing, vector indexing, the RAG pipeline, and RAGAS evaluation.

**Best result: chunk=512, top_k=10, faithfulness 0.7357 on the validation set (0.4935 on held-out test).**

---

## Results

### Validation Set (15 questions, judge: Gemini 2.5 Flash)

| Config | Faithfulness | Context Precision | Context Recall | Mean |
|---|---|---|---|---|
| **chunk=512, top_k=10 (baseline)** | **0.7357** | **0.2958** | **0.3333** | **0.4549** |
| chunk=512, top_k=20 | 0.6083 | 0.3124 | 0.3333 | 0.4180 |
| chunk=256, top_k=10 | 0.5394 | 0.1824 | 0.3333 | 0.3517 |

### Test Set (10 questions, best config only, run once)

| Faithfulness | Context Precision | Context Recall |
|---|---|---|
| 0.4935 | 0.0861 | 0.1000 |

Baseline wins on faithfulness. More context (top_k=20) slightly improves precision but adds retrieval noise that hurts the LLM's ability to stay grounded. Smaller chunks (256-char) score worst overall.

---

## The Story

### Chapter 1: 9.5 Million Characters from 10 Banks

I loaded FY2025 56-1 One Reports using PyMuPDF and extracted text from 3,282 pages across BBL, KBANK, KTB, SCBX, TTB, TISCO, BAY, LHFG, CREDIT, and KKP. The pipeline split text into overlapping chunks with three size configurations (256, 512, and 1024 chars) to test which best preserves financial table context for retrieval.

Financial ratio tables come out of PyMuPDF as plain left-to-right text. Chunk boundaries that cut through a table row lose the connection between a metric name and its value — which turned out to matter more than expected.

| Bank | Pages | Chars | Chunks (512-char) |
|---|---|---|---|
| BBL | 233 | 732K | 2,068 |
| KBANK | 180 | 1,204K | 3,399 |
| KTB | 393 | 1,316K | 3,878 |
| SCBX | 361 | 1,004K | 2,907 |
| TTB | 195 | 499K | 1,512 |
| TISCO | 364 | 899K | 2,614 |
| BAY | 487 | 1,038K | 3,006 |
| LHFG | 329 | 871K | 2,469 |
| CREDIT | 302 | 631K | 2,034 |
| KKP | 438 | 1,344K | 3,926 |
| **Total** | **3,282** | **9.5M** | **27,813** |

### Chapter 2: Embeddings Cluster by Topic, Not by Bank

All 27,813 chunks were embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-d, local, no API cost) and stored in ChromaDB with `bank_name`, `source_file`, `page_number`, and `chunk_index` as metadata. A UMAP projection of 2,000 sampled embeddings shows that chunks cluster by financial topic across banks rather than by which bank they came from. NPL chunks from KTB and BBL land near each other; NIM chunks from all 10 banks form their own neighborhood.

![UMAP of chunk embeddings colored by bank](https://raw.githubusercontent.com/suparuek2405/thai-bank-rag-qa/main/results/figures/umap_embeddings.png)

This means cross-bank retrieval works on embedding similarity alone, but single-bank queries need a metadata filter on `bank_name`. Without it, a question about KTB's NIM will pull the most similar text from whichever bank has the closest embedding, which may not be KTB.

### Chapter 3: Routing Matters as Much as Retrieval

The pipeline classifies each question as single-bank or cross-bank before querying ChromaDB. Single-bank queries apply a `where: {bank_name: X}` metadata filter; cross-bank queries search the full 51,674-chunk collection. Both paths pass the top-k chunks into a financial system prompt and then into Gemini 2.5 Flash.

Getting the router right turned out to be as important as chunk size. A cross-bank question routed as single-bank retrieves only one bank's data and produces a factually wrong comparison. 15 validation questions and 10 test questions were written before building the pipeline to avoid tuning the router toward the evaluation set.

### Chapter 4: 512-char Chunks Win; More Context Hurts Faithfulness

RAGAS scored three configurations on Faithfulness (is the answer grounded in context?), Context Precision (are the retrieved chunks useful?), and Context Recall (did retrieval surface what was needed?). A separate Gemini 2.5 Flash instance served as the judge to keep evaluation independent from the generation LLM. Experiments were tracked in MLflow with a SQLite backend saved to Google Drive.

The baseline (chunk=512, top_k=10) scored 0.74 faithfulness. Doubling retrieval to top_k=20 slightly improved precision but dropped faithfulness to 0.61, the LLM picks up noise from the extra chunks. The 256-char chunks performed worst: smaller chunks do reduce topic dilution in embeddings, but they strip enough surrounding context that the LLM has less to work with per chunk. All three configurations scored identically on context recall (0.33), pointing to a ceiling in how much top-10 retrieval can surface given how the questions are structured.

---

## Key Decisions and Lessons

**Setting aside test questions first.** Writing 25 questions before touching the pipeline forced a clean val/test split. The 15 validation questions were used freely to tune chunk size, top-k, and the system prompt. The 10 test questions were run exactly once at the end. The gap between val faithfulness (0.74) and test faithfulness (0.49) reveals how much the system was tuned to the val set, which is exactly what this design is supposed to show.

**256-char chunks are not always better.** The UMAP visualization suggested that 512-char embeddings suffer from topic dilution, so 256-char chunks looked like they would help retrieval quality. In practice they scored worst. Financial tables contain multiple related metrics in a short span of text, and splitting them more aggressively breaks the relationship between a metric's label and its value. Embedding quality improved slightly; answer quality did not.

**More context hurts faithfulness.** Increasing top_k from 10 to 20 added more relevant chunks but also more noise. Faithfulness dropped from 0.74 to 0.61. For a domain like financial filings where the LLM needs to stay tightly grounded in specific numbers, retrieval precision matters more than recall at the generation stage.

**Gemini 2.5 Flash needs a concurrency limit for RAGAS.** Running 45 evaluation jobs with the default worker setting caused all jobs to stall silently at 0% progress. Setting `RunConfig(max_workers=2, timeout=120)` fixed the stall. Each val config took 10-16 minutes to score. This is a real constraint to plan around for fast evaluation iteration.

---

## Project Structure

    thai-bank-rag-qa/
    ├── notebooks/
    │   ├── NB01_document_processing.py    # PDF extraction and chunking
    │   ├── NB02_vector_database.py        # Embedding, ChromaDB, UMAP
    │   ├── NB03_rag_pipeline.py           # Query router, retrieval, generation
    │   └── NB04_evaluation_mlflow.py      # RAGAS scoring, MLflow logging
    ├── src/
    │   ├── embedder.py                    # build_vectorstore, load_vectorstore, query
    │   └── __init__.py
    ├── scripts/
    │   └── build_chroma_256_local.py      # Rebuild 256-char ChromaDB locally on Mac
    ├── data/
    │   ├── raw/                           # PDFs (not in repo, 200MB+)
    │   └── processed/
    │       ├── chunks_c512_o100.json      # 27,813 chunks (512-char, 100-char overlap)
    │       └── chunks_c256_o50.json       # 51,674 chunks (256-char, 50-char overlap)
    ├── eval/
    │   ├── val_questions.json             # V01-V15 validation questions + expected answers
    │   └── test_questions.json            # T01-T10 test questions (run once in NB04)
    └── results/
        └── figures/
            └── umap_embeddings.png

---

## Quickstart

    git clone https://github.com/suparuek2405/thai-bank-rag-qa
    cd thai-bank-rag-qa
    pip install -r requirements.txt

Place bank PDFs in `data/raw/` as `{BANK}_56-1_2025.pdf`, then run NB01 through NB04 in order in Google Colab. Each notebook is a single `.py` file; copy cell blocks into separate Colab cells.

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.12 | Runtime (Google Colab) |
| PyMuPDF | latest | PDF text extraction |
| sentence-transformers | latest | all-MiniLM-L6-v2 embeddings (384-d, local) |
| ChromaDB | latest | Vector store with metadata filtering |
| LangChain | 0.3 | RAG orchestration |
| Gemini 2.5 Flash | API | Generation LLM and RAGAS judge |
| RAGAS | 0.4.x | Faithfulness, Context Precision, Context Recall |
| MLflow | latest | Experiment tracking (SQLite backend) |

---

## Dataset

10 FY2025 56-1 One Reports (Thai SEC annual registration statements) from BBL, KBANK, KTB, SCBX, TTB, TISCO, BAY, LHFG, CREDIT, and KKP. Total: 3,282 pages, 9.5 million characters, 22% cross-bank questions in the eval set. PDFs are not stored in this repo (200MB+). Download from each bank's investor relations page.

---

## What's Missing and What Would Improve It

**Section-level metadata.** Every chunk currently carries `bank_name` and `page_number`, but nothing about where in the report it came from. A chunk from the NPL section and a chunk from the governance section look identical to the retriever. Tagging chunks with a `section` field (financial highlights, capital adequacy, risk factors, etc.) would let queries filter by topic and would likely raise context precision above 0.30.

**Guardrails.** The system prompt instructs the model to stay grounded in retrieved context, but there is no explicit refusal layer. A question outside the scope of the annual reports (e.g., stock price predictions) currently gets a generated response rather than a graceful decline. Adding a simple topic classifier or a confidence threshold on retrieval distance would close this gap.

**Conversational memory.** The pipeline answers one question at a time with no session history. A follow-up like "how does that compare to last year?" has no context to resolve "that." Adding a short conversation buffer would turn the Q&A system into an actual chatbot.

**Production serving.** The pipeline runs in Colab and has no API or UI layer. Wrapping the retrieval and generation logic in a FastAPI endpoint and adding a simple front-end would make it deployable and interactive.

---

## References

- Es, S. et al. (2023). **RAGAS: Automated Evaluation of Retrieval Augmented Generation.** arXiv. https://arxiv.org/abs/2309.15217
- Wang, W. et al. (2020). **MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.** NeurIPS 2020. https://arxiv.org/abs/2002.10957
- Lewis, P. et al. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** NeurIPS 2020. https://arxiv.org/abs/2005.11401

---

## Author

**Suparuek Wattananupan**
Data Scientist · Banking · Bangkok, Thailand

Specializing in wealth analytics, deep learning, and financial ML.

[![GitHub](https://img.shields.io/badge/GitHub-suparuek2405-black?logo=github)](https://github.com/suparuek2405)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Suparuek%20Wattananupan-blue?logo=linkedin)](https://www.linkedin.com/in/suparuek-wattananupan-7509aa181/)
