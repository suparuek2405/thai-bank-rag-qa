# Thai Bank Financial Q&A

Answering natural language questions about Thai commercial banks using their FY2025 annual filings, retrieval-augmented generation, and a 51,674-chunk vector database.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--db-orange)
![RAGAS](https://img.shields.io/badge/RAGAS-0.4.x-purple)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)
![Gemini](https://img.shields.io/badge/Gemini-2.5--Flash-red?logo=google)

---

## Project Summary

I built a RAG pipeline over 10 Thai banks' 56-1 One Reports (FY2025), covering 3,282 pages and 9.5 million characters. The system routes questions to either a bank-filtered or cross-bank retrieval path, then passes the retrieved context to Gemini 2.5 Flash for generation. Four notebooks cover document processing, vector indexing, the RAG pipeline, and RAGAS evaluation tracked with MLflow. The best configuration (512-char chunks, top-10 retrieval) scored 0.74 faithfulness on the validation set and 0.49 on a held-out test set of 10 questions run once at the end.

---

## Results

### Validation Set (15 questions, judge: Gemini 2.5 Flash)

| Config | Faithfulness | Context Precision | Context Recall | Mean |
|--------|-------------|-------------------|----------------|------|
| **chunk=512, top_k=10 (baseline)** | **0.7357** | **0.2958** | **0.3333** | **0.4549** |
| chunk=512, top_k=20 | 0.6083 | 0.3124 | 0.3333 | 0.4180 |
| chunk=256, top_k=10 | 0.5394 | 0.1824 | 0.3333 | 0.3517 |

### Test Set (10 questions, best config only, run once)

| Faithfulness | Context Precision | Context Recall |
|-------------|-------------------|----------------|
| 0.4935 | 0.0861 | 0.1000 |

The drop from val to test is expected: the test questions cover harder cross-bank comparisons and trend questions that require information spread across multiple chunks. Faithfulness holds near 0.49, meaning the model mostly stays grounded even when retrieval is incomplete.

---

## The Story

### Chapter 1 -- Document Processing (NB01)

I loaded 10 FY2025 56-1 One Reports using PyMuPDF and extracted 9.5 million characters from 3,282 pages. The pipeline split text into overlapping chunks with three size configurations (256, 512, 1024 chars) to test which best preserves financial table context. Financial ratio tables (NPL, NIM, ROE, CAR) come out of PyMuPDF as plain left-to-right text, so chunk boundaries that cut through a table row lose the metric's label or value entirely.

| Bank | Pages | Chars | Chunks (512-char) |
|------|-------|-------|--------------------|
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

### Chapter 2 -- Embeddings Cluster by Topic, Not by Bank (NB02)

All 27,813 chunks were embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-d, local, no API cost) and stored in ChromaDB with bank-level metadata. A UMAP projection of 2,000 sampled embeddings showed something useful: chunks cluster by financial topic across banks, not by which bank they came from. NPL chunks from KTB and BBL land near each other; NIM chunks from all 10 banks form their own neighborhood.

![UMAP of chunk embeddings colored by bank](https://raw.githubusercontent.com/natsuparuek/thai-bank-rag-qa/main/results/figures/umap_embeddings.png)

This means cross-bank retrieval can work purely on embedding similarity, but single-bank queries need a metadata filter on `bank_name`. Without it, a question about KTB's NIM will pull the top chunks from whichever bank has the most similar text, which may not be KTB at all.

### Chapter 3 -- Routing Matters as Much as Retrieval (NB03)

The pipeline classifies each question as single-bank or cross-bank before querying ChromaDB. Single-bank queries apply a `where: {bank_name: X}` filter; cross-bank queries search the full collection. Both paths feed the retrieved chunks into a financial system prompt and then into Gemini 2.5 Flash for generation.

Getting the router right turned out to be as important as the chunk size. A cross-bank question routed as single-bank retrieves only one bank's data and produces a factually wrong comparison. The 15 validation questions and 10 test questions were written before building the pipeline to avoid tuning the router toward the evaluation set.

### Chapter 4 -- 512-char Chunks Win; More Context Hurts Faithfulness (NB04)

RAGAS evaluated three configurations using three LLM-only metrics: Faithfulness (is the answer grounded in the retrieved context?), Context Precision (are the retrieved chunks actually useful?), and Context Recall (did retrieval surface what was needed?). A separate Gemini 2.5 Flash instance served as the judge to keep evaluation independent of the generation LLM.

The baseline (chunk=512, top_k=10) scored best on faithfulness at 0.74. Doubling the retrieval to top_k=20 slightly improved context precision but dropped faithfulness to 0.61, suggesting that extra chunks introduce noise the model can't ignore. The 256-char chunks performed worst overall: smaller chunks do reduce topic dilution in embeddings, but they lose enough surrounding context that the LLM has less to work with per chunk. All three configurations scored identically on context recall (0.33), pointing to a ceiling in how much of the required information top-10 retrieval can surface given the question design. Experiments are tracked in MLflow with per-run parameters and metrics saved to `mlflow.db` on Google Drive.

---

## Key Decisions and Lessons

**Setting aside test questions first.** Writing 25 questions before touching the pipeline forced a cleaner val/test split. The 15 validation questions were used freely to tune chunk size, top-k, and the system prompt. The 10 test questions were run exactly once at the end. The gap between val faithfulness (0.74) and test faithfulness (0.49) reflects how much tuning happened on the val set, which is exactly what this design is supposed to reveal.

**Dropping AnswerRelevancy from RAGAS.** The fourth standard RAGAS metric (Answer Relevancy) requires an embedding model for scoring. On the free Colab environment it fell back to an OpenAI embeddings call with no API key, crashing the evaluation. The three remaining metrics are all LLM-only and cover the core failure modes: hallucination (faithfulness), retrieval noise (context precision), and retrieval completeness (context recall). For a portfolio context this is sufficient.

**256-char chunks are not always better.** The UMAP visualization from NB02 suggested that 512-char embeddings have topic dilution issues, so 256-char chunks looked promising. In practice they scored worse. Financial tables in these reports contain multiple related metrics in a short span of text, and splitting them more aggressively loses the relationship between a metric name and its value. The embedding dilution problem is real, but the context loss is worse.

**Free-tier Gemini concurrency.** Running 45 RAGAS evaluation jobs with the default concurrency caused all jobs to stall silently at 0% progress. Setting `RunConfig(max_workers=2, timeout=120)` fixed this. Each val config took 10-16 minutes to score; the test set took about 6 minutes. This is a real constraint to plan around if you want fast iteration on evaluation.

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
    │   └── build_chroma_256_local.py      # Rebuild 256-char ChromaDB locally
    ├── data/
    │   ├── raw/                           # PDFs (not in repo, 200MB+)
    │   └── processed/
    │       ├── chunks_c512_o100.json      # 27,813 chunks (512-char, 100-char overlap)
    │       └── chunks_c256_o50.json       # 51,674 chunks (256-char, 50-char overlap)
    ├── eval/
    │   ├── val_questions.json             # V01-V15 validation questions + expected answers
    │   └── test_questions.json            # T01-T10 test questions (run once in NB04)
    ├── results/
    │   └── figures/
    │       └── umap_embeddings.png
    └── mlflow.db                          # MLflow experiment store (on Drive, not in repo)

---

## Quickstart

```bash
git clone https://github.com/natsuparuek/thai-bank-rag-qa
cd thai-bank-rag-qa
pip install -r requirements.txt
```

Place bank PDFs in `data/raw/` following the naming convention `{BANK}_56-1_2025.pdf`, then run the notebooks in order (NB01 through NB04) in Google Colab. Each notebook is a single `.py` file; copy cell blocks into Colab cells.

---

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12 | Runtime (Google Colab) |
| PyMuPDF | latest | PDF text extraction |
| sentence-transformers | latest | all-MiniLM-L6-v2 embeddings (local) |
| ChromaDB | latest | Vector store with metadata filtering |
| LangChain | 0.3 | RAG orchestration |
| Gemini 2.5 Flash | API | Generation and RAGAS judge LLM |
| RAGAS | 0.4.x | Faithfulness, Context Precision, Context Recall |
| MLflow | latest | Experiment tracking (SQLite backend) |

---

## Dataset

10 FY2025 56-1 One Reports (Thai SEC annual registration statements) from BBL, KBANK, KTB, SCBX, TTB, TISCO, BAY, LHFG, CREDIT, and KKP. Total: 3,282 pages, 9.5 million characters. PDFs are not stored in this repo (200MB+). Download from each bank's investor relations page.

---

## Author

**Suparuek Wattananupan**
Data Scientist · Bangkok, Thailand

Interested in NLP, financial data, and applied ML for the banking sector.

[![GitHub](https://img.shields.io/badge/GitHub-natsuparuek-black?logo=github)](https://github.com/natsuparuek/thai-bank-rag-qa)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Suparuek-blue?logo=linkedin)](https://www.linkedin.com/in/suparuek-wattananupan-7509aa181/)
