# =============================================================================
# NB04 — Evaluation + MLflow
# Thai Bank Financial Q&A System
# =============================================================================
# Copy each cell block into a separate Colab cell.
# Run in order. Prerequisite: NB03 must have run and saved val_results_top10.json.
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies
# ─────────────────────────────────────────────────────────────────────────────

import os, subprocess, sys, types

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "google-genai", "mlflow", "langchain-google-genai",
    "langchain-google-vertexai",   # provides ChatVertexAI/VertexAI for the patch below
    "chromadb", "sentence-transformers",
    "pymupdf",                     # proper fitz — overrides Colab's broken fitz stub
    "jsonref",                     # required by ragas/instructor for schema resolution
    "-q"
])

os.environ["TQDM_NOTEBOOK"] = "0"   # plain text progress bars — prevents widget metadata errors

# ── Compatibility patch ──────────────────────────────────────────────────────
# Colab's ragas references two symbols removed from langchain-community ≥ 0.3:
#   ragas/llms/base.py line 12: from langchain_community.chat_models.vertexai import ChatVertexAI
#   ragas/llms/base.py line 13: from langchain_community.llms import VertexAI
# These were moved to langchain-google-vertexai. We inject stubs so the import
# resolves without error — no runtime restart needed.
from langchain_google_vertexai import ChatVertexAI as _CV, VertexAI as _VV

_vtx_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_vtx_stub.ChatVertexAI = _CV
sys.modules["langchain_community.chat_models.vertexai"] = _vtx_stub

import langchain_community.llms as _lc_llms
if not hasattr(_lc_llms, "VertexAI"):
    _lc_llms.VertexAI = _VV
# ────────────────────────────────────────────────────────────────────────────

import ragas, mlflow
print(f"RAGAS:  {ragas.__version__}  ✓")
print(f"MLflow: {mlflow.__version__}  ✓")
print("All packages ready — continue to Cell 2.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Mount Drive, clone/pull repo, restore ChromaDB (512-char baseline)
# ─────────────────────────────────────────────────────────────────────────────

from google.colab import drive
drive.mount('/content/drive')

import sys, shutil, subprocess, json

DRIVE_ROOT     = "/content/drive/MyDrive/Github experiment/thai-bank-rag-qa"
PROCESSED_DIR  = f"{DRIVE_ROOT}/data/processed"
RESULTS_DIR    = f"{DRIVE_ROOT}/results"
CHROMA_512_DIR = "/content/chroma_db_512"
CHROMA_256_DIR = "/content/chroma_db_256"

REPO_DIR = "/content/thai-bank-rag-qa"
if not os.path.exists(REPO_DIR):
    get_ipython().system("git clone https://github.com/suparuek2405/thai-bank-rag-qa.git /content/thai-bank-rag-qa")
else:
    subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=REPO_DIR)

sys.path.insert(0, REPO_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Copy 512-char ChromaDB from Drive to local (fast reads)
if not os.path.exists(CHROMA_512_DIR):
    print("Copying 512-char ChromaDB from Drive...")
    shutil.copytree(f"{DRIVE_ROOT}/chroma_db", CHROMA_512_DIR)
    print("Done.")
else:
    print("512-char ChromaDB already local.")

from src.embedder import load_vectorstore, build_vectorstore
collection_512 = load_vectorstore(CHROMA_512_DIR, collection_name="thai_banks")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Load Gemini LLM
# ─────────────────────────────────────────────────────────────────────────────

import getpass
from src.chain import build_llm
from langchain_core.messages import HumanMessage

API_KEY = getpass.getpass("Google AI Studio API Key: ")
llm = build_llm(API_KEY, model="gemini-2.5-flash")

test = llm.invoke([HumanMessage(content="Reply with exactly: 'Ready.'")]).content
print(test)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Load baseline val results (Config 1: chunk=512, top_k=10)
# ─────────────────────────────────────────────────────────────────────────────
# Already computed in NB03 — load from Drive instead of re-running.

with open(f"{RESULTS_DIR}/val_results_top10.json", encoding="utf-8") as f:
    results_c512_k10 = json.load(f)

print(f"Config 1 loaded: {len(results_c512_k10)} val results (chunk=512, top_k=10)")
print(f"Sample answer: {results_c512_k10[0]['answer'][:120]}...")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Run Config 2: chunk=512, top_k=20
# ─────────────────────────────────────────────────────────────────────────────
# Same 512-char chunks, but retrieve more context per question.
# Hypothesis: higher top_k helps cross-bank questions cover all 10 banks.

from src.chain import run_eval_questions

with open(f"{REPO_DIR}/data/eval_questions.json", encoding="utf-8") as f:
    eval_qs = json.load(f)

val_questions = eval_qs["val"]

print("Running Config 2: chunk=512, top_k=20...")
print(f"({len(val_questions)} questions × ~4s sleep = ~{len(val_questions)*4//60}min)\n")

results_c512_k20 = run_eval_questions(
    questions=val_questions,
    collection=collection_512,
    llm=llm,
    top_k=20,
    sleep_secs=4.0
)

save_path = f"{RESULTS_DIR}/val_results_512_top20.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results_c512_k20, f, ensure_ascii=False, indent=2)
print(f"\nSaved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Build or load 256-char ChromaDB
# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis: smaller chunks reduce topic dilution → better NIM/NPL retrieval.
# This takes ~15 min if building from scratch. Saved to Drive after first run.

from src.loader import load_chunks

chroma_256_drive = f"{DRIVE_ROOT}/chroma_db_256"

if os.path.exists(chroma_256_drive):
    print("Loading 256-char ChromaDB from Drive (already built)...")
    if not os.path.exists(CHROMA_256_DIR):
        shutil.copytree(chroma_256_drive, CHROMA_256_DIR)
    print("Done.")
else:
    print("Building 256-char ChromaDB from scratch (~15 min on T4)...")
    chunk_file = os.path.join(PROCESSED_DIR, "chunks_c256_o50.json")
    chunks_256, config_256 = load_chunks(chunk_file)
    print(f"Loaded {len(chunks_256):,} chunks | config: {config_256}")

    collection_256 = build_vectorstore(
        chunks=chunks_256,
        chroma_dir=CHROMA_256_DIR,
        collection_name="thai_banks_256",
        reset=False
    )

    # Persist to Drive for future sessions
    shutil.copytree(CHROMA_256_DIR, chroma_256_drive)
    print(f"Saved 256-char ChromaDB to Drive: {chroma_256_drive}")

collection_256 = load_vectorstore(CHROMA_256_DIR, collection_name="thai_banks_256")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Run Config 3: chunk=256, top_k=10
# ─────────────────────────────────────────────────────────────────────────────

print("Running Config 3: chunk=256, top_k=10...")
print(f"({len(val_questions)} questions × ~4s sleep = ~{len(val_questions)*4//60}min)\n")

results_c256_k10 = run_eval_questions(
    questions=val_questions,
    collection=collection_256,
    llm=llm,
    top_k=10,
    sleep_secs=4.0
)

save_path = f"{RESULTS_DIR}/val_results_256_top10.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results_c256_k10, f, ensure_ascii=False, indent=2)
print(f"\nSaved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Score all configs with RAGAS
# ─────────────────────────────────────────────────────────────────────────────
# RAGAS measures 3 LLM-based dimensions of RAG quality:
#   faithfulness      — is the answer grounded in the retrieved context? (no hallucination)
#   context_precision — are the retrieved chunks actually useful?
#   context_recall    — did we retrieve all chunks needed to answer correctly?
# Note: AnswerRelevancy is excluded — it requires a separate embedding model API call.

from google import genai as google_genai
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall  # noqa: ragas.metrics.collections causes TypeError with evaluate() — intentional
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.run_config import RunConfig

# Create RAGAS evaluator LLM — use gemini-2.0-flash as judge:
# faster than 2.5-flash (no thinking overhead), better concurrent throughput,
# and perfectly capable of grading RAG quality.
google_client = google_genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
ragas_llm = llm_factory("gemini-1.5-flash", provider="google", client=google_client)

# LLM-only metrics — no embedding model required
_metrics = [
    Faithfulness(llm=ragas_llm),
    ContextPrecision(llm=ragas_llm),
    ContextRecall(llm=ragas_llm),
]

# Limit concurrency: too many simultaneous calls stall the Gemini API
_run_config = RunConfig(max_workers=2, timeout=120)


def score_config(results: list[dict], config_name: str) -> dict:
    """
    Run RAGAS evaluation on a list of val results.

    Args:
        results:     Output of run_eval_questions()
        config_name: Label for printing (e.g. "chunk=512, top_k=10")

    Returns:
        Dict of metric name → float score (0-1)
    """
    import threading, time

    print(f"\nScoring: {config_name}")
    print(f"  {len(results)} samples × 3 metrics — expect ~{len(results) // 2 * 40 // 60} min with gemini-2.5-flash")

    # Heartbeat: print elapsed time every 30s so you know it's alive
    _stop = threading.Event()
    def _heartbeat():
        t0 = time.time()
        while not _stop.is_set():
            time.sleep(30)
            if not _stop.is_set():
                print(f"  ... still running ({int(time.time()-t0)}s elapsed)")
    threading.Thread(target=_heartbeat, daemon=True).start()

    # RAGAS 0.4 uses EvaluationDataset + SingleTurnSample
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["expected_answer"],
        )
        for r in results
    ]
    dataset = EvaluationDataset(samples=samples)

    try:
        scores = evaluate(dataset=dataset, metrics=_metrics, run_config=_run_config)
    finally:
        _stop.set()

    def _mean(vals):
        """Average a list of per-sample scores, ignoring None/NaN."""
        clean = [v for v in vals if v is not None and v == v]  # v==v filters NaN
        return round(sum(clean) / len(clean), 4) if clean else 0.0

    result = {
        "faithfulness":      _mean(scores["faithfulness"]),
        "context_precision": _mean(scores["context_precision"]),
        "context_recall":    _mean(scores["context_recall"]),
    }

    print(f"  faithfulness:      {result['faithfulness']:.4f}")
    print(f"  context_precision: {result['context_precision']:.4f}")
    print(f"  context_recall:    {result['context_recall']:.4f}")

    return result


# Score all 3 configs
if "scores_c512_k10" not in dir():
    scores_c512_k10 = score_config(results_c512_k10, "chunk=512, top_k=10  [baseline]")
else:
    print("scores_c512_k10 already computed — skipping")

if "scores_c512_k20" not in dir():
    scores_c512_k20 = score_config(results_c512_k20, "chunk=512, top_k=20")
else:
    print("scores_c512_k20 already computed — skipping")

if "scores_c256_k10" not in dir():
    scores_c256_k10 = score_config(results_c256_k10, "chunk=256, top_k=10")
else:
    print("scores_c256_k10 already computed — skipping")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Log all experiments to MLflow
# ─────────────────────────────────────────────────────────────────────────────
# MLflow tracks every experiment config so you can compare them later.
# Stored in Drive so logs persist across Colab sessions.

MLFLOW_DB  = f"{DRIVE_ROOT}/mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("thai-bank-rag-qa-val")

configs = [
    {
        "run_name":   "chunk512_top10_baseline",
        "chunk_size": 512,
        "overlap":    100,
        "top_k":      10,
        "scores":     scores_c512_k10,
    },
    {
        "run_name":   "chunk512_top20",
        "chunk_size": 512,
        "overlap":    100,
        "top_k":      20,
        "scores":     scores_c512_k20,
    },
    {
        "run_name":   "chunk256_top10",
        "chunk_size": 256,
        "overlap":    50,
        "top_k":      10,
        "scores":     scores_c256_k10,
    },
]

for cfg in configs:
    with mlflow.start_run(run_name=cfg["run_name"]):
        mlflow.log_param("chunk_size",  cfg["chunk_size"])
        mlflow.log_param("overlap",     cfg["overlap"])
        mlflow.log_param("top_k",       cfg["top_k"])
        mlflow.log_param("llm",         "gemini-2.5-flash")
        mlflow.log_param("embed_model", "all-MiniLM-L6-v2")
        mlflow.log_param("prompt_ver",  "v1")

        for metric_name, value in cfg["scores"].items():
            mlflow.log_metric(metric_name, value)

    print(f"Logged: {cfg['run_name']}")

print("\nAll experiments logged to MLflow.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Compare configs and pick best
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd

scorecard = pd.DataFrame([
    {
        "Config":             "chunk=512, top_k=10  [baseline]",
        "Faithfulness":       scores_c512_k10["faithfulness"],
        "Context Precision":  scores_c512_k10["context_precision"],
        "Context Recall":     scores_c512_k10["context_recall"],
        "Mean":               round(sum(scores_c512_k10.values()) / 3, 4),
    },
    {
        "Config":             "chunk=512, top_k=20",
        "Faithfulness":       scores_c512_k20["faithfulness"],
        "Context Precision":  scores_c512_k20["context_precision"],
        "Context Recall":     scores_c512_k20["context_recall"],
        "Mean":               round(sum(scores_c512_k20.values()) / 3, 4),
    },
    {
        "Config":             "chunk=256, top_k=10",
        "Faithfulness":       scores_c256_k10["faithfulness"],
        "Context Precision":  scores_c256_k10["context_precision"],
        "Context Recall":     scores_c256_k10["context_recall"],
        "Mean":               round(sum(scores_c256_k10.values()) / 3, 4),
    },
])

print("Val set RAGAS scores:\n")
print(scorecard.to_string(index=False))

# Pick best config by mean score
best_idx  = scorecard["Mean"].idxmax()
best_cfg  = scorecard.loc[best_idx, "Config"]
print(f"\nBest config: {best_cfg}  (mean={scorecard.loc[best_idx, 'Mean']:.4f})")

# Map best config to collection and top_k for test run
if "256" in best_cfg:
    best_collection = collection_256
    best_top_k      = 10
elif "top_k=20" in best_cfg or "top20" in best_cfg:
    best_collection = collection_512
    best_top_k      = 20
else:
    best_collection = collection_512
    best_top_k      = 10

print(f"Using for test run: collection={'256' if '256' in best_cfg else '512'}-char, top_k={best_top_k}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — Run test set ONCE with best config
# ─────────────────────────────────────────────────────────────────────────────
# TEST SET IS USED ONLY HERE — never for tuning.
# These are the final reported scores for the portfolio.

test_questions = eval_qs["test"]

print(f"Running test set ({len(test_questions)} questions) with best config...")
print("This is the ONE AND ONLY test run.\n")

test_results = run_eval_questions(
    questions=test_questions,
    collection=best_collection,
    llm=llm,
    top_k=best_top_k,
    sleep_secs=4.0
)

save_path = f"{RESULTS_DIR}/test_results_final.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(test_results, f, ensure_ascii=False, indent=2)
print(f"\nSaved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — Score test set + log to MLflow
# ─────────────────────────────────────────────────────────────────────────────

test_scores = score_config(test_results, f"TEST SET — {best_cfg}")

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("thai-bank-rag-qa-test")
with mlflow.start_run(run_name=f"FINAL_TEST_{best_cfg.replace(' ', '_').replace('=', '')}"):
    mlflow.log_param("chunk_size",  256 if "256" in best_cfg else 512)
    mlflow.log_param("top_k",       best_top_k)
    mlflow.log_param("llm",         "gemini-2.5-flash")
    mlflow.log_param("embed_model", "all-MiniLM-L6-v2")
    mlflow.log_param("dataset",     "test_set_T01_T10")

    for metric_name, value in test_scores.items():
        mlflow.log_metric(metric_name, value)

print("Test scores logged to MLflow.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 — Print final scorecard for README
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 62)
print("FINAL SCORECARD — copy into README Chapter 4")
print("=" * 62)
print()
print(f"{'Config':<30} {'Faith':>7} {'Prec':>7} {'Recall':>7} {'Mean':>7}")
print("-" * 62)

val_rows = [
    ("chunk=512, top_k=10 [baseline]", scores_c512_k10),
    ("chunk=512, top_k=20",            scores_c512_k20),
    ("chunk=256, top_k=10",            scores_c256_k10),
]
for label, s in val_rows:
    mean = sum(s.values()) / 3
    print(f"{label:<30} {s['faithfulness']:>7.4f} "
          f"{s['context_precision']:>7.4f} {s['context_recall']:>7.4f} {mean:>7.4f}")

print("-" * 62)
mean_test = sum(test_scores.values()) / 3
print(f"{'TEST SET (best config)':<30} {test_scores['faithfulness']:>7.4f} "
      f"{test_scores['context_precision']:>7.4f} "
      f"{test_scores['context_recall']:>7.4f} {mean_test:>7.4f}")
print("=" * 70)

# Save scorecard as JSON
scorecard_path = f"{RESULTS_DIR}/final_scorecard.json"
with open(scorecard_path, "w", encoding="utf-8") as f:
    json.dump({
        "val": {
            "chunk512_top10":  scores_c512_k10,
            "chunk512_top20":  scores_c512_k20,
            "chunk256_top10":  scores_c256_k10,
        },
        "test": {
            "best_config": best_cfg,
            "scores":      test_scores,
        }
    }, f, indent=2)
print(f"\nScorecard saved → {scorecard_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 14 — Push results + src to GitHub
# ─────────────────────────────────────────────────────────────────────────────

import getpass as gp

TOKEN = gp.getpass("GitHub Personal Access Token: ")
USER  = "suparuek2405"
REPO  = "thai-bank-rag-qa"

get_ipython().system(f"""
    cd {REPO_DIR} && \
    git config user.email "suparuek2405@gmail.com" && \
    git config user.name "Suparuek Wattananupan" && \
    git remote set-url origin https://{USER}:{TOKEN}@github.com/{USER}/{REPO}.git && \
    git pull --rebase origin main && \
    git add src/ data/eval_questions.json README.md && \
    git commit -m "[NB04] RAGAS evaluation complete — final scorecard added" && \
    git push origin main
""")
print("Done! Upload NB04 notebook manually to notebooks/ on GitHub.")
