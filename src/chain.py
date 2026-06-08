"""
chain.py — RAG pipeline for Thai Bank Financial Q&A system
===========================================================
Connects ChromaDB retrieval with Gemini 1.5 Flash via LangChain.
Supports two query modes:
  - single_bank: metadata filter on bank_name, precise ratio lookups
  - cross_bank:  no filter, searches all banks for comparison questions

Usage (in Colab):
    from src.chain import build_chain, ask
"""

import os
import re
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.embedder import query as retriever_query


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_SINGLE = """You are a financial analyst assistant specializing in Thai commercial banks.
You answer questions using ONLY the provided context excerpts from 56-1 One Report annual filings.

Rules:
1. Answer with specific numbers and percentages when available.
2. Always cite the source: bank name and page number (e.g. "Source: KBANK p.38").
3. If the context does not contain the answer, say exactly: "The information was not found in the retrieved context."
4. Do not use knowledge outside the provided context.
5. Keep answers concise — 2-4 sentences maximum.

Context format: each excerpt is labeled [Bank | Page X].
"""

SYSTEM_PROMPT_CROSS = """You are a financial analyst assistant specializing in Thai commercial banks.
You answer questions by comparing data across multiple banks using ONLY the provided context excerpts from 56-1 One Report annual filings.

Rules:
1. Read ALL context excerpts carefully before answering — the correct bank may not be the first one listed.
2. When the question asks for the highest/lowest/best, explicitly compare all values mentioned in the context.
3. Always cite which bank your answer refers to and the page number.
4. If the context does not contain enough data to compare all banks, state which banks are missing.
5. If the answer cannot be determined from context, say: "The information was not found in the retrieved context."
6. Keep answers concise — 3-5 sentences maximum.

Context format: each excerpt is labeled [Bank | Page X].
"""


# ---------------------------------------------------------------------------
# Format retrieved chunks into a prompt context block
# ---------------------------------------------------------------------------

def format_context(results: list[dict]) -> str:
    """
    Convert retrieval results into a labeled context block for the LLM prompt.

    Each chunk is labeled with bank name and page number so the model
    can cite its sources.

    Example output:
        [KBANK | Page 4]
        Net interest margin (NIM)
        3.23%
        3.60%
        ...

        [KBANK | Page 38]
        KBank's consolidated net interest income for 2025 was Baht 137,152 million...
    """
    blocks = []
    for r in results:
        header = f"[{r['bank_name']} | Page {r['page']}]"
        blocks.append(f"{header}\n{r['text'].strip()}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Build LLM
# ---------------------------------------------------------------------------

def build_llm(api_key: str, model: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    """
    Initialise Gemini 2.5 Flash via LangChain.

    Args:
        api_key: Google AI Studio API key
        model:   Gemini model name

    Returns:
        LangChain ChatGoogleGenerativeAI instance
    """
    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,        # deterministic — financial Q&A needs consistency
        max_output_tokens=1024
    )


# ---------------------------------------------------------------------------
# Core ask function
# ---------------------------------------------------------------------------

def ask(
    question: str,
    collection,
    llm,
    bank_name: Optional[str] = None,
    top_k: int = 10,
    return_context: bool = False
) -> dict:
    """
    Full RAG pipeline: retrieve → format → generate.

    Args:
        question:       Natural language question
        collection:     ChromaDB collection from embedder.load_vectorstore()
        llm:            Gemini LLM from build_llm()
        bank_name:      If set, restrict retrieval to this bank (single-bank mode)
                        If None, search all banks (cross-bank mode)
        top_k:          Number of chunks to retrieve
        return_context: If True, include retrieved context in output

    Returns:
        {
            "question":  str,
            "answer":    str,
            "mode":      "single_bank" | "cross_bank",
            "bank":      str | None,
            "top_k":     int,
            "context":   list[dict]  (only if return_context=True)
        }
    """
    # Step 1: Retrieve relevant chunks
    results = retriever_query(
        collection=collection,
        question=question,
        bank_name=bank_name,
        top_k=top_k
    )

    # Step 2: Format context block
    context = format_context(results)

    # Step 3: Choose system prompt based on mode
    mode = "single_bank" if bank_name else "cross_bank"
    system_prompt = SYSTEM_PROMPT_SINGLE if mode == "single_bank" else SYSTEM_PROMPT_CROSS

    # Step 4: Build messages and call LLM
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]

    response = llm.invoke(messages)
    answer = response.content.strip()

    output = {
        "question": question,
        "answer":   answer,
        "mode":     mode,
        "bank":     bank_name,
        "top_k":    top_k,
    }

    if return_context:
        output["context"] = results

    return output


# ---------------------------------------------------------------------------
# Batch run for evaluation
# ---------------------------------------------------------------------------

def run_eval_questions(
    questions: list[dict],
    collection,
    llm,
    top_k: int = 10
) -> list[dict]:
    """
    Run the RAG pipeline on a list of eval question dicts.

    Args:
        questions:  List of question dicts from eval_questions.json
                    Each must have: id, question, banks, expected_answer
        collection: ChromaDB collection
        llm:        Gemini LLM
        top_k:      Chunks to retrieve per question

    Returns:
        List of result dicts with question, expected_answer, answer, mode
    """
    results = []

    for q in questions:
        # Single-bank mode if only one bank listed
        bank = q["banks"][0] if len(q["banks"]) == 1 else None

        result = ask(
            question=q["question"],
            collection=collection,
            llm=llm,
            bank_name=bank,
            top_k=top_k,
            return_context=True
        )

        results.append({
            "id":              q["id"],
            "type":            q["type"],
            "question":        q["question"],
            "expected_answer": q["expected_answer"],
            "answer":          result["answer"],
            "mode":            result["mode"],
            "contexts":        [r["text"] for r in result["context"]]
        })

        print(f"  [{q['id']}] done")

    return results
