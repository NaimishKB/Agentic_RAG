"""
=============================================================================
Small Hydropower Plant (SHP) - Plain-RAG Baseline Experiment Runner
=============================================================================
Module: src.evals.run_plain_rag_baseline
Purpose: Implements a traditional, non-agentic "Plain-RAG" baseline pipeline:
           User Query -> Single Vector/Hybrid Retrieval (Top-K) -> Single LLM Call -> Output
         Compares directly against the Multi-Agent Hierarchical RAG architecture
         (with Planner, Orchestrator scatter-gather, CRAG Grader, Reflection loops,
         and Post-Gen Guardrail).

Key Metrics Evaluated in Head-to-Head Comparison:
1. End-to-end Latency (seconds)
2. Number of LLM inferences per query (1 for Plain-RAG vs 3-7 for Agentic RAG)
3. Retrieval Recall & Precision
4. Faithfulness & Hallucination Rate
5. Fallback & Circuit Breaker Behavior
=============================================================================
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import requests
from dotenv import load_dotenv
from loguru import logger

from src.ingestion.bm25_retriever import get_bm25_retriever, reciprocal_rank_fusion
from src.ingestion.embedder import OllamaEmbeddingClient
from src.ingestion.indexer import QdrantIndexManager
from src.states.orchestratoragent_state import RetrievedChunk

load_dotenv()

# ---------------------------------------------------------------------------
# Default Configurations
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:9b")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_INPUT_FILE = Path("src/evals/generated_answers_and_chunks.json")
DEFAULT_OUTPUT_FILE = Path("src/evals/plain_rag_answers_and_chunks.json")

# Logger setup
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

PLAIN_RAG_SYSTEM_PROMPT = """You are an expert Small Hydropower Plant (SHP) engineering consultant.
Answer the user's question accurately, authoritatively, and concisely based strictly on the provided technical documentation.

Directives:
1. Ground every claim, formula, and numerical threshold strictly in the provided Context.
2. Cite the source document filename in brackets where appropriate (e.g. [Source: filename.pdf]).
3. If the context does not contain sufficient data to answer the question, state clearly: "The provided engineering documentation does not contain sufficient data to answer this query."
4. Do not invent formulas, empirical constants, or policy thresholds.
"""


class PlainRAGEngine:
    """
    Standard, non-agentic single-shot RAG pipeline for baseline benchmarking.
    """

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model_name: str = DEFAULT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        retriever_mode: str = "hybrid",
        top_k: int = 6,
    ):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.retriever_mode = retriever_mode
        self.top_k = top_k

        # Initialize dense retriever
        self.embedder = OllamaEmbeddingClient(
            base_url=ollama_url,
            model_name=embed_model,
        )
        self.qdrant = QdrantIndexManager(
            url=qdrant_url,
            prefer_local=True,
        )

        # Initialize BM25 sparse retriever if hybrid/bm25 mode
        if retriever_mode in ("hybrid", "bm25"):
            self.bm25 = get_bm25_retriever()
        else:
            self.bm25 = None

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Performs single-shot retrieval for the given query."""
        dense_chunks: list[RetrievedChunk] = []

        if self.retriever_mode in ("dense", "hybrid"):
            try:
                vecs = self.embedder.embed_batch([query])
                if vecs and vecs[0]:
                    scored = self.qdrant.search(
                        query_vector=vecs[0],
                        top_k=self.top_k,
                        score_threshold=0.35,
                    )
                    for pt in scored:
                        payload = pt.payload or {}
                        dense_chunks.append(
                            RetrievedChunk(
                                chunk_id=str(payload.get("chunk_id", pt.id)),
                                content=str(payload.get("content", "")),
                                score=float(pt.score),
                                source_file=str(payload.get("source_file", "unknown")),
                                file_type=str(payload.get("file_type", "TXT")),
                                page_or_slide=payload.get("page_or_slide"),
                                breadcrumb=str(payload.get("breadcrumb", "")),
                                is_table=bool(payload.get("is_table", False)),
                                token_count=payload.get("token_count"),
                                worker_source="dense_retriever",
                            )
                        )
            except Exception as e:
                logger.warning(f"[PlainRAG] Dense retrieval failed: {e}")

        if self.retriever_mode == "dense":
            return dense_chunks[: self.top_k]

        # BM25 Retrieval
        bm25_chunks: list[RetrievedChunk] = []
        if self.bm25:
            try:
                bm25_chunks = self.bm25.search(query=query, top_k=self.top_k)
            except Exception as e:
                logger.warning(f"[PlainRAG] BM25 retrieval failed: {e}")

        if self.retriever_mode == "bm25":
            return bm25_chunks[: self.top_k]

        # Hybrid Fusion via RRF
        if not dense_chunks:
            return bm25_chunks[: self.top_k]
        if not bm25_chunks:
            return dense_chunks[: self.top_k]

        fused = reciprocal_rank_fusion(
            dense_chunks=dense_chunks,
            bm25_chunks=bm25_chunks,
            top_k=self.top_k,
        )
        return fused

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, float]:
        """Executes a single LLM inference call with retrieved context."""
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(
                f"--- Document Chunk [{i}] (Source: {c.source_file}) ---\n"
                f"Section: {c.breadcrumb}\n{c.content}"
            )
        context_str = "\n\n".join(context_parts)

        user_prompt = f"### Verified Engineering Context:\n{context_str}\n\n### User Query:\n{query}\n\n### Answer:"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": PLAIN_RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": 16384,
                "num_predict": 8192,
            },
        }

        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        duration = time.perf_counter() - t0

        data = resp.json()
        msg_dict = data.get("message", {})
        raw_answer = msg_dict.get("content", "").strip()

        # If content is empty but model emitted thinking, check thinking block
        if not raw_answer and msg_dict.get("thinking"):
            thinking_text = msg_dict.get("thinking", "").strip()
            h2_idx = thinking_text.find("##")
            if h2_idx != -1:
                raw_answer = thinking_text[h2_idx:].strip()
            else:
                raw_answer = thinking_text

        # Strip internal thinking tags if any
        cleaned_answer = re.sub(r"<think>[\s\S]*?</think>", "", raw_answer).strip()
        if not cleaned_answer and raw_answer:
            cleaned_answer = raw_answer
        return cleaned_answer, duration

    def query(self, query: str) -> dict[str, Any]:
        """Runs the complete Plain-RAG pipeline on a single query."""
        t0 = time.perf_counter()
        chunks = self.retrieve(query)
        answer, gen_latency = self.generate(query, chunks)
        total_latency = time.perf_counter() - t0

        return {
            "answer": answer,
            "chunks": [c.model_dump() for c in chunks],
            "chunks_count": len(chunks),
            "generation_latency": round(gen_latency, 2),
            "total_latency": round(total_latency, 2),
        }


def run_baseline_benchmark(
    input_file: Path = DEFAULT_INPUT_FILE,
    output_file: Path = DEFAULT_OUTPUT_FILE,
    limit: int | None = None,
    retriever_mode: str = "hybrid",
) -> None:
    """Runs Plain-RAG on the golden dataset and outputs structured benchmark results."""
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        agentic_data = json.load(f)

    test_cases = agentic_data.get("results", [])
    if limit:
        test_cases = test_cases[:limit]

    logger.info(
        f"Starting Plain-RAG baseline run on {len(test_cases)} cases | Mode: {retriever_mode}"
    )

    engine = PlainRAGEngine(retriever_mode=retriever_mode)

    results: list[dict[str, Any]] = []
    completed_ids: set[str] = set()

    # Load existing results to support graceful resumption
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                for r in prev_data.get("results", []):
                    if r.get("actual_output"):
                        results.append(r)
                        completed_ids.add(r["id"])
            if completed_ids:
                logger.info(f"Loaded {len(completed_ids)} existing results from {output_file}; will resume remaining.")
        except Exception as e:
            logger.warning(f"Could not load previous output for resume ({e}); starting fresh.")
            results = []
            completed_ids = set()

    total_time = sum(r.get("latency_seconds", 0.0) for r in results)

    for idx, case in enumerate(test_cases, 1):
        cid = case.get("id", f"case_{idx}")
        if cid in completed_ids:
            logger.debug(f"[{idx}/{len(test_cases)}] Skipping already completed {cid}")
            continue

        query = case.get("query", "")
        category = case.get("category", "General")
        logger.info(f"[{idx}/{len(test_cases)}] Executing Plain-RAG for {cid} ({category}): '{query[:60]}...'")

        try:
            out = engine.query(query)
            logger.success(
                f"[{cid}] Completed in {out['total_latency']}s | Retrieved {out['chunks_count']} chunks"
            )

            result_entry = {
                "id": cid,
                "category": category,
                "query": query,
                "expected_output": case.get("expected_output", ""),
                "actual_output": out["answer"],
                "retrieved_chunks": out["chunks"],
                "total_chunks_retrieved": out["chunks_count"],
                "latency_seconds": out["total_latency"],
                "generation_latency_seconds": out["generation_latency"],
                "architecture": "Plain-RAG (Single-Shot Direct Retrieval + Generation)",
                "agentic_comparison": {
                    "agentic_circuit_breaker_tripped": case.get("circuit_breaker_tripped", False),
                    "agentic_latency_seconds": case.get("latency_seconds", 0.0),
                },
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            results.append(result_entry)
            completed_ids.add(cid)
            total_time += out["total_latency"]

            # Save progress after every test case so progress is never lost
            output_payload = {
                "metadata": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "architecture": "Plain-RAG Baseline (No Planner, No CRAG, No Feedback Loops)",
                    "total_cases_evaluated": len(results),
                    "retriever_mode": retriever_mode,
                    "average_latency_seconds": round(total_time / len(results), 2) if results else 0,
                },
                "results": results,
            }
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_payload, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"[{cid}] Failed: {e}")
            results.append({
                "id": cid,
                "query": query,
                "error": str(e),
            })

    output_payload = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "architecture": "Plain-RAG Baseline (No Planner, No CRAG, No Feedback Loops)",
            "total_cases_evaluated": len(results),
            "retriever_mode": retriever_mode,
            "average_latency_seconds": round(total_time / len(results), 2) if results else 0,
        },
        "results": results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    logger.success(
        f"Plain-RAG baseline run complete! Saved {len(results)} results to {output_file}"
    )


def main():
    parser = argparse.ArgumentParser(description="Plain-RAG Baseline Experiment Runner")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit number of cases to run")
    parser.add_argument("--retriever", "-r", type=str, default="hybrid", choices=["hybrid", "dense", "bm25"], help="Retriever mode")
    parser.add_argument("--input", "-i", type=str, default=str(DEFAULT_INPUT_FILE), help="Input golden dataset JSON")
    parser.add_argument("--output", "-o", type=str, default=str(DEFAULT_OUTPUT_FILE), help="Output JSON path")
    args = parser.parse_args()

    run_baseline_benchmark(
        input_file=Path(args.input),
        output_file=Path(args.output),
        limit=args.limit,
        retriever_mode=args.retriever,
    )


if __name__ == "__main__":
    main()
