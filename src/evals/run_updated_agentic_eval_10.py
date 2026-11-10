"""
=============================================================================
Small Hydropower Plant (SHP) - Updated Agentic RAG Evaluation Runner (10 Cases)
=============================================================================
Module: src.evals.run_updated_agentic_eval_10
Purpose: Executes the freshly updated Agentic RAG pipeline (with structure-aware
         guardrail heuristics, typo-tolerant BM25, and dynamic fallback cycle labels)
         across the first 10 golden test cases (GD-01 to GD-10).
         Saves incrementally after each test case to support safe interruption & resume.
=============================================================================
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
from loguru import logger

from src.agents.main_agent import build_main_agent_graph

load_dotenv()

DEFAULT_DATASET_PATH = Path("src/evals/shp_agentic_rag_golden_dataset.json")
DEFAULT_OUTPUT_PATH = Path("src/evals/updated_agentic_answers_and_chunks.json")

# Logger setup
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/run_updated_agentic_eval_10.log",
    rotation="20 MB",
    retention="14 days",
    level="DEBUG",
    encoding="utf-8",
)


def load_test_cases(dataset_path: Path, limit: int = 10) -> list[dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("test_cases", [])
    if limit > 0:
        cases = cases[:limit]
    return [
        {
            "id": c.get("id"),
            "category": c.get("category", "Single-Hop Policy & Licensing"),
            "query": c.get("input", "").strip(),
            "expected_output": c.get("expected_output", "").strip(),
            "retrieval_context": c.get("retrieval_context", []),
        }
        for c in cases
    ]


def run_agentic_benchmark(
    limit: int = 10,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    resume: bool = True,
) -> None:
    test_cases = load_test_cases(dataset_path, limit=limit)
    logger.info(f"Loaded {len(test_cases)} test cases for updated Agentic RAG evaluation.")

    results: list[dict[str, Any]] = []
    completed_ids: set[str] = set()

    if resume and output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                for r in prev_data.get("results", []):
                    if r.get("actual_output"):
                        results.append(r)
                        completed_ids.add(r["id"])
            if completed_ids:
                logger.info(f"Loaded {len(completed_ids)} completed results from {output_path}; resuming remaining.")
        except Exception as e:
            logger.warning(f"Could not load previous output for resume ({e}); starting fresh.")
            results = []
            completed_ids = set()

    total_time = sum(r.get("latency_seconds", 0.0) for r in results)

    logger.info("Compiling updated Agentic RAG StateGraph...")
    graph = build_main_agent_graph()

    for idx, case in enumerate(test_cases, 1):
        cid = case["id"]
        if cid in completed_ids:
            logger.debug(f"[{idx}/{len(test_cases)}] Skipping already completed {cid}")
            continue

        query = case["query"]
        category = case["category"]
        logger.info(f"[{idx}/{len(test_cases)}] Executing Agentic RAG for {cid} ({category}): '{query[:65]}...'")

        t0 = time.perf_counter()
        try:
            thread_id = f"eval_updated_{cid}_{int(time.time())}"
            final_state = graph.invoke(
                {"query": query},
                config={"configurable": {"thread_id": thread_id}},
            )
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            final_response = final_state.get("final_response", "")
            documents = final_state.get("documents", []) or final_state.get("all_retrieved_chunks", [])
            is_grounded = final_state.get("is_grounded")
            circuit_breaker = final_state.get("circuit_breaker_tripped", False)
            cycle_count = final_state.get("cycle_count", 0)
            gen_metrics = final_state.get("generation_metrics", {})

            logger.success(
                f"[{cid}] Completed in {elapsed:.2f}s | Chunks: {len(documents)} | Grounded: {is_grounded} | "
                f"Circuit Breaker: {circuit_breaker} | Response len: {len(final_response)}"
            )

            result_entry = {
                "id": cid,
                "category": category,
                "query": query,
                "expected_output": case.get("expected_output", ""),
                "actual_output": final_response,
                "retrieved_chunks": documents,
                "total_chunks_retrieved": len(documents),
                "latency_seconds": round(elapsed, 2),
                "is_grounded": is_grounded,
                "circuit_breaker_tripped": circuit_breaker,
                "cycle_count": cycle_count,
                "generation_metrics": gen_metrics,
                "architecture": "Updated Multi-Agent Hierarchical RAG (Structure-Aware Guardrail + Typo-Tolerant BM25)",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            results.append(result_entry)
            completed_ids.add(cid)

            # Checkpoint progress after every query
            output_payload = {
                "metadata": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "architecture": "Updated Multi-Agent Hierarchical RAG (Structure-Aware Guardrail + Typo-Tolerant BM25)",
                    "total_cases_evaluated": len(results),
                    "target_cases": len(test_cases),
                    "average_latency_seconds": round(total_time / len(results), 2) if results else 0,
                },
                "results": results,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_payload, f, indent=2, ensure_ascii=False)

        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error(f"[{cid}] Failed after {elapsed:.2f}s: {e}")
            results.append({
                "id": cid,
                "query": query,
                "error": str(e),
                "latency_seconds": round(elapsed, 2),
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run updated Agentic RAG benchmark on 10 cases.")
    parser.add_argument("--limit", type=int, default=10, help="Number of test cases to run (default: 10)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh without resuming")
    args = parser.parse_args()

    run_agentic_benchmark(limit=args.limit, resume=not args.no_resume)

