"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Golden Dataset Batch Answer Generator
=============================================================================
Module: src.evals.generate_dataset_answers
Purpose: Iterates through golden evaluation datasets, sends each engineering query
         to the live FastAPI backend (http://127.0.0.1:8600/api/v1/chat), collects
         the generated answer, full retrieved chunks, citations, routing decisions,
         and telemetry, and saves everything incrementally to a structured JSON file.
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

import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Default Configurations
# ---------------------------------------------------------------------------
DEFAULT_API_URL = os.getenv("API_SERVER_URL", "http://127.0.0.1:8600")
DEFAULT_TIMEOUT_S = int(os.getenv("API_CHAT_TIMEOUT_S", "1800"))
DATASET_SHP_PATH = Path("src/evals/shp_agentic_rag_golden_dataset.json")
DATASET_TURBINES_PATH = Path("src/evals/agentic_rag_shp_golden_dataset.json")
DEFAULT_OUTPUT_PATH = Path("src/evals/generated_answers_and_chunks.json")
DEFAULT_DEEPEVAL_READY_PATH = Path("src/evals/generated_answers_and_chunks_deepeval_ready.json")

# Logger setup
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/generate_dataset_answers.log",
    rotation="20 MB",
    retention="14 days",
    level="DEBUG",
    encoding="utf-8",
)


# ---------------------------------------------------------------------------
# Dataset Loading & Normalization
# ---------------------------------------------------------------------------
def load_and_normalize_cases(dataset_choice: str = "all") -> list[dict[str, Any]]:
    """
    Loads test cases from golden datasets and standardizes keys across both formats.

    Returns list of dicts with keys:
      - id: str
      - dataset_source: str
      - category: str
      - difficulty: str
      - query: str
      - expected_output: str
      - reference_context: list[str] | str
      - source_documents: list[str] | str
    """
    normalized_cases: list[dict[str, Any]] = []

    # 1. SHP Dataset (30 cases)
    if dataset_choice in ("all", "shp") and DATASET_SHP_PATH.exists():
        with open(DATASET_SHP_PATH, "r", encoding="utf-8") as f:
            shp_data = json.load(f)
        cases = shp_data.get("test_cases", [])
        for c in cases:
            normalized_cases.append({
                "id": c.get("id"),
                "dataset_source": DATASET_SHP_PATH.name,
                "category": c.get("category", "Policy & Regulatory"),
                "difficulty": c.get("hop_count", 1),
                "query": c.get("input", "").strip(),
                "expected_output": c.get("expected_output", "").strip(),
                "reference_context": c.get("retrieval_context", []),
                "source_documents": c.get("source_documents", []),
                "evaluation_criteria": c.get("evaluation_criteria", ""),
            })
        logger.info(f"Loaded {len(cases)} test cases from {DATASET_SHP_PATH.name}")

    # 2. Turbines & Formulas Dataset (25 cases)
    if dataset_choice in ("all", "turbines") and DATASET_TURBINES_PATH.exists():
        with open(DATASET_TURBINES_PATH, "r", encoding="utf-8") as f:
            turbines_data = json.load(f)
        for c in turbines_data:
            normalized_cases.append({
                "id": c.get("id"),
                "dataset_source": DATASET_TURBINES_PATH.name,
                "category": c.get("query_type", "Engineering & Formulas"),
                "difficulty": c.get("difficulty", "Medium"),
                "query": c.get("user_input", "").strip(),
                "expected_output": c.get("ground_truth_answer", "").strip(),
                "reference_context": c.get("reference_context_summary", ""),
                "source_documents": c.get("source_documents", []),
                "must_include_keywords": c.get("must_include_keywords", []),
            })
        logger.info(f"Loaded {len(turbines_data)} test cases from {DATASET_TURBINES_PATH.name}")

    return normalized_cases


# ---------------------------------------------------------------------------
# Output File Persistence (Incremental / Atomic)
# ---------------------------------------------------------------------------
def load_existing_results(output_path: Path) -> dict[str, Any]:
    """Loads existing results dict if present to support resuming."""
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "results" in data:
                return data
        except Exception as e:
            logger.warning(f"Could not parse existing output file {output_path}: {e}")
    return {
        "metadata": {
            "title": "SHP Agentic RAG - Generated Answers and Retrieved Chunks",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "api_endpoint": DEFAULT_API_URL,
            "total_test_cases": 0,
            "completed_cases": 0,
        },
        "results": [],
    }


def save_results(output_path: Path, data: dict[str, Any], deepeval_ready_path: Path | None = None) -> None:
    """Saves results incrementally and exports DeepEval-ready format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(output_path)

    # Also write out DeepEval-ready list
    if deepeval_ready_path:
        deepeval_list = []
        for r in data.get("results", []):
            chunks = r.get("retrieved_chunks", [])
            retrieval_contexts = [
                f"[Source: {c.get('source_file')} | {c.get('breadcrumb')}]\n{c.get('content')}"
                for c in chunks
                if c.get("content")
            ]
            deepeval_list.append({
                "id": r.get("id"),
                "category": r.get("category"),
                "input": r.get("query"),
                "actual_output": r.get("actual_output"),
                "expected_output": r.get("expected_output"),
                "retrieval_context": retrieval_contexts,
                "context": retrieval_contexts,
                "citations": [c.get("source_file") for c in r.get("citations", [])],
                "routing_decision": r.get("routing_decision"),
                "circuit_breaker_tripped": r.get("circuit_breaker_tripped", False),
                "latency_seconds": r.get("latency_seconds", 0.0),
            })
        tmp_de = deepeval_ready_path.with_suffix(".tmp")
        with open(tmp_de, "w", encoding="utf-8") as f:
            json.dump(deepeval_list, f, indent=2, ensure_ascii=False)
        tmp_de.replace(deepeval_ready_path)


# ---------------------------------------------------------------------------
# Single Query Execution
# ---------------------------------------------------------------------------
def execute_query(
    api_url: str,
    query: str,
    session_id: str,
    timeout: int = DEFAULT_TIMEOUT_S,
    retries: int = 2,
) -> dict[str, Any]:
    """
    Posts a single engineering query to the backend chat endpoint with retry logic.
    """
    chat_url = f"{api_url.rstrip('/')}/api/v1/chat"
    payload = {"query": query, "session_id": session_id}

    for attempt in range(1, retries + 1):
        try:
            t0 = time.perf_counter()
            resp = requests.post(chat_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            duration = time.perf_counter() - t0
            data = resp.json()
            data["_client_latency_s"] = round(duration, 2)
            return data
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"Query attempt {attempt}/{retries} failed for '{query[:60]}...': {e}")
            if attempt < retries:
                time.sleep(5)
            else:
                raise


# ---------------------------------------------------------------------------
# Main Batch Runner
# ---------------------------------------------------------------------------
def run_batch_evaluation(
    dataset_choice: str = "all",
    limit: int | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    deepeval_ready_path: Path = DEFAULT_DEEPEVAL_READY_PATH,
    api_url: str = DEFAULT_API_URL,
    timeout: int = DEFAULT_TIMEOUT_S,
    resume: bool = True,
    delay_s: float = 2.0,
) -> None:
    """Runs through dataset test cases and saves generated answers and retrieved chunks."""
    all_cases = load_and_normalize_cases(dataset_choice)
    if limit is not None and limit > 0:
        all_cases = all_cases[:limit]

    output_data = load_existing_results(output_path)
    existing_ids = {r["id"] for r in output_data.get("results", [])}

    output_data["metadata"]["total_test_cases"] = len(all_cases)
    output_data["metadata"]["api_endpoint"] = api_url

    total_count = len(all_cases)
    logger.info(f"Starting batch generation: {total_count} total cases (Already completed: {len(existing_ids)})")

    for idx, case in enumerate(all_cases, start=1):
        case_id = case["id"]
        query = case["query"]

        if resume and case_id in existing_ids:
            logger.info(f"[{idx}/{total_count}] Skipping {case_id} (already generated)")
            continue

        logger.info(f"[{idx}/{total_count}] Executing {case_id}: '{query[:75]}...'")
        session_id = f"batch_{case_id}_{int(time.time())}"

        try:
            resp_data = execute_query(
                api_url=api_url,
                query=query,
                session_id=session_id,
                timeout=timeout,
            )

            actual_output = resp_data.get("response", "")
            citations = resp_data.get("citations", [])
            retrieved_chunks = resp_data.get("retrieved_chunks", [])
            telemetry = resp_data.get("telemetry", {})
            routing_decision = resp_data.get("routing_decision", "rag_retrieval")
            circuit_breaker_tripped = resp_data.get("circuit_breaker_tripped", False)
            latency = telemetry.get("latency_seconds", resp_data.get("_client_latency_s", 0.0))

            record = {
                "id": case_id,
                "dataset_source": case["dataset_source"],
                "category": case["category"],
                "difficulty": case["difficulty"],
                "query": query,
                "expected_output": case["expected_output"],
                "reference_context": case["reference_context"],
                "source_documents": case["source_documents"],
                "actual_output": actual_output,
                "retrieved_chunks": retrieved_chunks,
                "citations": citations,
                "routing_decision": routing_decision,
                "circuit_breaker_tripped": circuit_breaker_tripped,
                "latency_seconds": latency,
                "total_chunks_retrieved": len(retrieved_chunks),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

            # If there are additional dataset-specific fields, preserve them
            if "evaluation_criteria" in case:
                record["evaluation_criteria"] = case["evaluation_criteria"]
            if "must_include_keywords" in case:
                record["must_include_keywords"] = case["must_include_keywords"]

            output_data["results"].append(record)
            existing_ids.add(case_id)
            output_data["metadata"]["completed_cases"] = len(output_data["results"])

            # Save immediately after each case
            save_results(output_path, output_data, deepeval_ready_path)

            logger.success(
                f"[{idx}/{total_count}] Finished {case_id} in {latency:.2f}s | "
                f"Answer len: {len(actual_output)} chars | Chunks: {len(retrieved_chunks)} | "
                f"Saved -> {output_path.name}"
            )

        except Exception as e:
            logger.error(f"[{idx}/{total_count}] Failed {case_id}: {e}")
            error_record = {
                "id": case_id,
                "dataset_source": case["dataset_source"],
                "category": case["category"],
                "difficulty": case["difficulty"],
                "query": query,
                "expected_output": case["expected_output"],
                "reference_context": case["reference_context"],
                "source_documents": case["source_documents"],
                "actual_output": f"ERROR: Execution failed: {str(e)}",
                "retrieved_chunks": [],
                "citations": [],
                "routing_decision": "error",
                "circuit_breaker_tripped": True,
                "latency_seconds": 0.0,
                "total_chunks_retrieved": 0,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }
            output_data["results"].append(error_record)
            existing_ids.add(case_id)
            save_results(output_path, output_data, deepeval_ready_path)

        if delay_s > 0:
            time.sleep(delay_s)

    logger.success(
        f"Batch execution completed! Total saved: {len(output_data['results'])} records into {output_path} and {deepeval_ready_path}"
    )


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHP Agentic RAG - Batch Answer & Retrieval Generator")
    parser.add_argument(
        "--dataset",
        choices=["all", "shp", "turbines"],
        default="all",
        help="Which golden dataset to process (default: all)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to run")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output JSON file (default: src/evals/generated_answers_and_chunks.json)",
    )
    parser.add_argument(
        "--deepeval-output",
        type=Path,
        default=DEFAULT_DEEPEVAL_READY_PATH,
        help="Path to DeepEval-formatted output JSON (default: src/evals/generated_answers_and_chunks_deepeval_ready.json)",
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Backend API server URL")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="Per-query timeout in seconds")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume; restart from beginning")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between queries in seconds")

    args = parser.parse_args()

    run_batch_evaluation(
        dataset_choice=args.dataset,
        limit=args.limit,
        output_path=args.output,
        deepeval_ready_path=args.deepeval_output,
        api_url=args.api_url,
        timeout=args.timeout,
        resume=not args.no_resume,
        delay_s=args.delay,
    )

