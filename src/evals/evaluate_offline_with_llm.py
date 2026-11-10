"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Multi-Provider Offline Evaluator
=============================================================================
Module: src.evals.evaluate_offline_with_llm
Purpose: Evaluates generated answers and retrieved chunks from
         'generated_answers_and_chunks.json' against golden dataset benchmarks.
         Uses free-tier API keys with a robust multi-provider waterfall fallback:
           1. Google Gemini (gemini-2.5-flash, gemini-flash-latest, gemini-2.5-flash-lite, gemini-2.5-pro)
           2. Groq (openai/gpt-oss-120b, qwen/qwen3.8-27b, openai/gpt-oss-20b, allam-2-7b)
           3. Hugging Face Inference API (if enabled/available)
           4. Local Ollama (qwen3.5:9b) fallback
         Evaluates on 4 standard RAG metrics:
           - Faithfulness (Context Groundedness) [0.0 - 1.0]
           - Answer Relevancy (Query Alignment & Completeness) [0.0 - 1.0]
           - Context Precision (Signal-to-Noise in Retrieval) [0.0 - 1.0]
           - Context Recall (Ground Truth Coverage in Retrieval) [0.0 - 1.0]
         Features atomic incremental saving, --resume support, and rate-limit handling.
=============================================================================
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ---------------------------------------------------------------------------
# Default Paths
# ---------------------------------------------------------------------------
DEFAULT_INPUT_FILE = Path("src/evals/generated_answers_and_chunks.json")
DEFAULT_REPORT_FILE = Path("src/evals/evaluation_report.json")

# Logger setup
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# ---------------------------------------------------------------------------
# Prompt Definition
# ---------------------------------------------------------------------------
EVAL_SYSTEM_PROMPT = """You are an expert impartial evaluation judge for a Small Hydropower Plant (SHP) Engineering Agentic RAG system.
Your job is to objectively score the AI's generated response against the retrieved reference context and expected ground truth answer.

Score each dimension strictly between 0.00 and 1.00:
1. "faithfulness_score":
   - 1.00: Every factual claim, equation, parameter, and engineering recommendation in the generated answer is directly substantiated by the retrieved context.
   - 0.60-0.85: Mostly grounded, but introduces minor extrapolated details not strictly found in chunks.
   - <0.50: Contains unsubstantiated claims, incorrect facts, or contradicts the retrieved context.

2. "answer_relevancy_score":
   - 1.00: The answer completely, directly, and specifically addresses all facets of the user query with correct technical depth and clarity.
   - 0.60-0.85: Answers the main question but omits minor parameters or adds unnecessary verbosity.
   - <0.50: Off-topic, vague, or fails to address the core inquiry.

3. "context_precision_score":
   - 1.00: The retrieved context chunks are highly relevant, specific to the query, and free of irrelevant noise.
   - 0.60-0.85: Contains relevant chunks along with some tangential context.
   - <0.50: Mostly irrelevant or misaligned chunks.

4. "context_recall_score":
   - 1.00: The retrieved context contains all necessary facts and clauses required to formulate the expected ground truth answer.
   - 0.60-0.85: Contains the primary facts but misses some specific numbers or secondary details from the ground truth.
   - <0.50: Missing critical source information needed to answer the question accurately.

5. "hallucination_detected":
   - Boolean: true if the answer contains fabricated facts/formulas not grounded in the context; false otherwise.

6. "reasoning":
   - Concise technical critique explaining the assigned scores in 2-3 sentences.

You must respond ONLY with a valid JSON object matching this schema:
{
  "faithfulness_score": <float 0.0-1.0>,
  "answer_relevancy_score": <float 0.0-1.0>,
  "context_precision_score": <float 0.0-1.0>,
  "context_recall_score": <float 0.0-1.0>,
  "hallucination_detected": <boolean>,
  "reasoning": "<string>"
}"""


def build_case_prompt(case: dict[str, Any], max_chunk_chars: int | None = None) -> str:
    """Builds the evaluation user prompt for a single case, with optional chunk capping for token-limited free tiers."""
    query = case.get("query", "").strip()
    actual_output = case.get("actual_output", "").strip()
    expected_output = case.get("expected_output", "").strip()
    retrieved_chunks = case.get("retrieved_chunks", [])

    context_parts = []
    for idx, c in enumerate(retrieved_chunks, start=1):
        source = c.get("source_file", "Unknown")
        breadcrumb = c.get("breadcrumb", "")
        content = c.get("content", "").strip()
        if max_chunk_chars and len(content) > max_chunk_chars:
            content = content[:max_chunk_chars] + "\n... [content truncated for rate limits]"
        header = f"[Chunk {idx} | Source: {source}"
        if breadcrumb:
            header += f" | Section: {breadcrumb}"
        header += "]"
        context_parts.append(f"{header}\n{content}")

    context_text = "\n\n".join(context_parts) if context_parts else "(No context chunks retrieved)"

    return f"""### User Query:
{query}

### Retrieved Reference Context:
{context_text}

### Expected Ground Truth Answer:
{expected_output}

### AI Generated Answer to Evaluate:
{actual_output}

Provide your objective JSON evaluation."""


def extract_json(raw_text: str) -> dict[str, Any]:
    """Robustly extracts JSON from an LLM response string."""
    clean = raw_text.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    # Locate outer curly braces
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1:
        clean = clean[start : end + 1]

    data = json.loads(clean)
    # Ensure float bounds
    for k in [
        "faithfulness_score",
        "answer_relevancy_score",
        "context_precision_score",
        "context_recall_score",
    ]:
        if k in data:
            data[k] = max(0.0, min(1.0, float(data[k])))
    if "hallucination_detected" not in data:
        data["hallucination_detected"] = data.get("faithfulness_score", 1.0) < 0.7
    return data


# ---------------------------------------------------------------------------
# Individual Provider Evaluators
# ---------------------------------------------------------------------------
def call_gemini(prompt: str, model_name: str, api_key: str) -> dict[str, Any]:
    """Invokes Google Gemini API with JSON response format."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=EVAL_SYSTEM_PROMPT,
        generation_config={"response_mime_type": "application/json", "temperature": 0.0},
    )
    response = model.generate_content(prompt)
    return extract_json(response.text)


def call_groq(prompt: str, model_name: str, api_key: str) -> dict[str, Any]:
    """Invokes Groq API with JSON response format."""
    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": EVAL_SYSTEM_PROMPT + "\nRespond strictly in valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    return extract_json(content)


def call_huggingface(prompt: str, model_name: str, api_key: str) -> dict[str, Any]:
    """Invokes HuggingFace Inference API."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.01,
        max_tokens=1024,
    )
    content = response.choices[0].message.content or "{}"
    return extract_json(content)


def call_ollama(prompt: str, model_name: str = "qwen3.5:9b") -> dict[str, Any]:
    """Local Ollama fallback evaluator."""
    import httpx

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model_name,
        "prompt": f"{EVAL_SYSTEM_PROMPT}\n\n{prompt}",
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return extract_json(data.get("response", "{}"))


# ---------------------------------------------------------------------------
# Multi-Provider Waterfall Router
# ---------------------------------------------------------------------------
class MultiProviderEvaluator:
    """
    Manages evaluation across multiple providers and models with automatic
    rate-limit detection, backoff, and waterfall failover.
    """

    def __init__(self, mode: str = "auto", preferred_model: str | None = None):
        self.mode = mode
        self.preferred_model = preferred_model

        # Extract API keys from .env
        self.gemini_key = (
            os.getenv("free_gemini_key")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        self.groq_key = os.getenv("Groq_key") or os.getenv("GROQ_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")

        # Build prioritized candidate list
        self.candidates: list[dict[str, str]] = []
        self._build_candidate_pipeline()

        logger.info(f"Initialized MultiProviderEvaluator (mode='{mode}'). Active waterfall candidates: {len(self.candidates)}")
        for idx, c in enumerate(self.candidates, start=1):
            logger.info(f"  Priority {idx}: [{c['provider']}] model='{c['model']}'")

    def _build_candidate_pipeline(self) -> None:
        """Constructs the fallback order of (provider, model) pairs."""
        # 1. Google Gemini Models (Primary high-context judge)
        if self.gemini_key:
            gemini_models = [
                "gemini-2.5-flash",
                "gemini-flash-latest",
                "gemini-2.5-flash-lite",
                "gemini-2.5-pro",
            ]
            if self.preferred_model and self.mode == "gemini":
                gemini_models = [self.preferred_model] + [m for m in gemini_models if m != self.preferred_model]
            for m in gemini_models:
                self.candidates.append({"provider": "gemini", "model": m})

        # 2. Groq Models (Fast inference, multiple free-tier open models)
        if self.groq_key:
            groq_models = [
                "openai/gpt-oss-120b",
                "qwen/qwen3.8-27b",
                "openai/gpt-oss-20b",
                "allam-2-7b",
            ]
            if self.preferred_model and self.mode == "groq":
                groq_models = [self.preferred_model] + [m for m in groq_models if m != self.preferred_model]
            for m in groq_models:
                self.candidates.append({"provider": "groq", "model": m})

        # 3. Hugging Face Models (Optional fallback if available)
        if self.hf_token:
            hf_models = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-72B-Instruct"]
            for m in hf_models:
                self.candidates.append({"provider": "huggingface", "model": m})

        # 4. Local Ollama (Ultimate safety net)
        self.candidates.append({"provider": "ollama", "model": os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:9b")})

    def evaluate_case(self, case: dict[str, Any], max_retries_per_candidate: int = 2) -> dict[str, Any]:
        """
        Evaluates a single test case using the waterfall sequence.
        If a candidate encounters a rate limit (429) or error, it gracefully falls
        back to the next candidate model/provider.
        """
        errors_logged = []

        for candidate in self.candidates:
            provider = candidate["provider"]
            model = candidate["model"]

            # Filter if mode is strictly constrained
            if self.mode != "auto" and provider != self.mode and provider != "ollama":
                continue

            prompt = build_case_prompt(case, max_chunk_chars=1200 if provider == "groq" else None)

            for attempt in range(1, max_retries_per_candidate + 1):
                try:
                    if provider == "gemini":
                        scores = call_gemini(prompt, model, self.gemini_key)
                    elif provider == "groq":
                        scores = call_groq(prompt, model, self.groq_key)
                    elif provider == "huggingface":
                        scores = call_huggingface(prompt, model, self.hf_token)
                    elif provider == "ollama":
                        scores = call_ollama(prompt, model)
                    else:
                        continue

                    # Validate required keys
                    if all(k in scores for k in ["faithfulness_score", "answer_relevancy_score"]):
                        scores["evaluator_provider"] = provider
                        scores["evaluator_model"] = model
                        return scores

                except Exception as exc:
                    err_str = str(exc)
                    is_rate_limit = any(
                        s in err_str.lower()
                        for s in ["429", "resourceexhausted", "rate_limit", "quota", "too many requests"]
                    )

                    if is_rate_limit:
                        backoff = 4.0 * attempt
                        logger.warning(
                            f"[{provider}:{model}] Rate limit encountered (attempt {attempt}/{max_retries_per_candidate}). "
                            f"Pausing {backoff:.1f}s before fallback..."
                        )
                        time.sleep(backoff)
                    else:
                        logger.warning(f"[{provider}:{model}] Evaluation error: {err_str[:120]} (attempt {attempt})")
                        time.sleep(1.0)
                    errors_logged.append(f"{provider}:{model} -> {err_str[:80]}")

        # If all candidates fail, return zero scores with diagnostic info
        logger.error(f"All waterfall candidates failed for case {case.get('id')}.")
        return {
            "faithfulness_score": 0.0,
            "answer_relevancy_score": 0.0,
            "context_precision_score": 0.0,
            "context_recall_score": 0.0,
            "hallucination_detected": True,
            "reasoning": f"Waterfall exhaustion. Errors: {'; '.join(errors_logged[:3])}",
            "evaluator_provider": "none",
            "evaluator_model": "failed",
        }


# ---------------------------------------------------------------------------
# Batch Evaluation Runner with Resume & Checkpointing
# ---------------------------------------------------------------------------
def run_evaluation(
    input_file: Path = DEFAULT_INPUT_FILE,
    output_file: Path = DEFAULT_REPORT_FILE,
    mode: str = "auto",
    preferred_model: str | None = None,
    limit: int | None = None,
    delay: float = 2.5,
    resume: bool = True,
) -> None:
    """
    Evaluates all cases in input_file and writes structured report to output_file.
    Supports atomic checkpointing and --resume.
    """
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}. Ensure batch responses are generated first.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if limit:
        results = results[:limit]

    total_cases = len(results)
    logger.info(f"Loaded {total_cases} test cases from {input_file}.")

    # Load existing report if resuming
    evaluated_records: list[dict[str, Any]] = []
    completed_ids: set[str] = set()

    if resume and output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                prev_report = json.load(f)
                evaluated_records = [
                    r for r in prev_report.get("detailed_results", [])
                    if r.get("scores", {}).get("evaluator_provider") not in ("none", "failed")
                ]
                completed_ids = {r.get("id") for r in evaluated_records if r.get("id")}
                logger.info(f"Resuming evaluation: found {len(completed_ids)} already successfully evaluated cases in {output_file}.")
        except Exception as e:
            logger.warning(f"Could not read previous report for resume ({e}); starting fresh.")
            evaluated_records = []
            completed_ids = set()

    evaluator = MultiProviderEvaluator(mode=mode, preferred_model=preferred_model)

    start_time = time.time()
    for idx, case in enumerate(results, start=1):
        case_id = case.get("id", f"case_{idx}")

        if case_id in completed_ids:
            logger.info(f"[{idx}/{total_cases}] Skipping {case_id} (already evaluated)")
            continue

        query_preview = (case.get("query") or "")[:60]
        logger.info(f"[{idx}/{total_cases}] Evaluating {case_id}: '{query_preview}...'")

        scores = evaluator.evaluate_case(case)

        case_record = {
            "id": case_id,
            "category": case.get("category", "General"),
            "query": case.get("query", ""),
            "actual_output": case.get("actual_output", ""),
            "expected_output": case.get("expected_output", ""),
            "scores": scores,
            "retrieved_chunks_count": len(case.get("retrieved_chunks", [])),
            "circuit_breaker_tripped": case.get("circuit_breaker_tripped", False),
            "latency_seconds": case.get("latency_seconds", 0.0),
        }
        evaluated_records.append(case_record)
        completed_ids.add(case_id)

        # Log individual evaluation verdict
        prov = scores.get("evaluator_provider", "unknown")
        mod = scores.get("evaluator_model", "unknown")
        faith = scores.get("faithfulness_score", 0.0)
        relev = scores.get("answer_relevancy_score", 0.0)
        prec = scores.get("context_precision_score", 0.0)
        rec = scores.get("context_recall_score", 0.0)
        logger.info(
            f"  -> [{prov}:{mod}] Faithfulness: {faith:.2f} | Relevancy: {relev:.2f} | Precision: {prec:.2f} | Recall: {rec:.2f}"
        )

        # Atomic incremental save
        save_report(output_file, evaluated_records, mode, evaluator)

        # Free tier rate-limit pacing delay
        if delay > 0 and idx < total_cases:
            time.sleep(delay)

    elapsed = time.time() - start_time
    logger.success(f"All {len(evaluated_records)} evaluations complete in {elapsed:.1f}s! Final report saved to {output_file}")
    print_evaluation_summary(evaluated_records)


def save_report(
    output_file: Path,
    evaluated_records: list[dict[str, Any]],
    mode: str,
    evaluator: MultiProviderEvaluator,
) -> None:
    """Calculates aggregate metrics and writes the report atomically to disk."""
    if not evaluated_records:
        return

    faith_scores = [r["scores"].get("faithfulness_score", 0.0) for r in evaluated_records]
    relev_scores = [r["scores"].get("answer_relevancy_score", 0.0) for r in evaluated_records]
    prec_scores = [r["scores"].get("context_precision_score", 0.0) for r in evaluated_records]
    rec_scores = [r["scores"].get("context_recall_score", 0.0) for r in evaluated_records]
    hallucination_count = sum(1 for r in evaluated_records if r["scores"].get("hallucination_detected", False))

    # Category breakdown
    category_metrics: dict[str, dict[str, Any]] = {}
    for r in evaluated_records:
        cat = r.get("category", "Uncategorized")
        if cat not in category_metrics:
            category_metrics[cat] = {"faith": [], "relev": [], "prec": [], "rec": [], "count": 0}
        category_metrics[cat]["count"] += 1
        category_metrics[cat]["faith"].append(r["scores"].get("faithfulness_score", 0.0))
        category_metrics[cat]["relev"].append(r["scores"].get("answer_relevancy_score", 0.0))
        category_metrics[cat]["prec"].append(r["scores"].get("context_precision_score", 0.0))
        category_metrics[cat]["rec"].append(r["scores"].get("context_recall_score", 0.0))

    cat_breakdown = {}
    for cat, vals in category_metrics.items():
        n = vals["count"]
        cat_breakdown[cat] = {
            "test_cases": n,
            "avg_faithfulness": round(sum(vals["faith"]) / n, 4),
            "avg_answer_relevancy": round(sum(vals["relev"]) / n, 4),
            "avg_context_precision": round(sum(vals["prec"]) / n, 4),
            "avg_context_recall": round(sum(vals["rec"]) / n, 4),
        }

    # Model provider usage breakdown
    provider_counts: dict[str, int] = {}
    for r in evaluated_records:
        key = f"{r['scores'].get('evaluator_provider', 'none')}:{r['scores'].get('evaluator_model', 'unknown')}"
        provider_counts[key] = provider_counts.get(key, 0) + 1

    report = {
        "metadata": {
            "evaluation_mode": mode,
            "total_cases_evaluated": len(evaluated_records),
            "hallucination_rate_percent": round((hallucination_count / len(evaluated_records)) * 100, 2),
            "provider_model_usage": provider_counts,
        },
        "aggregate_metrics": {
            "average_faithfulness": round(sum(faith_scores) / len(faith_scores), 4),
            "average_answer_relevancy": round(sum(relev_scores) / len(relev_scores), 4),
            "average_context_precision": round(sum(prec_scores) / len(prec_scores), 4),
            "average_context_recall": round(sum(rec_scores) / len(rec_scores), 4),
        },
        "category_breakdown": cat_breakdown,
        "detailed_results": evaluated_records,
    }

    tmp_file = output_file.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    tmp_file.replace(output_file)


def print_evaluation_summary(records: list[dict[str, Any]]) -> None:
    """Prints a formatted ASCII evaluation scorecard to standard output."""
    if not records:
        return

    n = len(records)
    avg_faith = sum(r["scores"].get("faithfulness_score", 0.0) for r in records) / n
    avg_relev = sum(r["scores"].get("answer_relevancy_score", 0.0) for r in records) / n
    avg_prec = sum(r["scores"].get("context_precision_score", 0.0) for r in records) / n
    avg_rec = sum(r["scores"].get("context_recall_score", 0.0) for r in records) / n
    hallucination_count = sum(1 for r in records if r["scores"].get("hallucination_detected", False))

    print("\n" + "=" * 78)
    print("           SMALL HYDROPOWER (SHP) AGENTIC RAG EVALUATION BENCHMARK")
    print("=" * 78)
    print(f"Total Test Cases Evaluated : {n}")
    print(f"Average Faithfulness       : {avg_faith:.4f}  ({avg_faith * 100:.1f}%)")
    print(f"Average Answer Relevancy   : {avg_relev:.4f}  ({avg_relev * 100:.1f}%)")
    print(f"Average Context Precision  : {avg_prec:.4f}  ({avg_prec * 100:.1f}%)")
    print(f"Average Context Recall     : {avg_rec:.4f}  ({avg_rec * 100:.1f}%)")
    print(f"Hallucination Cases        : {hallucination_count}/{n} ({(hallucination_count / n) * 100:.1f}%)")
    print("-" * 78)
    print("Category Breakdown:")
    cat_counts: dict[str, list] = {}
    for r in records:
        cat = r.get("category", "General")
        cat_counts.setdefault(cat, []).append(r)
    for cat, r_list in cat_counts.items():
        c_n = len(r_list)
        c_faith = sum(x["scores"].get("faithfulness_score", 0.0) for x in r_list) / c_n
        c_relev = sum(x["scores"].get("answer_relevancy_score", 0.0) for x in r_list) / c_n
        print(f"  * {cat:<24} (N={c_n:2d}): Faithfulness={c_faith:.2f} | Relevancy={c_relev:.2f}")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Provider Offline LLM Evaluator for SHP Agentic RAG")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE, help="Path to generated answers JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_FILE, help="Path to output report JSON")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "gemini", "groq", "huggingface", "ollama"],
        help="Evaluation provider mode. 'auto' uses Gemini -> Groq -> HF -> Ollama waterfall.",
    )
    parser.add_argument("--model", default=None, help="Optional model name override")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases to evaluate (for testing)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay in seconds between calls to prevent rate limits")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume; start fresh evaluation")

    args = parser.parse_args()
    run_evaluation(
        input_file=args.input,
        output_file=args.output,
        mode=args.mode,
        preferred_model=args.model,
        limit=args.limit,
        delay=args.delay,
        resume=args.resume,
    )
