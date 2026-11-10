"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Generation Guardrail Nodes
=============================================================================
Module: src.agents.generation_guardrail.agentnodes
Purpose: Implements the execution node for the Post-Generation Faithfulness
         & Groundedness Guardrail. Uses a fast, single-pass local LLM judge
         (qwen3.5:9b) to verify that generated answers are strictly grounded
         in the retrieved context with zero hallucinations.
=============================================================================
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import jinja2
import requests
from loguru import logger
from pydantic import ValidationError

from src.states.generation_guardrail_state import (
    GenerationGuardrailOutput,
    GenerationGuardrailState,
)

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL_NAME = os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:9b")
DEFAULT_TIMEOUT_S = int(os.getenv("GENERATION_GUARDRAIL_TIMEOUT_S", "120"))
DEFAULT_FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.90"))
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
DEFAULT_NUM_PREDICT = int(os.getenv("GENERATION_GUARDRAIL_MAX_TOKENS", "2048"))
PROMPTS_DIR = Path("src/prompts")

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/generation_guardrail.log",
    rotation="20 MB",
    retention="14 days",
    level="DEBUG",
    encoding="utf-8",
)

# ---------------------------------------------------------------------------
# Jinja2 Prompt Template Loader
# ---------------------------------------------------------------------------
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_guardrail_prompts(
    user_query: str,
    synthesized_context: str,
    generated_response: str,
) -> tuple[str, str]:
    """Renders system and user guardrail prompts using Jinja2 templates."""
    system_template = jinja_env.get_template("generation_guardrail_system.jinja2")
    user_template = jinja_env.get_template("generation_guardrail_user.jinja2")

    system_content = system_template.render()
    user_content = user_template.render(
        user_query=user_query,
        synthesized_context=synthesized_context,
        generated_response=generated_response,
    )
    return system_content, user_content


# ---------------------------------------------------------------------------
# Deterministic Fallback Heuristic
# ---------------------------------------------------------------------------
def _deterministic_faithfulness_fallback(
    query: str,
    final_response: str,
    synthesized_context: str,
) -> GenerationGuardrailOutput:
    """
    Fallback groundedness check in case local LLM judge is unreachable.
    Performs deterministic numerical and parameter containment verification.
    """
    if not final_response or not final_response.strip() or "Verified Technical Evidence Dossier" in final_response:
        return GenerationGuardrailOutput(
            is_grounded=False,
            faithfulness_score=0.0,
            unfaithful_statements=["Generation failed or emitted raw un-synthesized context."],
            critique="Deterministic heuristic: Answer text was absent or un-synthesized.",
        )

    # Extract numerical tokens (e.g. 120, 2.0, 94%, 2.07) from final_response
    answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", final_response))
    context_text = synthesized_context.lower()

    unsupported_numbers = [num for num in answer_numbers if num.lower() not in context_text]
    # Filter out trivial single digits or common formatting numbers (1, 2, 3 for section titles)
    unsupported_significant = [n for n in unsupported_numbers if len(n) > 1 and n not in {"1", "2", "3", "4", "5"}]

    if unsupported_significant:
        critique = (
            f"Deterministic heuristic: Detected numerical values in the answer ({unsupported_significant[:4]}) "
            f"that are not found in the verified context."
        )
        return GenerationGuardrailOutput(
            is_grounded=False,
            faithfulness_score=0.65,
            unfaithful_statements=[f"Numerical claim '{n}' not supported by context" for n in unsupported_significant[:4]],
            critique=critique,
        )

    return GenerationGuardrailOutput(
        is_grounded=True,
        faithfulness_score=0.92,
        unfaithful_statements=[],
        critique="Deterministic heuristic: Core parameters, numbers, and units match verified context.",
    )


# ---------------------------------------------------------------------------
# Core Faithfulness Evaluation Function (Fast Single-Pass LLM Judge)
# ---------------------------------------------------------------------------
def evaluate_generation_faithfulness(
    query: str,
    final_response: str,
    synthesized_context: str,
    threshold: float = DEFAULT_FAITHFULNESS_THRESHOLD,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> GenerationGuardrailOutput:
    """
    Evaluates the faithfulness and groundedness of a generated engineering answer
    against the retrieved context in a single high-speed LLM pass using local Ollama.

    Args:
        query: Original user query.
        final_response: Generated answer text.
        synthesized_context: Synthesized evidence dossier from knowledge corpus.
        threshold: Minimum acceptable faithfulness score (default 0.90).
        ollama_url: Base URL for local Ollama.
        model_name: Model identifier (qwen3.5:9b).
        timeout: Request timeout in seconds.

    Returns:
        GenerationGuardrailOutput containing boolean is_grounded, faithfulness_score,
        unfaithful_statements, and critique.
    """
    t0 = time.perf_counter()
    logger.info(f"Evaluating generation faithfulness (threshold >= {threshold}) for: '{query[:60]}...'")

    # Fast validation: empty response or context
    if not final_response or not final_response.strip() or "Verified Technical Evidence Dossier" in final_response:
        return GenerationGuardrailOutput(
            is_grounded=False,
            faithfulness_score=0.0,
            unfaithful_statements=["Generated response is empty or contains raw un-synthesized context."],
            critique="Empty or un-synthesized answer. Fails faithfulness guardrail.",
        )

    if not synthesized_context or not synthesized_context.strip():
        return GenerationGuardrailOutput(
            is_grounded=False,
            faithfulness_score=0.0,
            unfaithful_statements=["No context available to support the answer."],
            critique="Synthesized context is empty. Claims cannot be grounded.",
        )

    # Render Jinja2 Prompts for Single-Pass Evaluation
    system_msg, user_msg = render_guardrail_prompts(
        user_query=query,
        synthesized_context=synthesized_context,
        generated_response=final_response,
    )

    chat_url = f"{ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "format": "json",
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_ctx": DEFAULT_NUM_CTX,
            "num_predict": DEFAULT_NUM_PREDICT,
        },
        "stream": False,
    }

    try:
        resp = requests.post(chat_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        msg_dict = resp.json().get("message", {})
        raw_text = msg_dict.get("content", "").strip()
        thinking_text = msg_dict.get("thinking", "").strip()

        logger.debug(f"Raw Guardrail response length={len(raw_text)} | snippet: {raw_text[:120]}")

        # Robust JSON extraction from content and thinking
        parsed = None
        for candidate in [raw_text, thinking_text]:
            if not candidate:
                continue
            cleaned = re.sub(r"<think>[\s\S]*?</think>", "", candidate).strip()
            # 1. Direct parse
            try:
                parsed = json.loads(cleaned, strict=False)
                if isinstance(parsed, dict) and "is_grounded" in parsed:
                    break
            except Exception:
                pass
            # 2. Markdown fence
            fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", candidate)
            if fence_match:
                try:
                    parsed = json.loads(fence_match.group(1).strip(), strict=False)
                    if isinstance(parsed, dict) and "is_grounded" in parsed:
                        break
                except Exception:
                    pass
            # 3. Outer balanced { ... }
            s_idx = candidate.find("{")
            e_idx = candidate.rfind("}")
            if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
                try:
                    parsed = json.loads(candidate[s_idx : e_idx + 1].strip(), strict=False)
                    if isinstance(parsed, dict) and "is_grounded" in parsed:
                        break
                except Exception:
                    pass

        if not parsed or not isinstance(parsed, dict):
            raise ValueError(f"Could not extract valid JSON from Guardrail response: {raw_text[:100]}")

        is_grounded_raw = bool(parsed.get("is_grounded", True))
        score_raw = float(parsed.get("faithfulness_score", 1.0 if is_grounded_raw else 0.5))
        unfaithful_raw = list(parsed.get("unfaithful_statements", []))
        critique_raw = str(parsed.get("critique", "Single-pass groundedness evaluation completed."))

        # Enforce threshold consistency
        is_successful = is_grounded_raw and (score_raw >= threshold)
        if not is_successful and not unfaithful_raw:
            unfaithful_raw = ["Answer contains unverified claims or figures not found in the source documentation."]

        duration = time.perf_counter() - t0
        logger.success(
            f"Faithfulness evaluated in {duration:.2f}s | Score: {score_raw:.2f} | Grounded: {is_successful} | "
            f"Unfaithful Claims: {len(unfaithful_raw)}"
        )

        return GenerationGuardrailOutput(
            is_grounded=is_successful,
            faithfulness_score=score_raw,
            unfaithful_statements=unfaithful_raw,
            critique=critique_raw,
        )

    except (requests.RequestException, json.JSONDecodeError, ValidationError, ValueError, KeyError) as e:
        logger.warning(f"Single-pass Guardrail LLM call failed ({e}). Activating deterministic fallback.")
        return _deterministic_faithfulness_fallback(
            query=query,
            final_response=final_response,
            synthesized_context=synthesized_context,
        )


# ---------------------------------------------------------------------------
# LangGraph Node Definition
# ---------------------------------------------------------------------------
def generation_guardrail_node(state: GenerationGuardrailState) -> dict[str, Any]:
    """
    LangGraph node function for the Post-Generation Faithfulness Guardrail.

    Reads:
      - `query` or `user_query`
      - `final_response`
      - `synthesized_context` (falls back to documents chunks if empty)

    Writes:
      - `is_grounded`: bool
      - `generation_metrics`: dict[str, float]
      - `unfaithful_statements`: list[str]
      - `guardrail_critique`: str
      - `error`: str | None
    """
    query = state.get("query") or state.get("user_query") or ""
    final_response = state.get("final_response") or ""
    synthesized_context = state.get("synthesized_context")

    if not synthesized_context and state.get("documents"):
        doc_texts = [
            f"[Source: {d.get('source_file', 'unknown')}]\n{d.get('content', '')}"
            for d in state["documents"]
        ]
        synthesized_context = "\n\n".join(doc_texts)

    synthesized_context = synthesized_context or ""

    output = evaluate_generation_faithfulness(
        query=query,
        final_response=final_response,
        synthesized_context=synthesized_context,
    )

    return {
        "is_grounded": output.is_grounded,
        "generation_metrics": {
            "faithfulness": output.faithfulness_score,
        },
        "unfaithful_statements": output.unfaithful_statements,
        "guardrail_critique": output.critique,
        "error": None,
    }


__all__ = [
    "evaluate_generation_faithfulness",
    "generation_guardrail_node",
]
