"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Retrieval Grader Nodes
=============================================================================
Module: src.agents.retrieval_grader.agentnodes
Purpose: Implements the execution node for the Reference-Free Retrieval
         Sufficiency Grader (Production CRAG). Evaluates whether synthesized
         context contains sufficient engineering data, formulas, and parameters
         to answer the user query and planned sub-queries before generation.
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

from src.states.retrieval_grader_agent_state import (
    RetrievalGraderOutput,
    RetrievalGraderState,
)

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL_NAME = os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:9b")
DEFAULT_TIMEOUT_S = int(os.getenv("RETRIEVAL_GRADER_TIMEOUT_S", "120"))
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
DEFAULT_NUM_PREDICT = int(os.getenv("RETRIEVAL_GRADER_MAX_TOKENS", "2048"))
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
    "logs/retrieval_grader.log",
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


def render_grader_prompts(
    user_query: str,
    sub_queries: list[dict[str, Any]] | None,
    synthesized_context: str,
) -> tuple[str, str]:
    """Renders system and user evaluation prompts using Jinja2 templates."""
    system_template = jinja_env.get_template("retrieval_grader_system.jinja2")
    user_template = jinja_env.get_template("retrieval_grader_user.jinja2")

    system_content = system_template.render()
    user_content = user_template.render(
        user_query=user_query,
        sub_queries=sub_queries or [],
        synthesized_context=synthesized_context,
    )
    return system_content, user_content


# ---------------------------------------------------------------------------
# Deterministic Fallback Heuristic
# ---------------------------------------------------------------------------
def _deterministic_grader_fallback(
    user_query: str,
    synthesized_context: str,
    sub_queries: list[dict[str, Any]] | None = None,
) -> RetrievalGraderOutput:
    """
    Fallback sufficiency assessment in case Ollama service is unreachable
    or fails to emit valid JSON. Uses keyword density and length heuristics.
    """
    context_lower = synthesized_context.lower()
    query_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", user_query.lower()))

    # Ignore common query stop-words
    stop_words = {
        "what", "which", "where", "when", "could", "would", "should",
        "please", "explain", "calculate", "compare", "provide", "check",
    }
    keywords = [w for w in query_words if w not in stop_words]

    matched_keywords = [kw for kw in keywords if kw in context_lower]
    coverage = len(matched_keywords) / max(len(keywords), 1)

    # Check for specific quantitative constraints (e.g. 45m, 120m, 15m3/s)
    query_numbers = set(re.findall(r"\b\d+(?:\.\d+)?(?:m|m3/s|kw|mw)?\b", user_query.lower()))
    missing_numbers = [n for n in query_numbers if n not in context_lower and len(n) > 1]

    # Engineering context is sufficient if it has substance (>200 chars), covers >= 40% keywords (or >600 chars), and no critical numbers are missing
    is_sufficient = (len(synthesized_context.strip()) > 200) and (coverage >= 0.40 or len(synthesized_context.strip()) > 600) and (len(missing_numbers) == 0)

    missing = []
    if missing_numbers:
        missing.extend([f"Plant parameters, curves, or operating data for '{n}'" for n in missing_numbers])
    missing.extend([f"Technical data regarding {kw}" for kw in keywords if kw not in matched_keywords])
    if not missing:
        missing = ["Detailed comparative selection envelope and specific head suitability data"]

    critique = (
        f"Deterministic heuristic: Context length={len(synthesized_context)} chars, "
        f"keyword match={len(matched_keywords)}/{len(keywords)} ({coverage:.0%}), "
        f"missing parameters={missing_numbers}."
    )

    return RetrievalGraderOutput(
        is_sufficient=is_sufficient,
        missing_aspects=missing,
        critique=critique,
        confidence_score=0.70,
    )


# ---------------------------------------------------------------------------
# Core Sufficiency Grading Function
# ---------------------------------------------------------------------------
def grade_retrieval_sufficiency(
    user_query: str,
    synthesized_context: str,
    sub_queries: list[dict[str, Any]] | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> RetrievalGraderOutput:
    """
    Evaluates whether the synthesized context contains sufficient engineering facts,
    parameters, and formulas to answer the user query and planned sub-queries.

    Args:
        user_query: Primary engineering question.
        synthesized_context: Synthesized dossier or retrieved chunks.
        sub_queries: Optional list of planned SubQuery dicts from Planner.
        ollama_url: Base URL of local Ollama service.
        model_name: Model identifier (qwen3.5:9b).
        timeout: HTTP request timeout in seconds.

    Returns:
        RetrievalGraderOutput containing boolean is_sufficient, missing_aspects, critique.
    """
    t0 = time.perf_counter()
    logger.info(f"Grading retrieval sufficiency for query: '{user_query}'")

    # Fast check: completely empty context
    if not synthesized_context or not synthesized_context.strip():
        logger.warning("Empty synthesized context provided. Grader marking insufficient.")
        return RetrievalGraderOutput(
            is_sufficient=False,
            missing_aspects=["All technical documentation and calculation outputs"],
            critique="Retrieved context is empty. No engineering evidence was found.",
            confidence_score=1.0,
        )

    # Render Jinja2 Prompts
    system_msg, user_msg = render_grader_prompts(
        user_query=user_query,
        sub_queries=sub_queries,
        synthesized_context=synthesized_context,
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

        logger.debug(f"Raw Grader response length={len(raw_text)} | snippet: {raw_text[:120]}")

        # Robust JSON extraction from content and thinking
        parsed = None
        for candidate in [raw_text, thinking_text]:
            if not candidate:
                continue
            cleaned = re.sub(r"<think>[\s\S]*?</think>", "", candidate).strip()
            # 1. Direct parse
            try:
                parsed = json.loads(cleaned, strict=False)
                if isinstance(parsed, dict) and "is_sufficient" in parsed:
                    break
            except Exception:
                pass
            # 2. Markdown fence
            fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", candidate)
            if fence_match:
                try:
                    parsed = json.loads(fence_match.group(1).strip(), strict=False)
                    if isinstance(parsed, dict) and "is_sufficient" in parsed:
                        break
                except Exception:
                    pass
            # 3. Outer balanced { ... }
            s_idx = candidate.find("{")
            e_idx = candidate.rfind("}")
            if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
                try:
                    parsed = json.loads(candidate[s_idx : e_idx + 1].strip(), strict=False)
                    if isinstance(parsed, dict) and "is_sufficient" in parsed:
                        break
                except Exception:
                    pass

        if not parsed or not isinstance(parsed, dict):
            raise ValueError(f"Could not extract valid JSON from Grader response: {raw_text[:100]}")

        output = RetrievalGraderOutput.model_validate(parsed)

        # Enforce consistency rule: if is_sufficient is True, missing_aspects must be empty
        if output.is_sufficient and output.missing_aspects:
            logger.debug("Clearing non-empty missing_aspects because is_sufficient is True.")
            output.missing_aspects = []

        duration = time.perf_counter() - t0
        logger.success(
            f"Retrieval grading completed in {duration:.2f}s | Sufficient: {output.is_sufficient} | "
            f"Missing Aspects: {len(output.missing_aspects)} | Confidence: {output.confidence_score:.2f}"
        )
        return output

    except (
        requests.RequestException,
        json.JSONDecodeError,
        ValidationError,
        KeyError,
        ValueError,
    ) as e:
        logger.error(f"Retrieval Grader LLM call failed: {e}. Activating deterministic fallback.")
        return _deterministic_grader_fallback(
            user_query=user_query,
            synthesized_context=synthesized_context,
            sub_queries=sub_queries,
        )


# ---------------------------------------------------------------------------
# LangGraph Node Definition
# ---------------------------------------------------------------------------
def grade_retrieval_node(state: RetrievalGraderState) -> dict[str, Any]:
    """
    LangGraph node function for the Retrieval Sufficiency Grader.

    Reads:
      - `query` or `user_query`
      - `synthesized_context` (falls back to chunk contents if not synthesized)
      - `sub_queries`

    Writes:
      - `is_retrieval_sufficient`: bool
      - `missing_aspects`: list[str]
      - `retrieval_critique`: str
      - `retrieval_confidence`: float
    """
    query = state.get("query") or state.get("user_query") or ""
    sub_queries = state.get("sub_queries")
    synthesized_context = state.get("synthesized_context")

    # If synthesized_context is missing, assemble from documents chunks
    if not synthesized_context and state.get("documents"):
        doc_texts = [
            f"[Source: {d.get('source_file', 'unknown')}]\n{d.get('content', '')}"
            for d in state["documents"]
        ]
        synthesized_context = "\n\n".join(doc_texts)

    synthesized_context = synthesized_context or ""

    output = grade_retrieval_sufficiency(
        user_query=query,
        synthesized_context=synthesized_context,
        sub_queries=sub_queries,
    )

    return {
        "is_retrieval_sufficient": output.is_sufficient,
        "missing_aspects": output.missing_aspects,
        "retrieval_critique": output.critique,
        "retrieval_confidence": output.confidence_score,
        "error": None,
    }


__all__ = [
    "grade_retrieval_node",
    "grade_retrieval_sufficiency",
    "render_grader_prompts",
]

