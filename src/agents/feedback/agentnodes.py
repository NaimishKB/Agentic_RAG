"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Feedback Agent Nodes
=============================================================================
Module: src.agents.feedback.agentnodes
Purpose: Implements the execution node for the Feedback Agent. Performs root-cause
         diagnostic reflection on retrieval and generation failures, formulates
         structured remediation guidance, increments the cycle count, and
         loops back to the Planner Agent.
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

from src.states.feedbackagent_state import (
    FeedbackAgentState,
    FeedbackDiagnosis,
)

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL_NAME = os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:9b")
DEFAULT_TIMEOUT_S = int(os.getenv("FEEDBACK_TIMEOUT_S", "120"))
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
DEFAULT_NUM_PREDICT = int(os.getenv("FEEDBACK_MAX_TOKENS", "2048"))
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
    "logs/feedback_agent.log",
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


def render_feedback_prompts(
    user_query: str,
    cycle_count: int,
    is_retrieval_sufficient: bool | None,
    missing_aspects: list[str] | None,
    retrieval_critique: str | None,
    is_grounded: bool | None,
    generation_metrics: dict[str, float] | None,
    final_response: str | None,
    synthesized_context: str | None,
    unfaithful_statements: list[str] | None = None,
    guardrail_critique: str | None = None,
) -> tuple[str, str]:
    """Renders system and user diagnostic prompts using Jinja2 templates."""
    system_template = jinja_env.get_template("feedback_agent_system.jinja2")
    user_template = jinja_env.get_template("feedback_agent_user.jinja2")

    system_content = system_template.render()
    user_content = user_template.render(
        user_query=user_query,
        cycle_count=cycle_count,
        is_retrieval_sufficient=is_retrieval_sufficient,
        missing_aspects=missing_aspects or [],
        retrieval_critique=retrieval_critique or "",
        is_grounded=is_grounded,
        generation_metrics=generation_metrics or {},
        final_response=final_response or "",
        synthesized_context=synthesized_context or "",
        unfaithful_statements=unfaithful_statements or [],
        guardrail_critique=guardrail_critique or "",
    )
    return system_content, user_content


# ---------------------------------------------------------------------------
# Deterministic Fallback Heuristic
# ---------------------------------------------------------------------------
def _deterministic_feedback_fallback(
    user_query: str,
    missing_aspects: list[str] | None = None,
    retrieval_critique: str | None = None,
    is_retrieval_sufficient: bool | None = None,
) -> FeedbackDiagnosis:
    """
    Deterministic fallback diagnostic in case Ollama service is unreachable.
    Synthesizes actionable guidance directly from Grader failure notes.
    """
    missing = missing_aspects or ["Specific engineering formulas and parameters"]
    critique = retrieval_critique or "Retrieved context lacked required engineering specifications."

    remediation = (
        f"Reformulate retrieval sub-queries to explicitly search for: {', '.join(missing[:3])}. "
        f"Target engineering manuals or invoke relevant MCP engineering tools."
    )

    return FeedbackDiagnosis(
        failure_type="INSUFFICIENT_RETRIEVAL" if is_retrieval_sufficient is False else "HALLUCINATION_DETECTED",
        missing_aspects=missing,
        critique=critique,
        remediation_hint=remediation,
        target_recommendations=["narrative_retriever", "tabular_retriever"],
    )


# ---------------------------------------------------------------------------
# Core Diagnostic Function
# ---------------------------------------------------------------------------
def diagnose_failure(
    user_query: str,
    cycle_count: int = 0,
    is_retrieval_sufficient: bool | None = None,
    missing_aspects: list[str] | None = None,
    retrieval_critique: str | None = None,
    is_grounded: bool | None = None,
    generation_metrics: dict[str, float] | None = None,
    final_response: str | None = None,
    synthesized_context: str | None = None,
    unfaithful_statements: list[str] | None = None,
    guardrail_critique: str | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> FeedbackDiagnosis:
    """
    Performs root-cause analysis on guardrail failures and generates a structured
    FeedbackDiagnosis containing an actionable remediation hint for the Planner.
    """
    t0 = time.perf_counter()
    logger.info(f"Formulating feedback diagnosis for cycle {cycle_count}: '{user_query}'")

    system_msg, user_msg = render_feedback_prompts(
        user_query=user_query,
        cycle_count=cycle_count,
        is_retrieval_sufficient=is_retrieval_sufficient,
        missing_aspects=missing_aspects,
        retrieval_critique=retrieval_critique,
        is_grounded=is_grounded,
        generation_metrics=generation_metrics,
        final_response=final_response,
        synthesized_context=synthesized_context,
        unfaithful_statements=unfaithful_statements,
        guardrail_critique=guardrail_critique,
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

        # Check thinking channel if content empty
        if not raw_text and msg_dict.get("thinking"):
            thinking_text = msg_dict.get("thinking", "").strip()
            s_idx = thinking_text.find("{")
            e_idx = thinking_text.rfind("}")
            if s_idx != -1 and e_idx != -1:
                raw_text = thinking_text[s_idx : e_idx + 1].strip()

        logger.debug(f"Raw Feedback response length={len(raw_text)} | snippet: {raw_text[:120]}")

        raw_text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()

        clean_json = raw_text
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
        if fence_match:
            clean_json = fence_match.group(1).strip()
        else:
            start_idx = raw_text.find("{")
            end_idx = raw_text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                clean_json = raw_text[start_idx : end_idx + 1].strip()

        try:
            parsed = json.loads(clean_json, strict=False)
        except json.JSONDecodeError:
            sanitized = re.sub(r'[\r\n]+', r'\\n', clean_json)
            sanitized = re.sub(r'\\n\s*("[\w_]+":)', r'\n\1', sanitized)
            sanitized = re.sub(r'\\n\s*([\{\}\[\]])', r'\n\1', sanitized)
            parsed = json.loads(sanitized, strict=False)

        diagnosis = FeedbackDiagnosis.model_validate(parsed)
        duration = time.perf_counter() - t0
        logger.success(
            f"Feedback diagnosis formulated in {duration:.2f}s | Type: {diagnosis.failure_type} | "
            f"Missing Aspects: {len(diagnosis.missing_aspects)}"
        )
        return diagnosis

    except (
        requests.RequestException,
        json.JSONDecodeError,
        ValidationError,
        KeyError,
        ValueError,
    ) as e:
        logger.error(f"Feedback LLM call failed: {e}. Activating deterministic fallback.")
        return _deterministic_feedback_fallback(
            user_query=user_query,
            missing_aspects=missing_aspects,
            retrieval_critique=retrieval_critique,
            is_retrieval_sufficient=is_retrieval_sufficient,
        )


# ---------------------------------------------------------------------------
# LangGraph Node Definition
# ---------------------------------------------------------------------------
def feedback_node(state: FeedbackAgentState) -> dict[str, Any]:
    """
    LangGraph node function for the Feedback Agent.

    Reads failure signals from state, diagnoses the root cause, formulates
    actionable remediation guidance, and increments cycle_count.

    Returns:
      - `feedback_diagnosis`: dict conforming to FeedbackDiagnosis
      - `cycle_count`: incremented integer
    """
    query = state.get("query") or state.get("user_query") or ""
    current_cycle = state.get("cycle_count", 0)

    diagnosis = diagnose_failure(
        user_query=query,
        cycle_count=current_cycle,
        is_retrieval_sufficient=state.get("is_retrieval_sufficient"),
        missing_aspects=state.get("missing_aspects"),
        retrieval_critique=state.get("retrieval_critique"),
        is_grounded=state.get("is_grounded"),
        generation_metrics=state.get("generation_metrics"),
        final_response=state.get("final_response"),
        synthesized_context=state.get("synthesized_context"),
        unfaithful_statements=state.get("unfaithful_statements"),
        guardrail_critique=state.get("guardrail_critique"),
    )

    new_cycle = current_cycle + 1
    logger.info(f"Feedback Agent incrementing cycle_count: {current_cycle} -> {new_cycle}")

    return {
        "feedback_diagnosis": diagnosis.model_dump(),
        "cycle_count": new_cycle,
        "error": None,
    }


__all__ = [
    "diagnose_failure",
    "feedback_node",
    "render_feedback_prompts",
]

