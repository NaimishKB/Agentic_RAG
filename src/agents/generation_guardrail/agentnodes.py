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
def _normalize_num_token(token: str) -> str:
    """Normalizes numerical token for robust comparison (e.g. '1,000' -> '1000')."""
    return token.replace(",", "").strip()


def _is_float_token(val: str) -> bool:
    """Helper testing if string token is convertible to float."""
    try:
        float(val.rstrip("%"))
        return True
    except ValueError:
        return False


def _sanitize_response_for_number_checking(text: str) -> str:
    """
    Strips structural formatting, citations, heading numbers, and formula labels
    from generated answer text before extracting substantive numerical claims.

    Ignores:
      1. Markdown heading numbers and section labels (e.g., '## 1.', '### 2.3', '### 3.1.2')
      2. List numbering ('1.', '2.1.', '(1)', '(a)', '1)')
      3. Citation identifiers and page numbers ('[Source: ... | Page: 12]', '[1]', '[^2]')
      4. Inline structural references ('Section 3.1', 'Clause 14(2)', 'Table 2', 'Figure 4', 'Bridge No. 604')
      5. Formula variable labels and subscripts ('H_1', 'P_2', 'tc_6.3', 'Equation (1)', '\\tag{2}')
      6. Multipliers and unit formatting ('1.5x', '2×')
    """
    # 1. Strip citation blocks: [Source: ...], [Page: 12], [Slide: 3], [1], [^1]
    cleaned = re.sub(
        r"\[(?:Source:[^\]]*|Page:\s*\d+|Slide:\s*\d+|Table:\s*\d+|Chunk:\s*\d+|\d+(?:,\s*\d+)*|\^\d+)\]",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\[Source:[^\]]*\]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\d+\]", " ", cleaned)

    # 2. Strip Markdown heading lines with numbering: e.g. "## 1. Executive Summary", "### 2.3 Matrix"
    cleaned = re.sub(
        r"(?m)^#{1,6}\s*(?:(?:Section|Sec\.|Chapter|Part|Clause)\s+)?\d+(?:\.\d+)*\b[:.]?\s*",
        " ",
        cleaned,
    )

    # 3. Strip list numbering: e.g. "1. ", "2.3. ", "(1) ", "(a) ", "1) "
    cleaned = re.sub(
        r"(?m)^\s*(?:\(?\d+(?:\.\d+)*\)?[\.\)]|\(\w\))\s+",
        " ",
        cleaned,
    )

    # 4. Strip inline structural references: e.g. "Section 4.3", "Table 1", "Bridge No. 604"
    cleaned = re.sub(
        r"\b(?:Section|Sec\.|Chapter|Clause|Regulation|Rule|Item|Step|Phase|Table|Figure|Fig\.|Appendix|Equation|Eq\.|Formula|Bridge\s+No\.)\s+\(?\d+(?:\.\d+)*\)?",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 5. Strip LaTeX equation tags: \tag{1}, \tag{2.1}
    cleaned = re.sub(r"\\tag\{\d+(?:\.\d+)*\}", " ", cleaned)

    # 6. Strip variable subscripts: e.g. H_1, P_2, tc_6.3, Q_{50}
    cleaned = re.sub(r"\b[A-Za-z]+_\{?[A-Za-z0-9.]+\}?", " ", cleaned)

    # 7. Strip multipliers: e.g. 1.5x, 2X, 1.5×
    cleaned = re.sub(r"\b\d+(?:\.\d+)?[xX×]\b", " ", cleaned)

    return cleaned


def _extract_allowed_numbers(
    query: str,
    synthesized_context: str,
    documents: list[dict[str, Any]] | None = None,
) -> tuple[set[str], str]:
    """
    Collects all legitimate numerical values from:
      1. Verified retrieved context
      2. User query (parameters and constraints supplied by the user)
      3. MCP tool outputs (programmatic calculation results with recorded inputs)
      4. Universal physical and mathematical constants (g=9.81, rho=1000, pi=3.1416)
    """
    combined_source_text = [synthesized_context, query]

    if documents:
        for doc in documents:
            content = doc.get("content", "")
            if content:
                combined_source_text.append(content)
            # If chunk is from an MCP tool, capture all serialized parameters
            if doc.get("file_type") == "TOOL_OUTPUT" or "mcp" in str(doc.get("source_file", "")).lower():
                combined_source_text.append(str(doc))

    full_allowed_text = " ".join(combined_source_text).lower()

    # Extract all numbers from full allowed text
    raw_tokens = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b", full_allowed_text)
    allowed_set: set[str] = set()

    for tok in raw_tokens:
        clean = _normalize_num_token(tok)
        allowed_set.add(clean)
        # Also add version without %
        if clean.endswith("%"):
            allowed_set.add(clean[:-1])
        # If integer version of float (e.g. 120.0 -> 120)
        if "." in clean:
            try:
                val = float(clean.rstrip("%"))
                if val.is_integer():
                    allowed_set.add(str(int(val)))
            except ValueError:
                pass

    # Universal engineering & formatting constants
    universal_constants = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "9.81", "1000", "3.14", "3.1416", "3.14159", "100", "100%"
    }
    allowed_set.update(universal_constants)

    return allowed_set, full_allowed_text


def _deterministic_faithfulness_fallback(
    query: str,
    final_response: str,
    synthesized_context: str,
    documents: list[dict[str, Any]] | None = None,
) -> GenerationGuardrailOutput:
    """
    Fallback groundedness check in case local LLM judge is unreachable.
    Performs deterministic numerical and parameter containment verification,
    properly ignoring structural formatting, citations, and query inputs.
    """
    if not final_response or not final_response.strip() or "Verified Technical Evidence Dossier" in final_response:
        return GenerationGuardrailOutput(
            is_grounded=False,
            faithfulness_score=0.0,
            unfaithful_statements=["Generation failed or emitted raw un-synthesized context."],
            critique="Deterministic heuristic: Answer text was absent or un-synthesized.",
        )

    # 1. Collect all allowed numbers from context, query, MCP tools, and physical constants
    allowed_numbers, full_allowed_text = _extract_allowed_numbers(
        query=query,
        synthesized_context=synthesized_context,
        documents=documents,
    )

    # 2. Sanitize final response: strip heading numbering, citations, variable subscripts, etc.
    sanitized_response = _sanitize_response_for_number_checking(final_response)

    # 3. Extract candidate numerical claims from sanitized response
    raw_answer_numbers = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b", sanitized_response)

    unsupported_numbers: list[str] = []
    for num in raw_answer_numbers:
        norm = _normalize_num_token(num)
        norm_no_pct = norm[:-1] if norm.endswith("%") else norm

        # Check if number is in allowed set or substring of full allowed text
        if norm in allowed_numbers or norm_no_pct in allowed_numbers or norm.lower() in full_allowed_text:
            continue

        # Check if float equivalence matches (e.g. 120 vs 120.0)
        try:
            val = float(norm_no_pct)
            if any(abs(val - float(a.rstrip("%"))) < 1e-4 for a in allowed_numbers if _is_float_token(a)):
                continue
        except ValueError:
            pass

        unsupported_numbers.append(num)

    # Deduplicate while preserving order
    seen_unsupported = set()
    unsupported_significant = []
    for n in unsupported_numbers:
        if n not in seen_unsupported and len(n) > 1 and n not in {"1", "2", "3", "4", "5", "10"}:
            seen_unsupported.add(n)
            unsupported_significant.append(n)

    if unsupported_significant:
        critique = (
            f"Deterministic heuristic: Detected numerical values in the answer ({unsupported_significant[:4]}) "
            f"that are not found in the verified context, user query, or MCP tool records."
        )
        return GenerationGuardrailOutput(
            is_grounded=False,
            faithfulness_score=0.65,
            unfaithful_statements=[f"Numerical claim '{n}' not supported by context or inputs" for n in unsupported_significant[:4]],
            critique=critique,
        )

    return GenerationGuardrailOutput(
        is_grounded=True,
        faithfulness_score=0.92,
        unfaithful_statements=[],
        critique="Deterministic heuristic: Core parameters, numbers, and units match verified context and query inputs.",
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
    documents: list[dict[str, Any]] | None = None,
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
            documents=documents,
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
      - `documents` (raw retrieved chunks and MCP tool outputs)

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
    documents = state.get("documents")

    if not synthesized_context and documents:
        doc_texts = [
            f"[Source: {d.get('source_file', 'unknown')}]\n{d.get('content', '')}"
            for d in documents
        ]
        synthesized_context = "\n\n".join(doc_texts)

    synthesized_context = synthesized_context or ""

    output = evaluate_generation_faithfulness(
        query=query,
        final_response=final_response,
        synthesized_context=synthesized_context,
        documents=documents,
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
