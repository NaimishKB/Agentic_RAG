"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Generator Agent Nodes
=============================================================================
Module: src.agents.generator.agentnodes
Purpose: Implements the execution node for the Answer Generator Agent.
         Generates an authoritative, strictly grounded Markdown technical answer
         using local Ollama (qwen3.5:9b) and Jinja2 prompt templates.
=============================================================================
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import jinja2
import requests
from loguru import logger

from src.states.generatoragent_state import GeneratorAgentState

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL_NAME = os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:9b")
DEFAULT_TIMEOUT_S = int(os.getenv("GENERATOR_TIMEOUT_S", "600"))
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
DEFAULT_NUM_PREDICT = int(os.getenv("GENERATOR_MAX_TOKENS", "8192"))
DEFAULT_TEMPERATURE = 0.2
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
    "logs/generator_agent.log",
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


def render_generator_prompts(
    user_query: str,
    synthesized_context: str,
    executive_summary: str | None = None,
    sub_queries: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Renders system and user generation prompts using Jinja2 templates."""
    system_template = jinja_env.get_template("generator_agent_system.jinja2")
    user_template = jinja_env.get_template("generator_agent_user.jinja2")

    system_content = system_template.render()
    user_content = user_template.render(
        user_query=user_query,
        synthesized_context=synthesized_context,
        executive_summary=executive_summary or "",
        sub_queries=sub_queries or [],
    )
    return system_content, user_content


# ---------------------------------------------------------------------------
# Citation Extraction & Fallback Helpers
# ---------------------------------------------------------------------------
def extract_citations(text: str) -> list[str]:
    """Extracts unique source citations from inline citation tags in markdown."""
    pattern = r"\[Source:\s*([^\]|]+)(?:\|\s*[^\]]+)?\]"
    matches = re.findall(pattern, text, re.IGNORECASE)
    cleaned = sorted({m.strip() for m in matches if m.strip()})
    return cleaned


def _deterministic_generator_fallback(
    user_query: str,
    synthesized_context: str,
    executive_summary: str | None = None,
) -> tuple[str, list[str]]:
    """
    Fallback answer formulation in case Ollama service is unreachable.
    Returns empty string to signal generation failure to the guardrail and feedback loop.
    """
    return "", []


# ---------------------------------------------------------------------------
# Core Generation Function
# ---------------------------------------------------------------------------
def generate_answer(
    user_query: str,
    synthesized_context: str,
    executive_summary: str | None = None,
    sub_queries: list[dict[str, Any]] | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    timeout: int = DEFAULT_TIMEOUT_S,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[str, list[str], float]:
    """
    Generates a grounded technical response strictly based on synthesized context.

    Args:
        user_query: Primary engineering query.
        synthesized_context: Structured context dossier from Synthesizer.
        executive_summary: Optional summary from Synthesizer.
        sub_queries: List of planned sub-queries.
        ollama_url: Base URL for local Ollama.
        model_name: Model identifier (qwen3.5:9b).
        timeout: HTTP request timeout in seconds.
        temperature: Sampling temperature (default 0.2 for low variance).

    Returns:
        (final_response_markdown, list_of_citations, confidence_score)
    """
    t0 = time.perf_counter()
    logger.info(f"Generating grounded answer for: '{user_query}'")

    if not synthesized_context or not synthesized_context.strip():
        logger.warning("Empty synthesized context passed to Generator. Returning graceful notification.")
        empty_response = (
            "## Engineering Notice: Insufficient Data\n\n"
            "No verified technical documentation was found in the knowledge base to answer this query. "
            "Please upload relevant plant specifications or datasheets."
        )
        return empty_response, [], 0.0

    system_msg, user_msg = render_generator_prompts(
        user_query=user_query,
        synthesized_context=synthesized_context,
        executive_summary=executive_summary,
        sub_queries=sub_queries,
    )

    chat_url = f"{ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "options": {
            "temperature": temperature,
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

        # If content is empty but model emitted thinking, check if answer is in thinking
        if not raw_text and msg_dict.get("thinking"):
            thinking_text = msg_dict.get("thinking", "").strip()
            # If thinking contains markdown headers, extract answer portion
            h2_idx = thinking_text.find("##")
            if h2_idx != -1:
                raw_text = thinking_text[h2_idx:].strip()

        # Strip internal thinking tags if any
        final_answer = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()

        # If answer is empty or extremely short, signal generation failure
        if len(final_answer) < 30:
            logger.warning("LLM emitted unusually brief response. Signaling generation failure to feedback loop.")
            return "", [], 0.0

        citations = extract_citations(final_answer)
        # If model omitted explicit citation tags in text, harvest from synthesized context
        if not citations:
            citations = extract_citations(synthesized_context)

        duration = time.perf_counter() - t0
        logger.success(
            f"Answer generated in {duration:.2f}s | Length: {len(final_answer)} chars | Citations: {len(citations)}"
        )
        return final_answer, citations, 0.95

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.error(f"Answer generation LLM call failed: {e}. Signaling generation failure to feedback loop.")
        return "", [], 0.0


# ---------------------------------------------------------------------------
# LangGraph Node Definition
# ---------------------------------------------------------------------------
def generator_node(state: GeneratorAgentState) -> dict[str, Any]:
    """
    LangGraph node function for the Answer Generator Agent.

    Reads:
      - `query` or `user_query`
      - `synthesized_context` (falls back to documents chunks if empty)
      - `executive_summary`
      - `sub_queries`

    Writes:
      - `final_response`: str
      - `citations`: list[str]
      - `generation_confidence`: float
      - `error`: str | None
    """
    query = state.get("query") or state.get("user_query") or ""
    sub_queries = state.get("sub_queries")
    synthesized_context = state.get("synthesized_context")
    executive_summary = state.get("executive_summary")

    # If synthesized_context is missing, assemble from documents chunks
    if not synthesized_context and state.get("documents"):
        doc_texts = [
            f"[Source: {d.get('source_file', 'unknown')}]\n{d.get('content', '')}"
            for d in state["documents"]
        ]
        synthesized_context = "\n\n".join(doc_texts)

    synthesized_context = synthesized_context or ""

    response_text, citations, confidence = generate_answer(
        user_query=query,
        synthesized_context=synthesized_context,
        executive_summary=executive_summary,
        sub_queries=sub_queries,
    )

    return {
        "final_response": response_text,
        "citations": citations,
        "generation_confidence": confidence,
        "error": None,
    }


__all__ = [
    "extract_citations",
    "generate_answer",
    "generator_node",
    "render_generator_prompts",
]

