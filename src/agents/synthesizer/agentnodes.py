"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Synthesizer Agent Nodes
=============================================================================
Module: src.agents.synthesizer.agentnodes
Purpose: Implements the execution node for the Synthesizer Agent. Performs
         deterministic deduplication, fair-share sub-query balancing, MCP
         tool output integration, and LLM-assisted context synthesis using
         Ollama qwen3.5:9b and Jinja2 templates.
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

from src.states.synthesizeragent_state import SynthesizerAgentState, SynthesizerOutput

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL_NAME = os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:9b")
DEFAULT_TIMEOUT_S = 180
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 4096
PROMPTS_DIR = Path("src/prompts")
MAX_CONTEXT_CHUNKS = 8
MIN_CHUNKS_PER_SUBQUERY = 2

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/synthesizer_nodes.log",
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


def render_synthesizer_prompts(
    user_query: str,
    sub_queries: list[dict[str, Any]] | None,
    evidence_chunks: list[dict[str, Any]],
    mcp_evidence: list[dict[str, Any]],
) -> tuple[str, str]:
    """Renders system and user prompts using Jinja2 templates."""
    system_template = jinja_env.get_template("synthesizer_agent_system.jinja2")
    user_template = jinja_env.get_template("synthesizer_agent_user.jinja2")

    system_content = system_template.render()
    user_content = user_template.render(
        user_query=user_query,
        sub_queries=sub_queries or [],
        evidence_chunks=evidence_chunks,
        mcp_evidence=mcp_evidence,
    )
    return system_content, user_content


# ---------------------------------------------------------------------------
# Deterministic Chunk Deduplication & Sub-Query Balancing
# ---------------------------------------------------------------------------
def deduplicate_and_balance_chunks(
    sub_agent_results: dict[str, list[dict[str, Any]]] | None,
    all_retrieved_chunks: list[dict[str, Any]] | None,
    max_total_chunks: int = MAX_CONTEXT_CHUNKS,
    min_per_subquery: int = MIN_CHUNKS_PER_SUBQUERY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Fuses, deduplicates by chunk_id, and balances chunks across sub-queries
    to prevent a single broad sub-query from starving specialized tabular queries.
    Also separates MCP tool outputs from document text/table chunks.

    Returns:
        (balanced_doc_chunks, mcp_tool_chunks)
    """
    mcp_chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # 1. Extract MCP tool outputs separately
    raw_all = all_retrieved_chunks or []
    if sub_agent_results:
        for chunks in sub_agent_results.values():
            raw_all.extend(chunks)

    cleaned_docs: dict[str, dict[str, Any]] = {}
    for chunk in raw_all:
        c_id = chunk.get("chunk_id", "")
        file_type = str(chunk.get("file_type", "")).upper()

        if file_type == "TOOL_OUTPUT" or "TOOL:" in chunk.get("source_file", "").upper():
            if c_id not in seen_ids:
                seen_ids.add(c_id)
                mcp_chunks.append(chunk)
            continue

        if c_id not in cleaned_docs or chunk.get("score", 0.0) > cleaned_docs[c_id].get("score", 0.0):
            cleaned_docs[c_id] = chunk

    # 2. Fair-share allocation per sub_query_id
    balanced_doc_chunks: list[dict[str, Any]] = []
    allocated_ids: set[str] = set()

    if sub_agent_results and len(sub_agent_results) > 1:
        # Guarantee top-N per sub-query
        for sq_chunks in sub_agent_results.values():
            valid_sq_chunks = [
                c for c in sq_chunks
                if c.get("file_type", "").upper() != "TOOL_OUTPUT"
            ]
            sorted_sq = sorted(valid_sq_chunks, key=lambda x: x.get("score", 0.0), reverse=True)
            added_for_sq = 0
            for c in sorted_sq:
                cid = c.get("chunk_id")
                if cid and cid not in allocated_ids:
                    allocated_ids.add(cid)
                    balanced_doc_chunks.append(c)
                    added_for_sq += 1
                    if added_for_sq >= min_per_subquery:
                        break

    # 3. Fill remaining quota from the global pool sorted by score
    all_sorted = sorted(
        cleaned_docs.values(),
        key=lambda x: x.get("score", 0.0),
        reverse=True,
    )
    for c in all_sorted:
        cid = c.get("chunk_id")
        if cid and cid not in allocated_ids:
            allocated_ids.add(cid)
            balanced_doc_chunks.append(c)
            if len(balanced_doc_chunks) >= max_total_chunks:
                break

    logger.info(
        f"Deduplicated chunks: {len(balanced_doc_chunks)} document chunks, {len(mcp_chunks)} MCP tool outputs"
    )
    return balanced_doc_chunks, mcp_chunks


# ---------------------------------------------------------------------------
# Deterministic Fallback Context Assembler
# ---------------------------------------------------------------------------
def build_fallback_context(
    user_query: str,
    doc_chunks: list[dict[str, Any]],
    mcp_chunks: list[dict[str, Any]],
) -> SynthesizerOutput:
    """
    Constructs a structured, citation-anchored context dossier without an LLM call.
    Used as an emergency fallback if the LLM times out or produces invalid JSON.
    """
    logger.warning("Building deterministic fallback context dossier")
    sections: list[str] = []
    citations: list[str] = []

    # 1. MCP section
    if mcp_chunks:
        mcp_lines = ["### Programmatic Calculations & Sensor Telemetry"]
        for c in mcp_chunks:
            source = c.get("source_file", "MCP Tool")
            citations.append(source)
            mcp_lines.append(f"- **{source}**: {c.get('content')}")
        sections.append("\n".join(mcp_lines))

    # 2. Document Chunks section
    if doc_chunks:
        doc_lines = ["### Verified Hydro Technical Documentation"]
        for i, c in enumerate(doc_chunks, start=1):
            src = c.get("source_file", "Unknown")
            sec = c.get("breadcrumb", "General")
            score = c.get("score", 0.0)
            citations.append(src)
            citation_tag = f"[Source: {src} | Section: {sec} | Score: {score:.2f}]"
            doc_lines.append(f"\n#### Reference {i} {citation_tag}\n{c.get('content')}")
        sections.append("\n".join(doc_lines))

    synthesized_text = "\n\n".join(sections)
    unique_citations = sorted(set(citations))

    return SynthesizerOutput(
        executive_summary=(
            f"Assembled evidence for query '{user_query}' covering {len(doc_chunks)} verified technical chunks "
            f"and {len(mcp_chunks)} programmatic outputs across {len(unique_citations)} unique sources."
        ),
        synthesized_context=synthesized_text,
        conflicts_detected=[],
        citations_used=unique_citations,
        coverage_score=0.9 if doc_chunks else 0.4,
    )


# ---------------------------------------------------------------------------
# Core Synthesis Function
# ---------------------------------------------------------------------------
def synthesize_context(
    user_query: str,
    sub_queries: list[dict[str, Any]] | None = None,
    sub_agent_results: dict[str, list[dict[str, Any]]] | None = None,
    all_retrieved_chunks: list[dict[str, Any]] | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> tuple[SynthesizerOutput, list[dict[str, Any]]]:
    """
    Deduplicates retrieved chunks, renders Jinja2 synthesis prompts, queries Ollama,
    and returns a validated SynthesizerOutput alongside the deduplicated chunks.
    """
    t0 = time.perf_counter()
    logger.info(f"Synthesizing context for query: '{user_query}'")

    # Step 1: Deduplication & Fair-share balancing
    balanced_docs, mcp_outputs = deduplicate_and_balance_chunks(
        sub_agent_results=sub_agent_results,
        all_retrieved_chunks=all_retrieved_chunks,
    )

    if not balanced_docs and not mcp_outputs:
        logger.warning("No retrieved chunks or MCP outputs found to synthesize.")
        fallback = SynthesizerOutput(
            executive_summary="No technical documents or calculations were retrieved for this query.",
            synthesized_context="No relevant engineering evidence found in the knowledge base.",
            conflicts_detected=[],
            citations_used=[],
            coverage_score=0.0,
        )
        return fallback, []

    # Step 2: Render Jinja2 Prompts
    system_msg, user_msg = render_synthesizer_prompts(
        user_query=user_query,
        sub_queries=sub_queries,
        evidence_chunks=balanced_docs,
        mcp_evidence=mcp_outputs,
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

        # If content is empty but model emitted thinking, check if JSON is inside thinking
        if not raw_text and msg_dict.get("thinking"):
            thinking_text = msg_dict.get("thinking", "").strip()
            s_idx = thinking_text.find("{")
            e_idx = thinking_text.rfind("}")
            if s_idx != -1 and e_idx != -1:
                raw_text = thinking_text[s_idx : e_idx + 1].strip()

        logger.info(f"Raw Ollama response length={len(raw_text)} | snippet: {raw_text[:150]}")

        # Strip possible reasoning/thinking tags
        raw_text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()

        # Extract & validate JSON
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
            # Try sanitizing unescaped literal newlines in multiline strings
            # Only replace newlines that occur within string quotes
            sanitized = re.sub(r'[\r\n]+', r'\\n', clean_json)
            # Restore newlines around keys/syntax
            sanitized = re.sub(r'\\n\s*("[\w_]+":)', r'\n\1', sanitized)
            sanitized = re.sub(r'\\n\s*([\{\}\[\]])', r'\n\1', sanitized)
            parsed = json.loads(sanitized, strict=False)

        output = SynthesizerOutput.model_validate(parsed)
        if "\\n" in output.synthesized_context:
            output.synthesized_context = output.synthesized_context.replace("\\n", "\n")
        duration = time.perf_counter() - t0
        logger.success(
            f"Synthesis completed in {duration:.2f}s | Citations: {len(output.citations_used)} | "
            f"Conflicts: {len(output.conflicts_detected)} | Coverage: {output.coverage_score:.2f}"
        )
        return output, balanced_docs + mcp_outputs

    except (
        requests.RequestException,
        json.JSONDecodeError,
        ValidationError,
        KeyError,
        ValueError,
    ) as e:
        logger.error(f"Synthesis LLM call failed: {e}. Activating deterministic fallback.")
        fallback = build_fallback_context(
            user_query=user_query,
            doc_chunks=balanced_docs,
            mcp_chunks=mcp_outputs,
        )
        return fallback, balanced_docs + mcp_outputs


# ---------------------------------------------------------------------------
# LangGraph Node Definition
# ---------------------------------------------------------------------------
def synthesizer_node(state: SynthesizerAgentState) -> dict[str, Any]:
    """
    LangGraph execution node for the Synthesizer Agent.

    Reads:
      - state["query"] / state.get("user_query")
      - state.get("sub_queries")
      - state.get("sub_agent_results")
      - state.get("all_retrieved_chunks")

    Writes to state:
      - documents (list of deduplicated and balanced RetrievedChunk dicts)
      - synthesized_context (formatted Markdown evidence dossier)
      - executive_summary (high-level engineering briefing)
      - conflicts_detected (list of contradictory points)
      - citations (list of sources)
      - coverage_score (coverage metric 0.0 to 1.0)
    """
    query = state.get("query") or state.get("user_query", "")
    sub_queries = state.get("sub_queries")
    sub_agent_results = state.get("sub_agent_results")
    all_chunks = state.get("all_retrieved_chunks")

    logger.info(f"[NODE: synthesizer] Executing synthesis for: '{query}'")

    output, final_chunks = synthesize_context(
        user_query=query,
        sub_queries=sub_queries,
        sub_agent_results=sub_agent_results,
        all_retrieved_chunks=all_chunks,
    )

    return {
        "documents": final_chunks,
        "synthesized_context": output.synthesized_context,
        "executive_summary": output.executive_summary,
        "conflicts_detected": output.conflicts_detected,
        "citations": output.citations_used,
        "coverage_score": output.coverage_score,
    }


__all__ = [
    "build_fallback_context",
    "deduplicate_and_balance_chunks",
    "render_synthesizer_prompts",
    "synthesize_context",
    "synthesizer_node",
]
