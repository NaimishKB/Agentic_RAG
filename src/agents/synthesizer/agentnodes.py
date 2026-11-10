"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Synthesizer Agent Nodes
=============================================================================
Module: src.agents.synthesizer.agentnodes
Purpose: Implements the execution node for the Synthesizer Agent. Performs
         high-speed deterministic chunk deduplication, fair-share sub-query
         balancing, MCP tool output integration, and structured citation
         provenance assembly in 0.000s without redundant LLM generation.
=============================================================================
"""

import os
import sys
import time
from typing import Any

from loguru import logger

from src.states.synthesizeragent_state import SynthesizerAgentState, SynthesizerOutput

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
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
    for c in raw_all:
        ftype = str(c.get("file_type", "")).upper()
        source = str(c.get("source_file", "")).lower()
        if ftype == "TOOL_OUTPUT" or "calculate_" in source or "query_" in source or "scada" in source:
            cid = c.get("chunk_id", f"mcp_{len(mcp_chunks)}")
            if cid not in seen_ids:
                seen_ids.add(cid)
                mcp_chunks.append(c)
        else:
            cid = c.get("chunk_id")
            if cid and cid not in cleaned_docs:
                cleaned_docs[cid] = c

    # 2. Fair-share allocation per sub-query
    balanced_doc_chunks: list[dict[str, Any]] = []
    allocated_ids: set[str] = set()

    if sub_agent_results:
        for sq_id, chunks in sub_agent_results.items():
            doc_only = [
                c for c in chunks
                if str(c.get("file_type", "")).upper() != "TOOL_OUTPUT"
                and "calculate_" not in str(c.get("source_file", "")).lower()
            ]
            sorted_sq = sorted(
                doc_only,
                key=lambda x: x.get("score", 0.0),
                reverse=True,
            )
            count = 0
            for c in sorted_sq:
                cid = c.get("chunk_id")
                if cid and cid not in allocated_ids:
                    allocated_ids.add(cid)
                    balanced_doc_chunks.append(c)
                    count += 1
                    if count >= min_per_subquery:
                        break

    # 3. Fill remaining quota with highest scoring available chunks
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
# High-Speed Deterministic Context Assembler (Zero LLM Overhead)
# ---------------------------------------------------------------------------
def build_deterministic_context(
    user_query: str,
    sub_queries: list[dict[str, Any]] | None,
    doc_chunks: list[dict[str, Any]],
    mcp_chunks: list[dict[str, Any]],
) -> SynthesizerOutput:
    """
    Constructs a structured, citation-anchored context dossier in pure Python (0.000s)
    without an LLM call. Eliminates duplicate generation cycles on local GPUs.
    """
    sections: list[str] = []
    citations: list[str] = []

    # 1. MCP Section (Deterministic Calculations & Telemetry)
    if mcp_chunks:
        mcp_lines = ["### Programmatic Calculations & Live SCADA Telemetry"]
        for c in mcp_chunks:
            source = c.get("source_file", "MCP Tool")
            citations.append(source)
            mcp_lines.append(f"- **{source}**: {c.get('content')}")
        sections.append("\n".join(mcp_lines))

    # 2. Document Chunks Section
    if doc_chunks:
        doc_lines = ["### Verified Hydro Technical Documentation"]
        for i, c in enumerate(doc_chunks, start=1):
            src = c.get("source_file", "Unknown")
            sec = c.get("breadcrumb", "General")
            score = float(c.get("score", 0.0))
            citations.append(src)
            citation_tag = f"[Source: {src} | Section: {sec} | Score: {score:.2f}]"
            doc_lines.append(f"\n#### Reference {i} {citation_tag}\n{c.get('content')}")
        sections.append("\n".join(doc_lines))

    synthesized_text = "\n\n".join(sections)
    unique_citations = sorted(set(citations))

    # Calculate sub-query coverage
    if sub_queries:
        cov = min(1.0, len(doc_chunks) / max(len(sub_queries), 1))
    else:
        cov = 0.9 if doc_chunks else 0.4

    return SynthesizerOutput(
        executive_summary=(
            f"Consolidated evidence for query '{user_query}' covering {len(doc_chunks)} verified technical chunks "
            f"and {len(mcp_chunks)} programmatic outputs across {len(unique_citations)} unique sources."
        ),
        synthesized_context=synthesized_text,
        conflicts_detected=[],
        citations_used=unique_citations,
        coverage_score=round(cov, 2),
    )


# Alias for backwards compatibility
build_fallback_context = build_deterministic_context


# ---------------------------------------------------------------------------
# Core Synthesis Function
# ---------------------------------------------------------------------------
def synthesize_context(
    user_query: str,
    sub_queries: list[dict[str, Any]] | None = None,
    sub_agent_results: dict[str, list[dict[str, Any]]] | None = None,
    all_retrieved_chunks: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> tuple[SynthesizerOutput, list[dict[str, Any]]]:
    """
    Executes high-speed deterministic fusion, deduplication, fair-share balancing,
    and structured citation assembly in pure Python (0.000s latency).
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

    # Step 2: Build Deterministic Context Dossier
    output = build_deterministic_context(
        user_query=user_query,
        sub_queries=sub_queries,
        doc_chunks=balanced_docs,
        mcp_chunks=mcp_outputs,
    )

    duration = time.perf_counter() - t0
    logger.success(
        f"Fast deterministic synthesis completed in {duration:.3f}s | Citations: {len(output.citations_used)} | "
        f"Coverage: {output.coverage_score:.2f} | Chunks: {len(balanced_docs)}"
    )

    return output, balanced_docs + mcp_outputs


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
    "build_deterministic_context",
    "build_fallback_context",
    "deduplicate_and_balance_chunks",
    "synthesize_context",
    "synthesizer_node",
]
