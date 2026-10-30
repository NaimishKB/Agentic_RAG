"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Orchestrator Agent State & Schemas
=============================================================================
Module: src.states.orchestratoragent_state
Purpose: Defines Pydantic data models for worker tasks, retrieved chunks,
         and the LangGraph OrchestratorAgentState TypedDict with reducers
         for dynamic worker fan-out/fan-in (Send API).
=============================================================================
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Reducer Functions for Parallel Fan-In (LangGraph Send API)
# ---------------------------------------------------------------------------
def merge_sub_agent_results(
    left: dict[str, list[dict[str, Any]]] | None,
    right: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Merges sub_agent_results dictionaries emitted by parallel worker nodes."""
    merged = dict(left or {})
    if right:
        for k, v in right.items():
            if k in merged:
                # Merge lists without duplicating chunk_ids
                existing_ids = {c["chunk_id"] for c in merged[k] if "chunk_id" in c}
                for item in v:
                    if item.get("chunk_id") not in existing_ids:
                        merged[k].append(item)
            else:
                merged[k] = list(v)
    return merged


# ---------------------------------------------------------------------------
# Pydantic Schemas for Chunks & Tasks
# ---------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    """Normalized document chunk retrieved from Qdrant vector database or MCP tool."""
    chunk_id: str = Field(description="Unique deterministic chunk ID")
    content: str = Field(description="Full text/table content with breadcrumbs")
    score: float = Field(description="Cosine similarity score or tool confidence")
    source_file: str = Field(description="Source document filename or MCP tool identifier")
    file_type: str = Field(description="File extension (PDF, PPT, DOCX, XLS) or 'TOOL_OUTPUT'")
    page_or_slide: int | None = Field(default=None, description="Page number or slide index")
    breadcrumb: str = Field(default="", description="Hierarchical document location")
    is_table: bool = Field(default=False, description="Whether this chunk is a Markdown table")
    token_count: int | None = Field(default=None, description="Estimated token count")
    worker_source: str = Field(default="", description="Name of worker that produced this chunk")


class WorkerTask(BaseModel):
    """Specification of an atomic retrieval or calculation task assigned to a worker."""
    sub_query_id: str = Field(description="ID matching Planner SubQuery, e.g. sq_1")
    query_text: str = Field(description="Refined search query or tool input prompt")
    worker_type: Literal["narrative_retriever", "tabular_retriever", "mcp_tool_worker"] = Field(
        description="Assigned worker: narrative_retriever (manuals), tabular_retriever (tables), or mcp_tool_worker (math/SCADA)"
    )
    target_filter: dict[str, Any] | None = Field(
        default=None,
        description="Qdrant metadata payload filter (e.g. {'is_table': True}) or tool args"
    )
    top_k: int = Field(default=4, description="Maximum number of points to retrieve")
    score_threshold: float = Field(default=0.35, description="Minimum cosine similarity cutoff")
    rationale: str = Field(default="", description="Reasoning for worker selection and search strategy")


class OrchestratorPlan(BaseModel):
    """Structured LLM output for assigning sub-queries to specialized retrieval workers."""
    coordination_strategy: str = Field(
        description="High-level explanation of how sub-queries are dispatched across workers"
    )
    tasks: list[WorkerTask] = Field(
        default_factory=list,
        description="List of specific worker tasks to dispatch dynamically"
    )


# ---------------------------------------------------------------------------
# Worker Input State for LangGraph Send()
# ---------------------------------------------------------------------------
class WorkerInputState(TypedDict):
    """Input payload dispatched to a single worker node via LangGraph Send()."""
    task: dict[str, Any]


# ---------------------------------------------------------------------------
# LangGraph State Schema with Reducers
# ---------------------------------------------------------------------------
class OrchestratorAgentState(TypedDict, total=False):
    """
    LangGraph state channel for Orchestrator Agent execution.
    Features channel reducers to cleanly aggregate results from dynamically spawned workers.
    """
    # Inputs
    user_query: str                                        # Original user query
    sub_queries: list[dict[str, Any]]                      # Sub-queries received from Planner Agent

    # Internal Orchestration
    tasks: list[dict[str, Any]] | None                     # Assigned worker tasks
    coordination_strategy: str | None                      # Coordination strategy description

    # Worker Outputs (Accumulated via Reducers across dynamic workers)
    sub_agent_results: Annotated[
        dict[str, list[dict[str, Any]]],
        merge_sub_agent_results
    ]
    all_retrieved_chunks: Annotated[
        list[dict[str, Any]],
        operator.add
    ]

    # Post-Aggregation Summary
    total_retrieved: int | None                            # Count of unique retrieved chunks
    error: str | None                                      # Error description if orchestration fails
