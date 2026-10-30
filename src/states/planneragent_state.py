"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Planner Agent State & Schemas
=============================================================================
Module: src.states.planneragent_state
Purpose: Defines the TypedDict state channel for LangGraph and Pydantic validation
         models for structured output from the Planner Agent.
=============================================================================
"""

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Output Validation Schemas
# ---------------------------------------------------------------------------
class SubQuerySpec(BaseModel):
    """Specification for an atomic retrieval task executed by a sub-agent."""
    sub_query_id: str = Field(
        description="Unique identifier for sub-query, e.g. sq_1"
    )
    query_text: str = Field(
        description="Dense technical search phrase optimized for semantic embedding"
    )
    target_filter: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filter dictionary for Qdrant (e.g. {'is_table': 'true'})"
    )
    rationale: str = Field(
        description="Justification for why this sub-query is necessary"
    )


class PlannerPlan(BaseModel):
    """Full structured output produced by the Planner Agent."""
    query_type: Literal["DIRECT_ANSWER", "OUT_OF_SCOPE", "RAG_REQUIRED"] = Field(
        description="High-level classification of user intent"
    )
    direct_response: str | None = Field(
        default=None,
        description="Direct message to the user if query_type is DIRECT_ANSWER or OUT_OF_SCOPE"
    )
    thought_process: str = Field(
        description="Chain of thought reasoning analyzing intent, required data, and decomposition"
    )
    sub_queries: list[SubQuerySpec] = Field(
        default_factory=list,
        description="List of targeted sub-queries to execute in parallel"
    )


# ---------------------------------------------------------------------------
# LangGraph State Schema
# ---------------------------------------------------------------------------
class PlannerAgentState(TypedDict, total=False):
    """
    LangGraph state channel for Planner Agent execution.
    """
    # Inputs
    query: str                                             # Current active user query
    original_query: str | None                             # Unmodified user input
    feedback_diagnosis: dict[str, Any] | None              # Diagnostic from Feedback Agent
    cycle_count: int                                       # Execution iteration count

    # Outputs
    routing_decision: str | None                           # 'direct_answer', 'out_of_scope', 'rag_retrieval'
    plan: list[str] | None                                 # Chain of thought plan summary
    sub_queries: list[dict[str, Any]] | None               # List of SubQuerySpec dictionaries
    final_response: str | None                             # Direct response text (if not RAG)
    error: str | None                                      # Error details if planning encountered an issue
