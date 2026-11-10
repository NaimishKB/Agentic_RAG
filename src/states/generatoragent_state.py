"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Generator Agent State & Schemas
=============================================================================
Module: src.states.generatoragent_state
Purpose: Defines Pydantic validation models and LangGraph state schemas for
         the Answer Generator Agent. Generates authoritative, strictly grounded
         technical engineering answers using local Ollama (qwen3.5:9b).
=============================================================================
"""

from typing import Any, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Output Validation Schema
# ---------------------------------------------------------------------------
class GeneratorOutput(BaseModel):
    """Structured output representation for the Generator Agent."""

    final_response: str = Field(
        description="Comprehensive, authoritative markdown engineering answer strictly grounded in the provided context, "
        "featuring structured headers, numerical parameters, and inline [Source: ...] citations.",
    )
    citations_used: list[str] = Field(
        default_factory=list,
        description="List of distinct source files, standards, or MCP tools referenced in the generated answer.",
    )
    confidence_score: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0) reflecting how completely the evidence grounds the generated answer.",
    )


# ---------------------------------------------------------------------------
# LangGraph State Schema
# ---------------------------------------------------------------------------
class GeneratorAgentState(TypedDict, total=False):
    """
    LangGraph state channel for Generator Agent execution.
    Can execute standalone or as the generation node within overarching AgentState.
    """

    # Inputs from prior nodes
    query: str                                             # Current active query
    user_query: str | None                                 # Compatible alias
    synthesized_context: str | None                        # Clean context dossier from Synthesizer
    executive_summary: str | None                          # High-level briefing from Synthesizer
    documents: list[dict[str, Any]] | None                 # Balanced retrieved chunks
    sub_queries: list[dict[str, Any]] | None               # SubQuery specs from Planner
    cycle_count: int | None                                # Current loop iteration

    # Outputs
    final_response: str                                    # Final verified answer text
    citations: list[str]                                   # Distinct citations used
    generation_confidence: float                           # Confidence score
    error: str | None                                      # Error message if generation fails


__all__ = [
    "GeneratorAgentState",
    "GeneratorOutput",
]

