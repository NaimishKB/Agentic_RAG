"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Generation Guardrail State & Schemas
=============================================================================
Module: src.states.generation_guardrail_state
Purpose: Defines Pydantic validation models and LangGraph state schemas for
         the Post-Generation Faithfulness & Groundedness Guardrail.
         Leverages DeepEval's FaithfulnessMetric powered by local Ollama (qwen3.5:9b)
         to verify that every technical claim in the generated answer is strictly
         grounded in retrieved context, with zero external golden dataset required.
=============================================================================
"""

from typing import Any, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Schema for Guardrail Output
# ---------------------------------------------------------------------------
class GenerationGuardrailOutput(BaseModel):
    """Structured evaluation output produced by the Post-Generation Guardrail."""

    is_grounded: bool = Field(
        description="True if the generated answer meets or exceeds the faithfulness threshold (>= 0.90); "
        "False if unverified claims, incorrect parameters, or hallucinations are detected.",
    )
    faithfulness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Faithfulness score (0.0 to 1.0) measured by DeepEval FaithfulnessMetric against context.",
    )
    unfaithful_statements: list[str] = Field(
        default_factory=list,
        description="List of specific claims, formulas, or numbers in the answer that could NOT be verified in the context.",
    )
    critique: str = Field(
        description="Detailed technical critique explaining why the answer passed or detailing specific ungrounded statements.",
    )


# ---------------------------------------------------------------------------
# LangGraph State Schema
# ---------------------------------------------------------------------------
class GenerationGuardrailState(TypedDict, total=False):
    """
    LangGraph state channel for Generation Guardrail execution.
    Acts as the post-generation quality gate in the production LangGraph workflow.
    """

    # Inputs from Generator / prior nodes
    query: str                                             # Active user query
    user_query: str | None                                 # Compatible alias
    final_response: str                                    # Answer text drafted by Generator
    synthesized_context: str | None                        # Synthesized context dossier from Synthesizer
    documents: list[dict[str, Any]] | None                 # Raw retrieved chunks if context not synthesized
    cycle_count: int | None                                # Current loop iteration

    # Guardrail Outputs
    is_grounded: bool                                      # Post-generation gate decision
    generation_metrics: dict[str, float]                   # {"faithfulness": 0.95, ...}
    unfaithful_statements: list[str]                       # Specific hallucinated or unverified claims
    guardrail_critique: str                                # Critique forwarded to Feedback Agent on failure
    error: str | None                                      # Error message if evaluation fails


__all__ = [
    "GenerationGuardrailOutput",
    "GenerationGuardrailState",
]

