"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Feedback Agent State & Schemas
=============================================================================
Module: src.states.feedbackagent_state
Purpose: Defines Pydantic validation models and LangGraph state schemas for
         the Feedback Agent. Analyzes guardrail and grader failures, formulates
         actionable remediation guidance, increments the cycle counter, and
         loops back to the Planner Agent.
=============================================================================
"""

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Schemas for Structured Diagnostic Output
# ---------------------------------------------------------------------------
class FeedbackDiagnosis(BaseModel):
    """Structured diagnostic analysis produced by the Feedback Agent."""

    failure_type: Literal[
        "INSUFFICIENT_RETRIEVAL",
        "HALLUCINATION_DETECTED",
        "OFF_TOPIC_ANSWER",
    ] = Field(
        description="Categorical root cause of the evaluation failure."
    )
    missing_aspects: list[str] = Field(
        default_factory=list,
        description="List of specific engineering parameters, formulas, regulations, or facts missing.",
    )
    critique: str = Field(
        description="Detailed technical post-mortem explaining why the retrieval or generation failed quality gates.",
    )
    remediation_hint: str = Field(
        description="Concrete, actionable directive for the Planner Agent on how to re-decompose or filter in the next cycle.",
    )
    target_recommendations: list[str] = Field(
        default_factory=list,
        description="Optional list of recommended search targets, tools (e.g. MCP power calc), or Qdrant filters.",
    )


# ---------------------------------------------------------------------------
# LangGraph State Schema
# ---------------------------------------------------------------------------
class FeedbackAgentState(TypedDict, total=False):
    """
    LangGraph state channel for Feedback Agent execution.
    Acts as the self-correction reflection node looping failed runs back to Planner.
    """

    # Inputs from prior nodes
    query: str                                             # Current active query
    user_query: str | None                                 # Compatible alias
    is_retrieval_sufficient: bool | None                   # Verdict from Retrieval Grader
    missing_aspects: list[str] | None                      # Missing data points from Retrieval Grader
    retrieval_critique: str | None                         # Critique from Retrieval Grader
    is_grounded: bool | None                               # Verdict from Post-Gen Guardrail
    generation_metrics: dict[str, float] | None            # Faithfulness / hallucination scores
    unfaithful_statements: list[str] | None                # Unverified claims flagged by DeepEval Guardrail
    guardrail_critique: str | None                         # Critique from DeepEval Guardrail
    final_response: str | None                             # Draft response from Generator (if post-gen failed)
    synthesized_context: str | None                        # Synthesized context dossier
    cycle_count: int                                       # Current iteration count (0, 1, 2)

    # Outputs
    feedback_diagnosis: dict[str, Any]                     # Structured FeedbackDiagnosis dict for Planner
    error: str | None                                      # Error message if diagnostic fails


__all__ = [
    "FeedbackAgentState",
    "FeedbackDiagnosis",
]

