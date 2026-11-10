"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Feedback Agent Graph
=============================================================================
Module: src.agents.feedback.agentgraph
Purpose: Assembles and compiles the LangGraph StateGraph for the Feedback Agent.
         Provides standalone execution capability for root-cause diagnosis,
         remediation formulation, and cycle counter advancement.
=============================================================================
"""

import sys
from typing import Any

# Ensure standard UTF-8 stream handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langgraph.graph import END, StateGraph
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agents.feedback.agentnodes import feedback_node
from src.states.feedbackagent_state import FeedbackAgentState

console = Console(highlight=False)


def build_feedback_graph():
    """
    Constructs and compiles the Feedback Agent LangGraph workflow.

    Workflow:
      START ──> [feedback] ──> END
    """
    workflow = StateGraph(FeedbackAgentState)

    workflow.add_node("feedback", feedback_node)
    workflow.set_entry_point("feedback")
    workflow.add_edge("feedback", END)

    compiled_graph = workflow.compile()
    logger.info("Compiled Feedback Agent LangGraph StateGraph successfully.")
    return compiled_graph


def run_feedback(
    query: str,
    cycle_count: int = 0,
    is_retrieval_sufficient: bool | None = None,
    missing_aspects: list[str] | None = None,
    retrieval_critique: str | None = None,
    is_grounded: bool | None = None,
    generation_metrics: dict[str, float] | None = None,
    final_response: str | None = None,
    synthesized_context: str | None = None,
) -> FeedbackAgentState:
    """
    Executes the Feedback Agent graph to diagnose failure and formulate remediation.

    Args:
        query: User query.
        cycle_count: Current loop iteration.
        is_retrieval_sufficient: Retrieval Grader result.
        missing_aspects: Identified missing parameters.
        retrieval_critique: Retrieval Grader critique.
        is_grounded: Post-gen guardrail groundedness verdict.
        generation_metrics: Faithfulness / hallucination metrics.
        final_response: Draft generator response.
        synthesized_context: Evidence context available in current cycle.

    Returns:
        FeedbackAgentState containing updated feedback_diagnosis and incremented cycle_count.
    """
    graph = build_feedback_graph()

    initial_state: FeedbackAgentState = {
        "query": query,
        "user_query": query,
        "cycle_count": cycle_count,
        "is_retrieval_sufficient": is_retrieval_sufficient,
        "missing_aspects": missing_aspects or [],
        "retrieval_critique": retrieval_critique or "",
        "is_grounded": is_grounded,
        "generation_metrics": generation_metrics or {},
        "final_response": final_response,
        "synthesized_context": synthesized_context,
    }

    result = graph.invoke(initial_state)
    return result


def main():
    """CLI test runner verifying Feedback Agent diagnostic reflection on quality failures."""
    console.print("\n" + "=" * 80, style="bold cyan")
    console.print("[*] FEEDBACK AGENT (DIAGNOSTIC REFLECTION & REMEDIATION) VERIFICATION", style="bold green")
    console.print("=" * 80 + "\n", style="bold cyan")

    # -----------------------------------------------------------------------
    # Scenario 1: Pre-Gen Retrieval Failure (Missing Penstock Formulas)
    # -----------------------------------------------------------------------
    test_1_query = (
        "Calculate the economic penstock diameter and minimum steel wall thickness for a 150m net head "
        "and 3.5 m3/s design discharge, taking water hammer allowance into account."
    )
    test_1_missing = [
        "Sarkaria formula for economic penstock diameter calculation",
        "Penstock wall thickness formulas considering design pressure and corrosion allowance",
        "Water hammer pressure calculation methodology and allowable surge limits",
    ]
    test_1_critique = (
        "The provided context focuses solely on civil works (intake weir and desilting basin) and contains "
        "no hydraulic or structural engineering data, specifically missing the Sarkaria formula, wall thickness equations, "
        "and water hammer pressure allowances needed to answer the query."
    )
    test_1_context = (
        "Civil structures: Tyrolean bottom intake weir designed for boulder rivers. "
        "Desilting basin settles particles down to 0.2mm diameter."
    )

    console.print("[bold yellow]--> RUNNING SCENARIO 1: RETRIEVAL GRADER FAILURE (CYCLE 0 -> 1)[/bold yellow]")
    console.print(f"Query: [cyan]{test_1_query}[/cyan]\n")

    result_1 = run_feedback(
        query=test_1_query,
        cycle_count=0,
        is_retrieval_sufficient=False,
        missing_aspects=test_1_missing,
        retrieval_critique=test_1_critique,
        synthesized_context=test_1_context,
    )

    diag_1 = result_1.get("feedback_diagnosis", {})
    t1 = Table(title="Scenario 1 Diagnostic Output", show_header=True, header_style="bold magenta")
    t1.add_column("Field", style="cyan", width=24)
    t1.add_column("Value", style="white")
    t1.add_row("cycle_count (incremented)", str(result_1.get("cycle_count")))
    t1.add_row("failure_type", str(diag_1.get("failure_type")))
    t1.add_row("missing_aspects", str(diag_1.get("missing_aspects")))
    t1.add_row("critique", str(diag_1.get("critique")))
    t1.add_row("remediation_hint", str(diag_1.get("remediation_hint")))
    t1.add_row("target_recommendations", str(diag_1.get("target_recommendations")))
    console.print(t1)

    # -----------------------------------------------------------------------
    # Assertions / Verification Gate
    # -----------------------------------------------------------------------
    assert result_1.get("cycle_count") == 1, f"Expected cycle_count=1, got {result_1.get('cycle_count')}"
    assert diag_1.get("failure_type") == "INSUFFICIENT_RETRIEVAL", "Expected INSUFFICIENT_RETRIEVAL"
    assert len(diag_1.get("missing_aspects", [])) > 0, "Expected non-empty missing_aspects"
    assert len(diag_1.get("remediation_hint", "")) > 15, "Expected actionable remediation_hint"

    console.print(
        Panel(
            "[bold green][PASS] Feedback Agent verification tests PASSED successfully![/bold green]\n"
            "* Correctly categorized failure as INSUFFICIENT_RETRIEVAL\n"
            "* Incremented cycle_count (0 -> 1)\n"
            "* Formulated actionable remediation directive for Planner Agent.",
            title="Verification Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()

