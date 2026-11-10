"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Retrieval Grader Graph
=============================================================================
Module: src.agents.retrieval_grader.agentgraph
Purpose: Assembles and compiles the LangGraph StateGraph for the Reference-Free
         Retrieval Sufficiency Grader (Production CRAG). Provides standalone
         execution capability for verifying retrieval sufficiency before answer
         generation.
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

from src.agents.retrieval_grader.agentnodes import grade_retrieval_node
from src.states.retrieval_grader_agent_state import RetrievalGraderState

console = Console(highlight=False)


def build_retrieval_grader_graph():
    """
    Constructs and compiles the Retrieval Grader LangGraph workflow.

    Workflow:
      START ──> [grade_retrieval] ──> END
    """
    workflow = StateGraph(RetrievalGraderState)

    # Add evaluation node
    workflow.add_node("grade_retrieval", grade_retrieval_node)

    # Flow
    workflow.set_entry_point("grade_retrieval")
    workflow.add_edge("grade_retrieval", END)

    compiled_graph = workflow.compile()
    logger.info("Compiled Retrieval Grader LangGraph StateGraph successfully.")
    return compiled_graph


def run_retrieval_grader(
    query: str,
    synthesized_context: str,
    sub_queries: list[dict[str, Any]] | None = None,
    cycle_count: int = 0,
) -> RetrievalGraderState:
    """
    Executes the Retrieval Grader graph for a given query and retrieved context.

    Args:
        query: Primary user engineering query.
        synthesized_context: Synthesized context dossier or chunk evidence.
        sub_queries: Optional list of planned sub-queries from Planner Agent.
        cycle_count: Current loop iteration.

    Returns:
        Final state dictionary containing is_retrieval_sufficient, missing_aspects, critique.
    """
    graph = build_retrieval_grader_graph()

    initial_state: RetrievalGraderState = {
        "query": query,
        "user_query": query,
        "synthesized_context": synthesized_context,
        "sub_queries": sub_queries or [],
        "cycle_count": cycle_count,
    }

    result = graph.invoke(initial_state)
    return result


def main():
    """CLI test runner verifying Retrieval Grader on both Sufficient and Insufficient contexts."""
    console.print("\n" + "=" * 80, style="bold cyan")
    console.print("[*] RETRIEVAL SUFFICIENCY GRADER (PRODUCTION CRAG) VERIFICATION", style="bold green")
    console.print("=" * 80 + "\n", style="bold cyan")

    # -----------------------------------------------------------------------
    # Test Case 1: SUFFICIENT CONTEXT (Turbine Selection & Silt Wear)
    # -----------------------------------------------------------------------
    test_1_query = (
        "Compare Francis vs Pelton turbines for 120m net head and 2.0 m3/s discharge "
        "operating in high quartz silt conditions."
    )
    test_1_sub_queries = [
        {
            "sub_query_id": "sq_1",
            "target_filter": "narrative_retriever",
            "query_text": "Turbine selection head ranges for 120m head Francis vs Pelton",
            "rationale": "Identify operating head envelopes and part-load efficiency curves.",
        },
        {
            "sub_query_id": "sq_2",
            "target_filter": "narrative_retriever",
            "query_text": "Silt erosion countermeasures Francis vs Pelton quartz silt",
            "rationale": "Assess abrasive wear resistance and maintenance replacement costs.",
        },
    ]
    test_1_context = (
        "### 1. Turbine Application Envelopes\n"
        "- **Francis Turbines**: Applicable for net heads from 40m to 350m. High peak efficiency (up to 94%), "
        "but efficiency drops significantly below 50% rated discharge [Source: SHP_Guidelines_Part2.pdf | Section: Turbines].\n"
        "- **Pelton Turbines**: Applicable for heads from 100m to 1000m+. Multi-jet configurations maintain high "
        "efficiency (>88%) across 20% to 110% rated flow [Source: CEA_Hydro_Standards.pdf | Section: Electro-Mechanical].\n\n"
        "### 2. Quartz Silt Wear & Maintenance Longevity\n"
        "- In rivers with high sediment and quartz content (>0.2mm), Pelton turbines are strongly recommended "
        "because runner buckets and spear valves can be coated with tungsten carbide (WC-Co-Cr) and replaced easily. "
        "Conversely, Francis runners and guide vanes suffer severe labyrinth seal clearance degradation, leading to rapid "
        "efficiency loss and complex dismantling [Source: Silt_Handling_Manual.pdf | Section: Hydro-Abrasive Wear]."
    )

    console.print("[bold yellow]--> RUNNING TEST CASE 1: SUFFICIENT CONTEXT[/bold yellow]")
    console.print(f"Query: [cyan]{test_1_query}[/cyan]\n")

    result_1 = run_retrieval_grader(
        query=test_1_query,
        synthesized_context=test_1_context,
        sub_queries=test_1_sub_queries,
    )

    t1 = Table(title="Test Case 1 Evaluation Result", show_header=True, header_style="bold magenta")
    t1.add_column("Field", style="cyan", width=26)
    t1.add_column("Value", style="white")
    t1.add_row("is_retrieval_sufficient", str(result_1.get("is_retrieval_sufficient")))
    t1.add_row("retrieval_confidence", f"{result_1.get('retrieval_confidence', 0.0):.2f}")
    t1.add_row("missing_aspects", str(result_1.get("missing_aspects")))
    t1.add_row("retrieval_critique", str(result_1.get("retrieval_critique")))
    console.print(t1)

    # -----------------------------------------------------------------------
    # Test Case 2: INSUFFICIENT CONTEXT (Missing Penstock Formulas & Sizing)
    # -----------------------------------------------------------------------
    test_2_query = (
        "Calculate the economic penstock diameter and minimum steel wall thickness for a 150m net head "
        "and 3.5 m3/s design discharge, taking water hammer allowance into account."
    )
    test_2_sub_queries = [
        {
            "sub_query_id": "sq_1",
            "target_filter": "mcp_worker",
            "query_text": "Calculate economic penstock diameter using Sarkaria formula for Q=3.5 m3/s",
            "rationale": "Compute economic internal diameter and check flow velocity limits (2.5 - 5.0 m/s).",
        },
        {
            "sub_query_id": "sq_2",
            "target_filter": "narrative_retriever",
            "query_text": "Penstock steel plate minimum thickness formula and water hammer pressure for 150m head",
            "rationale": "Determine shell thickness considering design pressure and corrosion allowance.",
        },
    ]
    test_2_insufficient_context = (
        "### 1. Intake Civil Structures\n"
        "The diversion weir is a trench weir (Tyrolean type) designed for boulder rivers. "
        "Intake trash rack spacing is 50mm to prevent large debris entry [Source: Civil_Design_Manual.pdf | Section: Intake].\n\n"
        "### 2. Desilting Basin\n"
        "A hopper-type desilting basin settles particles down to 0.2mm diameter with a horizontal settling velocity "
        "of 0.25 m/s. Continuous flushing consumes approximately 15% of the river discharge [Source: Civil_Design_Manual.pdf | Section: Desilting]."
    )

    console.print("\n[bold yellow]--> RUNNING TEST CASE 2: INSUFFICIENT CONTEXT[/bold yellow]")
    console.print(f"Query: [cyan]{test_2_query}[/cyan]\n")

    result_2 = run_retrieval_grader(
        query=test_2_query,
        synthesized_context=test_2_insufficient_context,
        sub_queries=test_2_sub_queries,
    )

    t2 = Table(title="Test Case 2 Evaluation Result", show_header=True, header_style="bold magenta")
    t2.add_column("Field", style="cyan", width=26)
    t2.add_column("Value", style="white")
    t2.add_row("is_retrieval_sufficient", str(result_2.get("is_retrieval_sufficient")))
    t2.add_row("retrieval_confidence", f"{result_2.get('retrieval_confidence', 0.0):.2f}")
    t2.add_row("missing_aspects", str(result_2.get("missing_aspects")))
    t2.add_row("retrieval_critique", str(result_2.get("retrieval_critique")))
    console.print(t2)

    # -----------------------------------------------------------------------
    # Assertions / Verification Gate
    # -----------------------------------------------------------------------
    assert result_1.get("is_retrieval_sufficient") is True, "Test Case 1 failed: Expected sufficient=True"
    assert result_2.get("is_retrieval_sufficient") is False, "Test Case 2 failed: Expected sufficient=False"
    assert len(result_2.get("missing_aspects", [])) > 0, "Test Case 2 failed: Expected non-empty missing_aspects"

    console.print(
        Panel(
            "[bold green][PASS] All Retrieval Sufficiency Grader verification tests PASSED successfully![/bold green]\n"
            "* Sufficient context passed with is_sufficient=True\n"
            "* Insufficient context accurately flagged with is_sufficient=False and actionable missing aspects.",
            title="Verification Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
