"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Generation Guardrail Graph
=============================================================================
Module: src.agents.generation_guardrail.agentgraph
Purpose: Assembles and compiles the LangGraph StateGraph for the Post-Generation
         Faithfulness Guardrail. Provides standalone execution capability for
         validating that generated answers are strictly grounded in context.
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

from src.agents.generation_guardrail.agentnodes import generation_guardrail_node
from src.states.generation_guardrail_state import GenerationGuardrailState

console = Console(highlight=False)


def build_generation_guardrail_graph():
    """
    Constructs and compiles the Generation Guardrail LangGraph workflow.

    Workflow:
      START ──> [generation_guardrail] ──> END
    """
    workflow = StateGraph(GenerationGuardrailState)

    workflow.add_node("generation_guardrail", generation_guardrail_node)
    workflow.set_entry_point("generation_guardrail")
    workflow.add_edge("generation_guardrail", END)

    compiled_graph = workflow.compile()
    logger.info("Compiled Generation Guardrail LangGraph StateGraph successfully.")
    return compiled_graph


def run_generation_guardrail(
    query: str,
    final_response: str,
    synthesized_context: str,
    cycle_count: int = 0,
) -> GenerationGuardrailState:
    """
    Executes the Generation Guardrail graph to evaluate answer faithfulness.

    Args:
        query: Primary user engineering query.
        final_response: Generated answer text.
        synthesized_context: Synthesized evidence dossier from Synthesizer.
        cycle_count: Current loop iteration.

    Returns:
        GenerationGuardrailState containing is_grounded, generation_metrics, unfaithful_statements.
    """
    graph = build_generation_guardrail_graph()

    initial_state: GenerationGuardrailState = {
        "query": query,
        "user_query": query,
        "final_response": final_response,
        "synthesized_context": synthesized_context,
        "cycle_count": cycle_count,
    }

    result = graph.invoke(initial_state)
    return result


def main():
    """CLI test runner verifying Post-Gen Guardrail on both Faithful and Hallucinated answers."""
    console.print("\n" + "=" * 80, style="bold cyan")
    console.print("[*] POST-GEN FAITHFULNESS GUARDRAIL (DEEPEVAL) VERIFICATION RUN", style="bold green")
    console.print("=" * 80 + "\n", style="bold cyan")

    shared_query = (
        "Compare Francis vs Pelton turbines for 120m net head and 2.0 m3/s discharge "
        "operating in high quartz silt conditions, and provide power potential."
    )
    shared_context = (
        "### 1. Programmatic Power Calculation\n"
        "- Calculated Power: 2,072.76 kW (2.07 MW) [Source: MCP Tool | Tool: calculate_hydro_power]\n"
        "- Input Parameters: Net Head H = 120m, Design Discharge Q = 2.0 m3/s, Overall Efficiency = 0.88 "
        "[Source: MCP Tool | Tool: calculate_hydro_power]\n\n"
        "### 2. Turbine Operating Head Ranges & Efficiency\n"
        "- Francis Turbines: Applicable for net heads from 40m to 350m. High peak efficiency (up to 94%), "
        "but efficiency drops significantly below 50% rated discharge [Source: SHP_Guidelines_Part2.pdf | Section: Turbines].\n"
        "- Pelton Turbines: Applicable for net heads from 100m to 1000m+. Multi-jet configurations maintain high "
        "efficiency (>88%) across 20% to 110% rated flow [Source: CEA_Hydro_Standards.pdf | Section: Electro-Mechanical].\n\n"
        "### 3. Quartz Silt Wear & Maintenance Longevity\n"
        "- In rivers with high sediment and quartz content (>0.2mm), Pelton turbines are strongly recommended "
        "because runner buckets and spear valves can be coated with tungsten carbide (WC-Co-Cr) via HVOF and replaced easily. "
        "Conversely, Francis runners and guide vanes suffer severe labyrinth seal clearance degradation, leading to rapid "
        "efficiency loss and complex plant dismantling [Source: Silt_Handling_Manual.pdf | Section: Hydro-Abrasive Wear]."
    )

    # -----------------------------------------------------------------------
    # Test Case 1: FAITHFUL / GROUNDED ANSWER
    # -----------------------------------------------------------------------
    faithful_answer = (
        "## 1. Executive Summary\n"
        "At 120m net head and 2.0 m3/s discharge, the calculated hydro power potential is 2.07 MW (2,072.76 kW) "
        "[Source: MCP Tool | Tool: calculate_hydro_power]. While both Francis and Pelton turbines operate within 120m head, "
        "Pelton is strongly recommended due to superior quartz silt abrasion resistance [Source: Silt_Handling_Manual.pdf].\n\n"
        "## 2. Technical Comparison\n"
        "- Francis peak efficiency is up to 94% (heads 40m-350m), but efficiency degrades below 50% flow [Source: SHP_Guidelines_Part2.pdf].\n"
        "- Pelton multi-jet maintains >88% efficiency from 20% to 110% flow (heads 100m-1000m+) [Source: CEA_Hydro_Standards.pdf].\n"
        "- In quartz silt (>0.2mm), Pelton buckets are coated with tungsten carbide (WC-Co-Cr) via HVOF and replaced easily, "
        "whereas Francis suffers severe labyrinth seal wear requiring complex dismantling [Source: Silt_Handling_Manual.pdf]."
    )

    console.print("[bold yellow]--> RUNNING TEST CASE 1: FAITHFUL ANSWER (EXPECTED PASS)[/bold yellow]\n")

    result_1 = run_generation_guardrail(
        query=shared_query,
        final_response=faithful_answer,
        synthesized_context=shared_context,
    )

    t1 = Table(title="Test Case 1 Guardrail Result", show_header=True, header_style="bold magenta")
    t1.add_column("Field", style="cyan", width=24)
    t1.add_column("Value", style="white")
    t1.add_row("is_grounded", str(result_1.get("is_grounded")))
    t1.add_row("faithfulness_score", f"{result_1.get('generation_metrics', {}).get('faithfulness', 0.0):.2f}")
    t1.add_row("unfaithful_statements", str(result_1.get("unfaithful_statements")))
    t1.add_row("guardrail_critique", str(result_1.get("guardrail_critique")))
    console.print(t1)

    # -----------------------------------------------------------------------
    # Test Case 2: HALLUCINATED / UNGROUNDED ANSWER
    # -----------------------------------------------------------------------
    hallucinated_answer = (
        "## 1. Executive Summary\n"
        "At 120m net head and 2.0 m3/s discharge, the calculated power is 95.5 MW. "
        "Francis turbines achieve an unprecedented 99.8% peak efficiency using experimental diamond bearings. "
        "Furthermore, the runners are fabricated from solid 24-karat gold to completely prevent all corrosion. "
        "The estimated project capital cost is 45 billion dollars."
    )

    console.print("\n[bold yellow]--> RUNNING TEST CASE 2: HALLUCINATED ANSWER (EXPECTED FAIL)[/bold yellow]\n")

    result_2 = run_generation_guardrail(
        query=shared_query,
        final_response=hallucinated_answer,
        synthesized_context=shared_context,
    )

    t2 = Table(title="Test Case 2 Guardrail Result", show_header=True, header_style="bold magenta")
    t2.add_column("Field", style="cyan", width=24)
    t2.add_column("Value", style="white")
    t2.add_row("is_grounded", str(result_2.get("is_grounded")))
    t2.add_row("faithfulness_score", f"{result_2.get('generation_metrics', {}).get('faithfulness', 0.0):.2f}")
    t2.add_row("unfaithful_statements", str(result_2.get("unfaithful_statements")))
    t2.add_row("guardrail_critique", str(result_2.get("guardrail_critique")))
    console.print(t2)

    # -----------------------------------------------------------------------
    # Assertions
    # -----------------------------------------------------------------------
    assert result_1.get("is_grounded") is True, f"Expected Case 1 is_grounded=True, got {result_1.get('is_grounded')}"
    assert result_2.get("is_grounded") is False, f"Expected Case 2 is_grounded=False, got {result_2.get('is_grounded')}"
    assert len(result_2.get("unfaithful_statements", [])) > 0, "Expected unfaithful statements in Case 2"

    console.print(
        Panel(
            "[bold green][PASS] Post-Generation Guardrail verification tests PASSED successfully![/bold green]\n"
            f"* Faithful answer passed (Score: {result_1.get('generation_metrics', {}).get('faithfulness', 0.0):.2f})\n"
            f"* Hallucinated answer caught (Score: {result_2.get('generation_metrics', {}).get('faithfulness', 0.0):.2f}) "
            f"with {len(result_2.get('unfaithful_statements', []))} unverified statements flagged.",
            title="Verification Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()

