"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Generator Agent Graph
=============================================================================
Module: src.agents.generator.agentgraph
Purpose: Assembles and compiles the LangGraph StateGraph for the Answer Generator
         Agent. Provides standalone execution capability for generating verified,
         grounded technical responses.
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

from src.agents.generator.agentnodes import generator_node
from src.states.generatoragent_state import GeneratorAgentState

console = Console(highlight=False)


def build_generator_graph():
    """
    Constructs and compiles the Generator Agent LangGraph workflow.

    Workflow:
      START ──> [generator] ──> END
    """
    workflow = StateGraph(GeneratorAgentState)

    workflow.add_node("generator", generator_node)
    workflow.set_entry_point("generator")
    workflow.add_edge("generator", END)

    compiled_graph = workflow.compile()
    logger.info("Compiled Generator Agent LangGraph StateGraph successfully.")
    return compiled_graph


def run_generator(
    query: str,
    synthesized_context: str,
    executive_summary: str | None = None,
    sub_queries: list[dict[str, Any]] | None = None,
    cycle_count: int = 0,
) -> GeneratorAgentState:
    """
    Executes the Generator Agent graph to synthesize a verified technical answer.

    Args:
        query: Primary user engineering query.
        synthesized_context: Structured context dossier from Synthesizer.
        executive_summary: Optional technical summary.
        sub_queries: Optional planned sub-queries.
        cycle_count: Current loop iteration.

    Returns:
        GeneratorAgentState containing final_response, citations, generation_confidence.
    """
    graph = build_generator_graph()

    initial_state: GeneratorAgentState = {
        "query": query,
        "user_query": query,
        "synthesized_context": synthesized_context,
        "executive_summary": executive_summary,
        "sub_queries": sub_queries or [],
        "cycle_count": cycle_count,
    }

    result = graph.invoke(initial_state)
    return result


def main():
    """CLI test runner verifying Answer Generator execution on realistic SHP engineering evidence."""
    console.print("\n" + "=" * 80, style="bold cyan")
    console.print("[*] ANSWER GENERATOR AGENT (QWEN3.5:9B) VERIFICATION RUN", style="bold green")
    console.print("=" * 80 + "\n", style="bold cyan")

    # Sample realistic SHP query and synthesized evidence
    test_query = (
        "Compare Francis vs Pelton turbines for 120m net head and 2.0 m3/s discharge "
        "operating in high quartz silt conditions, and provide power potential."
    )
    test_sub_queries = [
        {
            "sub_query_id": "sq_1",
            "query_text": "Turbine selection for 120m head Francis vs Pelton",
        },
        {
            "sub_query_id": "sq_2",
            "query_text": "Silt erosion countermeasures Francis vs Pelton quartz silt",
        },
        {
            "sub_query_id": "sq_3",
            "query_text": "Hydro power potential calculation for Q=2.0 m3/s and H=120m",
        },
    ]
    test_summary = (
        "At 120m head and 2.0 m3/s discharge, calculated power potential is 2.07 MW. "
        "Both Francis and Pelton turbines operate in this head range, but Pelton offers significant "
        "longevity and maintenance cost advantages under high quartz silt conditions."
    )
    test_context = (
        "### 1. Programmatic Power Calculation\n"
        "- **Calculated Power**: 2,072.76 kW (2.07 MW) [Source: MCP Tool | Tool: calculate_hydro_power]\n"
        "- **Input Parameters**: Net Head H = 120m, Design Discharge Q = 2.0 m3/s, Overall Efficiency = 0.88 "
        "[Source: MCP Tool | Tool: calculate_hydro_power]\n\n"
        "### 2. Turbine Operating Head Ranges & Efficiency\n"
        "- **Francis Turbines**: Applicable for net heads from 40m to 350m. High peak efficiency (up to 94%), "
        "but efficiency drops significantly below 50% rated discharge [Source: SHP_Guidelines_Part2.pdf | Section: Turbines].\n"
        "- **Pelton Turbines**: Applicable for net heads from 100m to 1000m+. Multi-jet configurations maintain high "
        "efficiency (>88%) across 20% to 110% rated flow [Source: CEA_Hydro_Standards.pdf | Section: Electro-Mechanical].\n\n"
        "### 3. Quartz Silt Wear & Maintenance Longevity\n"
        "- In rivers with high sediment and quartz content (>0.2mm), Pelton turbines are strongly recommended "
        "because runner buckets and spear valves can be coated with tungsten carbide (WC-Co-Cr) via HVOF and replaced easily. "
        "Conversely, Francis runners and guide vanes suffer severe labyrinth seal clearance degradation, leading to rapid "
        "efficiency loss and complex plant dismantling [Source: Silt_Handling_Manual.pdf | Section: Hydro-Abrasive Wear]."
    )

    console.print(f"Test Query: [cyan]{test_query}[/cyan]\n")

    result = run_generator(
        query=test_query,
        synthesized_context=test_context,
        executive_summary=test_summary,
        sub_queries=test_sub_queries,
    )

    t = Table(title="Generator Verification Result", show_header=True, header_style="bold magenta")
    t.add_column("Field", style="cyan", width=24)
    t.add_column("Value", style="white")
    t.add_row("generation_confidence", f"{result.get('generation_confidence', 0.0):.2f}")
    t.add_row("citations", str(result.get("citations")))
    t.add_row("response_length", f"{len(result.get('final_response', ''))} characters")
    console.print(t)

    console.print("\n[bold yellow]-- GENERATED TECHNICAL ANSWER --[/bold yellow]")
    console.print(Panel(result.get("final_response", ""), border_style="cyan"))

    # Assertions
    assert len(result.get("final_response", "")) > 100, "Expected non-trivial answer length"
    assert len(result.get("citations", [])) > 0, "Expected at least one extracted citation"

    console.print(
        Panel(
            "[bold green][PASS] Answer Generator Agent verification tests PASSED successfully![/bold green]\n"
            f"* Generated comprehensive response ({len(result.get('final_response', ''))} chars)\n"
            f"* Harvested {len(result.get('citations', []))} distinct citations.",
            title="Verification Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()

