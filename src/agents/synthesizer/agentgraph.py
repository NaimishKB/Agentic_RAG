"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Synthesizer Agent Graph
=============================================================================
Module: src.agents.synthesizer.agentgraph
Purpose: Assembles and compiles the LangGraph StateGraph for the Synthesizer Agent.
         Provides standalone execution capability for context fusion,
         deduplication, citation generation, and verification.
=============================================================================
"""

from typing import Any

from langgraph.graph import END, StateGraph
from loguru import logger

from src.agents.synthesizer.agentnodes import synthesizer_node
from src.states.synthesizeragent_state import SynthesizerAgentState


def build_synthesizer_graph():
    """
    Constructs and compiles the Synthesizer Agent LangGraph workflow.

    Topology:
      START ──> [synthesizer] ──> END
    """
    workflow = StateGraph(SynthesizerAgentState)

    # Add synthesis node
    workflow.add_node("synthesizer", synthesizer_node)

    # Set flow
    workflow.set_entry_point("synthesizer")
    workflow.add_edge("synthesizer", END)

    compiled_graph = workflow.compile()
    logger.info("Compiled Synthesizer Agent LangGraph StateGraph successfully.")
    return compiled_graph


def run_synthesizer(
    query: str,
    sub_queries: list[dict[str, Any]] | None = None,
    sub_agent_results: dict[str, list[dict[str, Any]]] | None = None,
    all_retrieved_chunks: list[dict[str, Any]] | None = None,
) -> SynthesizerAgentState:
    """
    Executes the Synthesizer Agent graph in standalone mode.

    Args:
        query: User input query.
        sub_queries: SubQuery specifications from Planner.
        sub_agent_results: Chunks grouped by sub_query_id.
        all_retrieved_chunks: Flat list of retrieved chunks.

    Returns:
        Final state dictionary with synthesized_context, documents, citations, etc.
    """
    graph = build_synthesizer_graph()

    initial_state: SynthesizerAgentState = {
        "query": query,
        "user_query": query,
        "sub_queries": sub_queries or [],
        "sub_agent_results": sub_agent_results or {},
        "all_retrieved_chunks": all_retrieved_chunks or [],
    }

    result = graph.invoke(initial_state)
    return result


def main():
    """CLI test runner verifying Synthesizer Agent Graph execution with realistic SHP data."""
    print("\n" + "=" * 80)
    print("SYNTHESIZER AGENT LANGGRAPH VERIFICATION RUN")
    print("=" * 80)

    test_query = (
        "Evaluate Kaplan vs Francis turbine for 22m head with silt, and calculate power output for 15 m3/s discharge."
    )

    test_sub_queries = [
        {
            "sub_query_id": "sq_1",
            "query_text": "Kaplan vs Francis turbine selection head 22m silt erosion wear resistance",
            "rationale": "Evaluate turbine suitability under low head and silt abrasive wear",
        },
        {
            "sub_query_id": "sq_2",
            "query_text": "Hydro power calculation head 22m discharge 15 m3/s",
            "rationale": "Calculate expected power generation",
        },
    ]

    # Mock realistic chunks from narrative worker, tabular worker, and MCP tool
    mock_chunks_sq1 = [
        {
            "chunk_id": "chk_kaplan_01",
            "content": (
                "For low-head regimes between 10m and 35m, Kaplan turbines offer the highest part-load efficiency "
                "(up to 92%) through simultaneous adjustment of runner blades and wicket gates. "
                "However, high tip clearance velocities make Kaplan blades vulnerable to cavitation and silt pitting."
            ),
            "score": 0.88,
            "source_file": "Turbine_Selection_Manual_CEA.pdf",
            "file_type": "PDF",
            "breadcrumb": "Chapter 4 > Low Head Hydro Turbines > Kaplan Selection",
            "is_table": False,
        },
        {
            "chunk_id": "chk_francis_02",
            "content": (
                "Francis turbines can operate down to 20m head but exhibit steep efficiency drop-offs below 50% load. "
                "Where quartz sediment exceeds 2000 ppm, Francis runners with tungsten carbide coatings provide superior "
                "abrasion resistance compared to Kaplan runner blade seals."
            ),
            "score": 0.84,
            "source_file": "Hydro_Mechanical_Equipment_Guidelines.docx",
            "file_type": "DOCX",
            "breadcrumb": "Section 3 > Sediment Erosion & Runner Longevity",
            "is_table": False,
        },
    ]

    mock_mcp_output = [
        {
            "chunk_id": "mcp_power_calc_01",
            "content": (
                "Inputs: discharge_m3s=15.0, net_head_m=22.0, efficiency=0.88, gravity=9.81 m/s²\n"
                "Results: Electrical Power P = 2.849 MW (2848.86 kW), Annual Energy Potential = 24.96 GWh (at 100% plant factor)"
            ),
            "score": 1.0,
            "source_file": "calculate_hydro_power",
            "file_type": "TOOL_OUTPUT",
            "breadcrumb": "MCP Tool: calculate_hydro_power",
            "is_table": False,
        }
    ]

    sub_agent_results = {
        "sq_1": mock_chunks_sq1,
        "sq_2": mock_mcp_output,
    }

    all_chunks = mock_chunks_sq1 + mock_mcp_output

    print(f"\nUser Query: {test_query}")
    print(f"Sub-Queries: {len(test_sub_queries)}")
    print(f"Total Chunks Provided: {len(all_chunks)}")
    print("\nExecuting Synthesizer Graph...\n")

    result = run_synthesizer(
        query=test_query,
        sub_queries=test_sub_queries,
        sub_agent_results=sub_agent_results,
        all_retrieved_chunks=all_chunks,
    )

    print("=" * 80)
    print("SYNTHESIZER RESULTS:")
    print("=" * 80)
    print(f"Executive Summary:\n{result.get('executive_summary')}\n")
    print(f"Coverage Score: {result.get('coverage_score')}")
    print(f"Citations Used: {result.get('citations')}")
    print(f"Conflicts Detected: {result.get('conflicts_detected')}\n")
    print("-" * 80)
    print("SYNTHESIZED CONTEXT DOSSIER:")
    print("-" * 80)
    print(result.get("synthesized_context"))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
