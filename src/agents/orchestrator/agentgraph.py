"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Orchestrator Agent Graph
=============================================================================
Module: src.agents.orchestrator.agentgraph
Purpose: Assembles and compiles the LangGraph StateGraph for the Orchestrator
         Agent using dynamic Map-Reduce (Send API) to spawn parallel worker
         sub-agents (narrative_worker, tabular_worker, mcp_worker).
=============================================================================
"""

from typing import Any

from langgraph.graph import END, StateGraph
from loguru import logger

from src.agents.orchestrator.agentnodes import (
    aggregate_results_node,
    dispatch_sub_agents,
    mcp_worker_node,
    narrative_worker_node,
    plan_orchestration_node,
    tabular_worker_node,
)
from src.states.orchestratoragent_state import OrchestratorAgentState


def build_orchestrator_graph():
    """
    Constructs and compiles the Orchestrator Agent LangGraph workflow using
    dynamic scatter-gather (Send API).

    Topology:
      START
        ──> [plan_orchestration]
        ──(Send)──> [narrative_worker]  ──> [aggregate_results] ──> END
        ──(Send)──> [tabular_worker]    ──> [aggregate_results] ──> END
        ──(Send)──> [mcp_worker]        ──> [aggregate_results] ──> END
    """
    workflow = StateGraph(OrchestratorAgentState)

    # 1. Register Orchestration Root Node
    workflow.add_node("plan_orchestration", plan_orchestration_node)

    # 2. Register Independent Worker Nodes
    workflow.add_node("narrative_worker", narrative_worker_node)
    workflow.add_node("tabular_worker", tabular_worker_node)
    workflow.add_node("mcp_worker", mcp_worker_node)

    # 3. Register Fan-In Aggregation Node
    workflow.add_node("aggregate_results", aggregate_results_node)

    # 4. Entry Point
    workflow.set_entry_point("plan_orchestration")

    # 5. Dynamic Fan-Out via Send() Router
    workflow.add_conditional_edges(
        "plan_orchestration",
        dispatch_sub_agents,
        ["narrative_worker", "tabular_worker", "mcp_worker", "aggregate_results"],
    )

    # 6. Fan-In Edges from Workers to Aggregator
    workflow.add_edge("narrative_worker", "aggregate_results")
    workflow.add_edge("tabular_worker", "aggregate_results")
    workflow.add_edge("mcp_worker", "aggregate_results")

    # 7. Final Edge
    workflow.add_edge("aggregate_results", END)

    compiled_graph = workflow.compile()
    logger.info("Compiled Orchestrator Agent StateGraph with Send() Map-Reduce.")
    return compiled_graph


def run_orchestrator(
    user_query: str,
    sub_queries: list[dict[str, Any]],
) -> OrchestratorAgentState:
    """
    Executes the Orchestrator Agent graph for a set of sub-queries.

    Args:
        user_query: Original user question.
        sub_queries: List of SubQuerySpec dictionaries from Planner Agent.

    Returns:
        Final state dictionary containing sub_agent_results and all_retrieved_chunks.
    """
    graph = build_orchestrator_graph()

    initial_state: OrchestratorAgentState = {
        "user_query": user_query,
        "sub_queries": sub_queries,
        "tasks": None,
        "coordination_strategy": None,
        "sub_agent_results": {},
        "all_retrieved_chunks": [],
        "total_retrieved": None,
        "error": None,
    }

    final_state = graph.invoke(initial_state)
    return final_state


# ---------------------------------------------------------------------------
# Standalone Verification CLI Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ORCHESTRATOR AGENT MAP-REDUCE (SEND API) VERIFICATION RUN")
    print("=" * 80)

    # Test Query: Multi-part query requiring Narrative + Tabular + MCP Calculation
    test_query = (
        "Compare Pelton and Francis turbines for 150m head, provide Khari Khola monthly discharge, "
        "and calculate power output for Q=4.5 m3/s at 150m head."
    )
    test_sub_queries = [
        {
            "sub_query_id": "sq_1",
            "query_text": "Pelton and Francis turbine head range application efficiency for 150m head",
            "target_filter": {"is_table": False},
            "rationale": "Retrieve turbine application manual text.",
        },
        {
            "sub_query_id": "sq_2",
            "query_text": "Khari Khola monthly discharge flow data and design discharge table",
            "target_filter": {"is_table": True},
            "rationale": "Retrieve structured discharge tables.",
        },
        {
            "sub_query_id": "sq_3",
            "query_text": "Calculate power output for discharge 4.5 m3/s and net head 150m",
            "target_filter": {"discharge_m3s": 4.5, "net_head_m": 150.0},
            "rationale": "Exact formula execution via calculate_hydro_power MCP tool.",
        },
    ]

    result = run_orchestrator(test_query, test_sub_queries)

    print(f"\nUser Query: {result.get('user_query')}")
    print(f"Strategy  : {result.get('coordination_strategy')}")
    print(f"\nPlanned Tasks ({len(result.get('tasks', []))}):")
    for t in result.get("tasks", []):
        print(f"  • [{t['sub_query_id']}] -> Worker: {t['worker_type']} (filter: {t['target_filter']}, top_k={t['top_k']})")

    sub_results = result.get("sub_agent_results", {})
    print(f"\nSub-Query Results Groups (Workers Fanned-In): {len(sub_results)}")
    for sq_id, chunks in sub_results.items():
        print(f"\n  [{sq_id}] Returned {len(chunks)} items:")
        for i, c in enumerate(chunks[:2], 1):
            badge = "🧮 [MCP TOOL]" if c.get("file_type") == "TOOL_OUTPUT" else ("📊 [TABLE]" if c.get("is_table") else "📄 [TEXT]")
            print(f"    {i}. {badge} (Score: {c.get('score'):.3f}) Source: {c.get('source_file')}")
            print(f"       Breadcrumb: {c.get('breadcrumb')}")
            snippet = c.get("content", "")[:130].replace("\n", " ")
            print(f"       Preview   : {snippet}...")

    all_chunks = result.get("all_retrieved_chunks", [])
    print(f"\nTotal Unique Chunks Aggregated: {result.get('total_retrieved')} (raw: {len(all_chunks)})")
    print("=" * 80 + "\n")
