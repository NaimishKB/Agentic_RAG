"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Planner Agent Graph
=============================================================================
Module: src.agents.planner.agentgraph
Purpose: Assembles and compiles the LangGraph StateGraph for the Planner Agent.
         Provides standalone execution capability for the planning lifecycle.
=============================================================================
"""

from typing import Any

from langgraph.graph import END, StateGraph
from loguru import logger

from src.agents.planner.agentnodes import plan_query_node
from src.states.planneragent_state import PlannerAgentState


def build_planner_graph():
    """
    Constructs and compiles the Planner Agent LangGraph workflow.

    Workflow:
      START ──> [plan_query] ──> END
    """
    workflow = StateGraph(PlannerAgentState)

    # Add planning node
    workflow.add_node("plan_query", plan_query_node)

    # Set flow
    workflow.set_entry_point("plan_query")
    workflow.add_edge("plan_query", END)

    compiled_graph = workflow.compile()
    logger.info("Compiled Planner Agent LangGraph StateGraph successfully.")
    return compiled_graph


def run_planner(
    query: str,
    feedback_diagnosis: dict[str, Any] | None = None,
    cycle_count: int = 0,
) -> PlannerAgentState:
    """
    Executes the Planner Agent graph for a user query.

    Args:
        query: User input query.
        feedback_diagnosis: Optional critique from Feedback Agent.
        cycle_count: Current loop iteration.

    Returns:
        Final state dictionary containing routing_decision, plan, sub_queries, etc.
    """
    graph = build_planner_graph()

    initial_state: PlannerAgentState = {
        "query": query,
        "original_query": query,
        "feedback_diagnosis": feedback_diagnosis,
        "cycle_count": cycle_count,
    }

    result = graph.invoke(initial_state)
    return result


def main():
    """CLI test runner verifying Planner Agent Graph execution."""
    print("\n" + "=" * 80)
    print("PLANNER AGENT LANGGRAPH VERIFICATION RUN")
    print("=" * 80)

    test_queries = [
        "Can you write me a recipe for chocolate cake?",
        "Hi! What topics can you assist me with regarding our hydro plant?",
        "Compare Pelton and Francis turbines for 150m head, and check CEA tariff regulations.",
    ]

    for i, q in enumerate(test_queries, start=1):
        print(f"\n--- [Graph Run {i}] ---")
        print(f'Input Query: "{q}"')
        output = run_planner(query=q)

        print(f"Routing Decision: {output.get('routing_decision')}")
        if output.get("final_response"):
            print(f"Direct Response : {output.get('final_response')}")
        if output.get("sub_queries"):
            print(f"Sub-Queries ({len(output.get('sub_queries'))}):")
            for sq in output.get("sub_queries"):
                print(
                    f"  • [{sq.get('sub_query_id')}] Filter: {sq.get('target_filter')}"
                )
                print(f"    Text: {sq.get('query_text')}")
                print(f"    Why : {sq.get('rationale')}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
