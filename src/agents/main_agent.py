"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - End-to-End Main Agent Workflow
=============================================================================
Module: src.agents.main_agent
Purpose: Assembles, compiles, and executes the complete End-to-End LangGraph
         StateGraph for the Small Hydropower Plant Agentic RAG system.
         Features dynamic worker map-reduce (Send API), dual quality gates,
         self-healing feedback reflection loop, circuit breaker fallbacks,
         and In-Memory RAM Checkpointing (MemorySaver) for single-session runs.
=============================================================================
"""

import sys
from typing import Any

# Ensure standard UTF-8 stream handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import src.utils.env_setup  # Load .env and configure LangSmith / LangChain tracing

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agents.fallback.agentnodes import fallback_node
from src.agents.feedback.agentnodes import feedback_node
from src.agents.generation_guardrail.agentnodes import generation_guardrail_node
from src.agents.generator.agentnodes import generator_node
from src.agents.orchestrator.agentnodes import (
    aggregate_results_node,
    dispatch_sub_agents,
    mcp_worker_node,
    narrative_worker_node,
    plan_orchestration_node,
    tabular_worker_node,
)
from src.agents.planner.agentnodes import plan_query_node
from src.agents.retrieval_grader.agentnodes import grade_retrieval_node
from src.agents.synthesizer.agentnodes import synthesizer_node
from src.states.agent_state import AgentState

console = Console(highlight=False)

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


# ---------------------------------------------------------------------------
# Routing & Conditional Edge Functions
# ---------------------------------------------------------------------------
def route_after_planner(state: AgentState) -> str:
    """
    Evaluates Planner Agent classification and routes accordingly.
    - direct_answer -> END (delivers conversational reply immediately)
    - out_of_scope -> fallback (domain boundary refusal)
    - rag_retrieval -> plan_orchestration (begins multi-agent retrieval)
    """
    decision = state.get("routing_decision")
    logger.info(f"[ROUTER: planner] Decision={decision}")

    if decision == "direct_answer":
        return END
    elif decision == "out_of_scope":
        return "fallback"
    return "plan_orchestration"


def route_after_retrieval_grader(state: AgentState) -> str:
    """
    Evaluates Pre-Generation Retrieval Sufficiency Grader (Production CRAG).
    - is_sufficient is True -> generator
    - is_sufficient is False AND cycle_count < 3 -> feedback
    - is_sufficient is False AND cycle_count >= 3 -> fallback (circuit breaker)
    """
    is_sufficient = state.get("is_retrieval_sufficient", False)
    cycle = state.get("cycle_count", 0)
    logger.info(f"[ROUTER: retrieval_grader] is_sufficient={is_sufficient} | cycle_count={cycle}")

    if is_sufficient:
        return "generator"
    elif cycle < 3:
        return "feedback"
    return "fallback"


def route_after_generation_guardrail(state: AgentState) -> str:
    """
    Evaluates Post-Generation DeepEval Faithfulness Guardrail.
    - is_grounded is True -> END (deliver verified answer to user)
    - is_grounded is False AND cycle_count < 3 -> feedback (loop back for remediation)
    - is_grounded is False AND cycle_count >= 3 -> fallback (circuit breaker)
    """
    is_grounded = state.get("is_grounded", False)
    cycle = state.get("cycle_count", 0)
    logger.info(f"[ROUTER: generation_guardrail] is_grounded={is_grounded} | cycle_count={cycle}")

    if is_grounded:
        return END
    elif cycle < 3:
        return "feedback"
    return "fallback"


# ---------------------------------------------------------------------------
# End-to-End StateGraph Builder
# ---------------------------------------------------------------------------
def build_main_agent_graph(checkpointer: Any | None = None):
    """
    Assembles and compiles the complete End-to-End SHP Agentic RAG StateGraph.

    Args:
        checkpointer: Optional checkpointer instance. Defaults to in-RAM MemorySaver().

    Returns:
        Compiled LangGraph StateGraph instance.
    """
    workflow = StateGraph(AgentState)

    # 1. Register All Agent & Worker Nodes
    workflow.add_node("planner", plan_query_node)
    workflow.add_node("plan_orchestration", plan_orchestration_node)
    workflow.add_node("narrative_worker", narrative_worker_node)
    workflow.add_node("tabular_worker", tabular_worker_node)
    workflow.add_node("mcp_worker", mcp_worker_node)
    workflow.add_node("aggregate_results", aggregate_results_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("retrieval_grader", grade_retrieval_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("generation_guardrail", generation_guardrail_node)
    workflow.add_node("feedback", feedback_node)
    workflow.add_node("fallback", fallback_node)

    # 2. Workflow Entry Point
    workflow.set_entry_point("planner")

    # 3. Routing from Planner
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        ["plan_orchestration", "fallback", END],
    )

    # 4. Dynamic Scatter-Gather from Orchestration Layer (Send API)
    workflow.add_conditional_edges(
        "plan_orchestration",
        dispatch_sub_agents,
        ["narrative_worker", "tabular_worker", "mcp_worker", "aggregate_results"],
    )

    # 5. Fan-In from Workers to Aggregator
    workflow.add_edge("narrative_worker", "aggregate_results")
    workflow.add_edge("tabular_worker", "aggregate_results")
    workflow.add_edge("mcp_worker", "aggregate_results")

    # 6. Aggregator to Synthesizer
    workflow.add_edge("aggregate_results", "synthesizer")

    # 7. Synthesizer to Pre-Gen Retrieval Grader
    workflow.add_edge("synthesizer", "retrieval_grader")

    # 8. Conditional Routing from Retrieval Grader
    workflow.add_conditional_edges(
        "retrieval_grader",
        route_after_retrieval_grader,
        ["generator", "feedback", "fallback"],
    )

    # 9. Generator to Post-Gen Guardrail
    workflow.add_edge("generator", "generation_guardrail")

    # 10. Conditional Routing from Generation Guardrail
    workflow.add_conditional_edges(
        "generation_guardrail",
        route_after_generation_guardrail,
        [END, "feedback", "fallback"],
    )

    # 11. Self-Correction Loop: Feedback back to Planner
    workflow.add_edge("feedback", "planner")

    # 12. Fallback to END
    workflow.add_edge("fallback", END)

    # 13. Compile with In-RAM Checkpointer
    active_checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    compiled_app = workflow.compile(checkpointer=active_checkpointer)
    logger.info("Compiled End-to-End SHP Agentic RAG StateGraph with In-RAM MemorySaver.")
    return compiled_app


# ---------------------------------------------------------------------------
# High-Level Execution Helper
# ---------------------------------------------------------------------------
def run_shp_agent(
    query: str,
    thread_id: str = "default_session",
    checkpointer: Any | None = None,
) -> AgentState:
    """
    Executes the End-to-End SHP Agentic RAG workflow for a given query in a session thread.

    Args:
        query: User input query.
        thread_id: Session identifier for In-Memory checkpoint tracing.
        checkpointer: Optional custom checkpointer.

    Returns:
        Final AgentState dictionary containing response, citations, metrics, and plan.
    """
    app = build_main_agent_graph(checkpointer=checkpointer)

    initial_state: AgentState = {
        "query": query,
        "user_query": query,
        "original_query": query,
        "messages": [],
        "cycle_count": 0,
        "sub_agent_results": {},
        "all_retrieved_chunks": [],
        "documents": [],
    }

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = app.invoke(initial_state, config=config)
    return result


# ---------------------------------------------------------------------------
# Verification CLI Runner
# ---------------------------------------------------------------------------
def main():
    """CLI test runner verifying End-to-End graph on all 3 routing paths."""
    console.print("\n" + "=" * 80, style="bold cyan")
    console.print("[*] END-TO-END SHP AGENTIC RAG SYSTEM VERIFICATION (PHASE 6)", style="bold green")
    console.print("=" * 80 + "\n", style="bold cyan")

    # Shared In-Memory Checkpointer across test runs
    session_memory = MemorySaver()

    # -----------------------------------------------------------------------
    # TEST RUN 1: DIRECT CONVERSATIONAL GREETING (Fast Path)
    # -----------------------------------------------------------------------
    console.print("[bold yellow]► TEST 1: DIRECT CONVERSATIONAL GREETING[/bold yellow]")
    q1 = "Hello! What engineering topics and capabilities can you assist with?"
    console.print(f"Query: [cyan]\"{q1}\"[/cyan]\n")

    res1 = run_shp_agent(query=q1, thread_id="session_test_1", checkpointer=session_memory)

    t1 = Table(title="Test 1 Result (Direct Answer)", show_header=True, header_style="bold magenta")
    t1.add_column("Field", style="cyan", width=22)
    t1.add_column("Value", style="white")
    t1.add_row("Routing Decision", str(res1.get("routing_decision")))
    t1.add_row("Response Length", f"{len(res1.get('final_response', ''))} chars")
    console.print(t1)
    console.print(Panel(res1.get("final_response", ""), title="Direct Answer", border_style="cyan"))

    assert res1.get("routing_decision") == "direct_answer", f"Expected direct_answer, got {res1.get('routing_decision')}"
    assert len(res1.get("final_response", "")) > 20, "Expected non-empty direct answer"

    # -----------------------------------------------------------------------
    # TEST RUN 2: OUT OF SCOPE QUERY (Domain Refusal Fallback)
    # -----------------------------------------------------------------------
    console.print("\n[bold yellow]► TEST 2: OUT-OF-SCOPE DOMAIN REFUSAL[/bold yellow]")
    q2 = "Can you write me a recipe for delicious chocolate cake with cream?"
    console.print(f"Query: [cyan]\"{q2}\"[/cyan]\n")

    res2 = run_shp_agent(query=q2, thread_id="session_test_2", checkpointer=session_memory)

    t2 = Table(title="Test 2 Result (Out of Scope)", show_header=True, header_style="bold magenta")
    t2.add_column("Field", style="cyan", width=22)
    t2.add_column("Value", style="white")
    t2.add_row("Routing Decision", str(res2.get("routing_decision")))
    t2.add_row("Response Length", f"{len(res2.get('final_response', ''))} chars")
    console.print(t2)
    console.print(Panel(res2.get("final_response", ""), title="Domain Refusal Fallback", border_style="yellow"))

    assert res2.get("routing_decision") == "out_of_scope", f"Expected out_of_scope, got {res2.get('routing_decision')}"
    assert "Out of Scope" in res2.get("final_response", ""), "Expected domain refusal notice"

    # -----------------------------------------------------------------------
    # TEST RUN 3: FULL RAG PIPELINE (Pelton vs Francis + MCP Power Calc)
    # -----------------------------------------------------------------------
    console.print("\n[bold yellow]► TEST 3: FULL RAG ENGINEERING LIFECYCLE[/bold yellow]")
    q3 = (
        "Compare Francis vs Pelton turbines for 120m net head and 2.0 m3/s discharge "
        "operating in high quartz silt conditions, and provide power potential."
    )
    console.print(f"Query: [cyan]\"{q3}\"[/cyan]\n")

    res3 = run_shp_agent(query=q3, thread_id="session_test_3", checkpointer=session_memory)

    t3 = Table(title="Test 3 Result (Full RAG Pipeline)", show_header=True, header_style="bold magenta")
    t3.add_column("Field", style="cyan", width=24)
    t3.add_column("Value", style="white")
    t3.add_row("Routing Decision", str(res3.get("routing_decision")))
    t3.add_row("Sub-Queries Planned", str(len(res3.get("sub_queries", []))))
    t3.add_row("Total Retrieved Chunks", str(res3.get("total_retrieved", 0)))
    t3.add_row("Synthesized Coverage", f"{res3.get('coverage_score', 0.0):.2f}")
    t3.add_row("Retrieval Sufficient", str(res3.get("is_retrieval_sufficient")))
    t3.add_row("Is Grounded", str(res3.get("is_grounded")))
    t3.add_row("Citations Used", str(len(res3.get("citations", []))))
    t3.add_row("Final Response Length", f"{len(res3.get('final_response', ''))} chars")
    console.print(t3)

    console.print("\n[bold yellow]-- VERIFIED TECHNICAL ANSWER --[/bold yellow]")
    console.print(Panel(res3.get("final_response", ""), border_style="green", title="Final Answer"))

    assert res3.get("routing_decision") == "rag_retrieval", "Expected rag_retrieval"
    assert len(res3.get("final_response", "")) > 100, "Expected full engineering answer"

    console.print(
        Panel(
            "[bold green][PASS] End-to-End SHP Agentic RAG System verification tests PASSED successfully![/bold green]\n"
            "* Test 1: Direct Answer path executed seamlessly\n"
            "* Test 2: Out of Scope domain refusal correctly intercepted\n"
            "* Test 3: Full multi-agent RAG workflow executed end-to-end with in-memory checkpointing.",
            title="Phase 6 Verification Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
