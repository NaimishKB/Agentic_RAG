"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Interactive CLI Entry Point
=============================================================================
Module: src.main
Purpose: Provides an interactive command-line interface (CLI) for engineers
         to query the Small Hydropower Plant Agentic RAG system across turns.
         Features rich console styling, status spinners, telemetry tables,
         citation references, session resets, and prerequisite checks.
=============================================================================
"""

import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path whether executed as `python src/main.py` or `python -m src.main`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reconfigure Windows stdout/stderr to handle UTF-8 symbols gracefully
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table

from src.agents.main_agent import build_main_agent_graph
from src.mcp_server.client import FastAPIMCPClient
from src.states.agent_state import AgentState
import src.utils.env_setup  # Load .env and configure LangSmith / LangChain tracing

console = Console(highlight=False)

# Silence low-level node logs on the console during interactive chat; log to file
logger.remove()
logger.add(
    "logs/shp_cli.log",
    rotation="20 MB",
    retention="14 days",
    level="DEBUG",
    encoding="utf-8",
)


# ---------------------------------------------------------------------------
# System Health & Prerequisite Verification
# ---------------------------------------------------------------------------
def check_system_health(
    ollama_url: str = "http://127.0.0.1:11434",
    qdrant_url: str = "http://127.0.0.1:6333",
    mcp_url: str = "http://127.0.0.1:8000",
) -> bool:
    """
    Validates connectivity to Ollama LLM, Qdrant Vector Store, and MCP Server.
    Displays an informative status table.
    """
    table = Table(title="System Service Diagnostics", header_style="bold cyan", show_header=True)
    table.add_column("Service", style="bold white", width=22)
    table.add_column("Endpoint", style="cyan", width=28)
    table.add_column("Status", width=16)
    table.add_column("Details", style="yellow")

    all_ok = True

    # 1. Ollama Check
    try:
        t0 = time.perf_counter()
        r = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=3)
        dt = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            models = [m.get("name") for m in r.json().get("models", [])]
            has_llm = any("qwen3.5:9b" in m or "qwen3.5" in m for m in models)
            has_embed = any("qwen3-embedding" in m for m in models)
            status_text = "[bold green]ONLINE[/bold green]"
            detail = f"{len(models)} models ({dt:.1f}ms) | LLM: {'OK' if has_llm else 'MISSING'} | Embed: {'OK' if has_embed else 'MISSING'}"
            if not has_llm or not has_embed:
                all_ok = False
        else:
            status_text = "[bold red]ERROR[/bold red]"
            detail = f"HTTP {r.status_code}"
            all_ok = False
    except requests.RequestException as exc:
        status_text = "[bold red]OFFLINE[/bold red]"
        detail = str(exc)[:40]
        all_ok = False
    table.add_row("Ollama LLM Daemon", ollama_url, status_text, detail)

    # 2. Qdrant Check
    try:
        t0 = time.perf_counter()
        r = requests.get(f"{qdrant_url.rstrip('/')}/collections", timeout=3)
        dt = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            cols = [c.get("name") for c in r.json().get("result", {}).get("collections", [])]
            has_shp = "shp_knowledge" in cols
            status_text = "[bold green]ONLINE[/bold green]"
            detail = f"Collections: {cols} ({dt:.1f}ms)"
            if not has_shp:
                detail += " | WARNING: 'shp_knowledge' collection missing!"
                all_ok = False
        else:
            status_text = "[bold red]ERROR[/bold red]"
            detail = f"HTTP {r.status_code}"
            all_ok = False
    except requests.RequestException as exc:
        status_text = "[bold red]OFFLINE[/bold red]"
        detail = str(exc)[:40]
        all_ok = False
    table.add_row("Qdrant Vector Database", qdrant_url, status_text, detail)

    # 3. FastAPI MCP Tool Server Check
    mcp_client = FastAPIMCPClient(base_url=mcp_url)
    if mcp_client.is_server_online():
        status_text = "[bold green]ONLINE (HTTP)[/bold green]"
        detail = f"Connected to {mcp_url}"
    else:
        status_text = "[bold yellow]IN-PROCESS[/bold yellow]"
        detail = "Server offline; using fast in-process JSON-RPC dispatch"
    table.add_row("FastAPI MCP Tool Server", mcp_url, status_text, detail)

    # 4. LangSmith Tracing Service Check
    import os
    tracing_on = os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1")
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    project = os.getenv("LANGSMITH_PROJECT", "agentic-rag-shp")
    if tracing_on:
        try:
            t0 = time.perf_counter()
            from langsmith import Client
            ls = Client()
            dt = (time.perf_counter() - t0) * 1000
            status_text = "[bold green]CONNECTED[/bold green]"
            detail = f"Project: '{project}' | Write/Tracing Token Active"
        except Exception as exc:
            status_text = "[bold red]AUTH/CONN ERROR[/bold red]"
            detail = str(exc)[:40]
            all_ok = False
    else:
        status_text = "[dim]DISABLED[/dim]"
        detail = "LANGSMITH_TRACING is false"
    table.add_row("LangSmith Observability", endpoint, status_text, detail)

    console.print(table)
    return all_ok


# ---------------------------------------------------------------------------
# Output Formatting Helpers
# ---------------------------------------------------------------------------
def print_banner():
    """Prints the branded SHP Agentic RAG terminal header."""
    banner = (
        "[bold cyan]==============================================================================[/bold cyan]\n"
        "[bold green]   ⚡ SMALL HYDROPOWER PLANT (SHP) AGENTIC RAG SYSTEM[/bold green]\n"
        "[bold white]   Multi-Agent Engineering Assistant with Self-Healing Quality Feedback[/bold white]\n"
        "[bold cyan]==============================================================================[/bold cyan]\n"
        "[dim]Powered by LangGraph, Local Ollama (qwen3.5:9b), Qdrant Vector DB, & FastAPI MCP[/dim]\n"
        "[dim]Commands: [yellow]/reset[/yellow] (new session) | [yellow]/status[/yellow] (health check) | [yellow]/help[/yellow] | [yellow]/exit[/yellow][/dim]"
    )
    console.print(Panel(banner, border_style="cyan"))


def display_citations_table(citations: list[dict[str, Any]]):
    """Renders a clean citation provenance table for verified document references."""
    if not citations:
        return

    table = Table(
        title="Engineering Citations & Knowledge Provenance",
        header_style="bold magenta",
        show_header=True,
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Source Document", style="cyan", width=34)
    table.add_column("Type", style="yellow", width=8)
    table.add_column("Location / Section", style="white", width=28)
    table.add_column("Score", style="green", width=8)

    for i, c in enumerate(citations, 1):
        if isinstance(c, dict):
            src = c.get("source_file", "Unknown")
            ftype = c.get("file_type", "DOC")
            loc = c.get("breadcrumb") or (f"Page {c.get('page_or_slide')}" if c.get("page_or_slide") else "N/A")
            score = f"{float(c.get('score', 0.0)):.2f}"
        else:
            src = str(c)
            ftype = "DOC"
            loc = "In-Text Citation"
            score = "1.00"
        table.add_row(str(i), src[:34], ftype, loc[:28], score)

    console.print(table)


def display_telemetry_table(state: AgentState, duration_s: float):
    """Displays execution telemetry and quality gate metrics."""
    table = Table(
        title="Lifecycle Execution Telemetry",
        header_style="bold blue",
        show_header=True,
    )
    table.add_column("Metric / Stage", style="cyan", width=26)
    table.add_column("Value / Status", style="white")

    decision = state.get("routing_decision", "unknown")
    table.add_row("Routing Decision", f"[bold]{decision}[/bold]")
    table.add_row("Execution Latency", f"{duration_s:.2f} seconds")

    if decision == "rag_retrieval":
        sub_queries = state.get("sub_queries", [])
        table.add_row("Sub-Queries Planned", str(len(sub_queries)))
        table.add_row("Total Chunks Retrieved", str(state.get("total_retrieved", 0)))
        cov = state.get("coverage_score")
        if cov is not None:
            table.add_row("Synthesizer Coverage", f"{float(cov):.2f}")
        is_suff = state.get("is_retrieval_sufficient")
        suff_str = "[green]PASS[/green]" if is_suff else "[red]INSUFFICIENT[/red]"
        table.add_row("Retrieval Sufficiency (CRAG)", suff_str)
        is_grounded = state.get("is_grounded")
        grounded_str = "[green]FAITHFUL (1.00)[/green]" if is_grounded else "[red]UNGROUNDED[/red]"
        table.add_row("Faithfulness Guardrail", grounded_str)
        citations = state.get("citations", [])
        table.add_row("Verified Citations Used", str(len(citations)))
        cycles = state.get("cycle_count", 0)
        table.add_row("Remediation Cycles", str(cycles))
        cb_tripped = state.get("circuit_breaker_tripped", False)
        cb_str = "[red]TRIPPED (Max Retries Exceeded)[/red]" if cb_tripped else "[green]NORMAL[/green]"
        table.add_row("Circuit Breaker Status", cb_str)

    console.print(table)


# ---------------------------------------------------------------------------
# Core Turn Execution Engine
# ---------------------------------------------------------------------------
def execute_query(
    app: Any,
    query: str,
    thread_id: str,
    session_history: list[dict[str, str]] | None = None,
) -> AgentState:
    """
    Executes a user query through the compiled LangGraph workflow with a status spinner.
    """
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

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    t0 = time.perf_counter()
    with Status("[bold cyan]Agentic RAG thinking & retrieving engineering knowledge...[/bold cyan]", console=console) as status:
        result = app.invoke(initial_state, config=config)
    duration_s = time.perf_counter() - t0

    response_text = result.get("final_response") or "No response generated."
    decision = result.get("routing_decision", "rag_retrieval")

    # Render Final Answer in rich panel with markdown formatting
    console.print()
    is_cb = result.get("circuit_breaker_tripped", False)
    if decision == "direct_answer":
        console.print(Panel(Markdown(response_text), title="[bold green]Conversational Response[/bold green]", border_style="cyan"))
    elif decision == "out_of_scope":
        console.print(Panel(Markdown(response_text), title="[bold yellow]Domain Refusal Notice[/bold yellow]", border_style="yellow"))
    elif is_cb:
        console.print(Panel(Markdown(response_text), title="[bold red]⚠️ Circuit Breaker Notice (Max Retries Exceeded)[/bold red]", border_style="red"))
    else:
        console.print(Panel(Markdown(response_text), title="[bold green]Verified Engineering Assessment[/bold green]", border_style="green"))

    # Render Citations and Telemetry
    if decision == "rag_retrieval":
        citations = result.get("citations", [])
        if citations:
            console.print()
            display_citations_table(citations)

    console.print()
    display_telemetry_table(result, duration_s)
    return result


# ---------------------------------------------------------------------------
# Interactive Chat Loop
# ---------------------------------------------------------------------------
def run_interactive_session():
    """Starts the multi-turn interactive CLI session."""
    print_banner()

    # In-RAM checkpointer for active session
    checkpointer = MemorySaver()
    app = build_main_agent_graph(checkpointer=checkpointer)

    session_id = f"session_{uuid.uuid4().hex[:8]}"
    console.print(f"[dim]Active Session Thread ID: [bold]{session_id}[/bold] (RAM Checkpointer)[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]Engineer Query[/bold cyan]").strip()

            if not user_input:
                continue

            # Command Handlers
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                console.print("[yellow]Exiting SHP Engineering Assistant. Goodbye![/yellow]")
                break

            elif user_input.lower() in ("/reset", "reset"):
                session_id = f"session_{uuid.uuid4().hex[:8]}"
                checkpointer = MemorySaver()
                app = build_main_agent_graph(checkpointer=checkpointer)
                console.print(f"[bold green]Session reset.[/bold green] New Session Thread ID: [cyan]{session_id}[/cyan]")
                continue

            elif user_input.lower() in ("/status", "status"):
                check_system_health()
                continue

            elif user_input.lower() in ("/help", "help"):
                help_text = (
                    "[bold]Available Commands:[/bold]\n"
                    "  [cyan]/reset[/cyan]   - Clear conversation history and start a fresh session\n"
                    "  [cyan]/status[/cyan]  - Test connectivity to Ollama, Qdrant, and MCP Server\n"
                    "  [cyan]/help[/cyan]    - Display this help message\n"
                    "  [cyan]/exit[/cyan]    - Quit the interactive session\n\n"
                    "[bold]Example Engineering Queries:[/bold]\n"
                    "  * [italic]What are the main civil structures in a run-of-river Small Hydropower Plant?[/italic]\n"
                    "  * [italic]Compare Francis vs Pelton turbines for 120m net head and 2.0 m3/s discharge in high silt conditions.[/italic]\n"
                    "  * [italic]Calculate hydro power potential for 150m head, 3.5 m3/s discharge, and 88% efficiency.[/italic]"
                )
                console.print(Panel(help_text, title="Help & Guidance", border_style="blue"))
                continue

            # Execute Query through Agent Graph
            execute_query(app=app, query=user_input, thread_id=session_id)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session interrupted by user. Exiting...[/yellow]")
            break
        except Exception as e:
            logger.exception(f"Unexpected error during query execution: {e}")
            console.print(f"\n[bold red]Error:[/bold red] {e}")


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------
def main():
    """Main CLI entry point with argument dispatch."""
    parser = argparse.ArgumentParser(
        description="Small Hydropower Plant (SHP) Agentic RAG System - Engineering Assistant CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Optional single query to execute. If omitted, enters interactive chat mode.",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default="cli_session",
        help="Custom session thread ID for LangGraph memory checkpointing.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run health checks for Ollama, Qdrant, and MCP tool services then exit.",
    )

    args = parser.parse_args()

    if args.check:
        print_banner()
        ok = check_system_health()
        sys.exit(0 if ok else 1)

    if args.query:
        query_text = " ".join(args.query).strip()
        print_banner()
        checkpointer = MemorySaver()
        app = build_main_agent_graph(checkpointer=checkpointer)
        execute_query(app=app, query=query_text, thread_id=args.thread_id)
    else:
        run_interactive_session()


if __name__ == "__main__":
    main()
