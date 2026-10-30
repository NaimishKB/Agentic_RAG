"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Orchestrator Agent Nodes & Workers
=============================================================================
Module: src.agents.orchestrator.agentnodes
Purpose: Implements the execution nodes for the Orchestrator Agent and its
         independent worker agents (narrative_worker, tabular_worker, mcp_worker).
         Uses LangGraph Send() API for dynamic map-reduce dispatch.
=============================================================================
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import jinja2
import requests
from langgraph.types import Send
from loguru import logger
from pydantic import ValidationError

from src.ingestion.embedder import OllamaEmbeddingClient
from src.ingestion.indexer import QdrantIndexManager
from src.mcp_server.client import FastAPIMCPClient
from src.states.orchestratoragent_state import (
    OrchestratorAgentState,
    OrchestratorPlan,
    RetrievedChunk,
    WorkerInputState,
    WorkerTask,
)

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:9b")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_TIMEOUT_S = 90
PROMPTS_DIR = Path("src/prompts")

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/orchestrator_nodes.log",
    rotation="20 MB",
    retention="14 days",
    level="DEBUG",
    encoding="utf-8",
)

# ---------------------------------------------------------------------------
# Jinja2 Prompt Template Loader
# ---------------------------------------------------------------------------
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_orchestrator_prompts(
    user_query: str,
    sub_queries: list[dict[str, Any]],
) -> tuple[str, str]:
    """Renders system and user prompts for the Orchestrator Agent."""
    sys_tmpl = jinja_env.get_template("orchestrator_agent_system.jinja2")
    user_tmpl = jinja_env.get_template("orchestrator_agent_user.jinja2")

    system_msg = sys_tmpl.render()
    user_msg = user_tmpl.render(
        user_query=user_query,
        sub_queries=sub_queries,
    )
    return system_msg, user_msg


# ---------------------------------------------------------------------------
# Rule-Based Fallback Task Generator
# ---------------------------------------------------------------------------
def build_fallback_orchestration_plan(
    user_query: str,
    sub_queries: list[dict[str, Any]],
    reason: str = "Fallback activated",
) -> OrchestratorPlan:
    """
    Constructs a deterministic heuristic assignment plan if LLM orchestration
    is unavailable or produces invalid JSON.
    """
    tasks: list[WorkerTask] = []
    for sq in sub_queries:
        sq_id = sq.get("sub_query_id", f"sq_{len(tasks) + 1}")
        q_text = sq.get("query_text", user_query)
        target_filter = sq.get("target_filter") or {}
        text_lower = q_text.lower()

        # 1. Check for MCP math or SCADA calculations
        math_clues = ("calculate", "power output", "specific speed", "penstock diameter", "scada", "sensor")
        is_math = any(k in text_lower for k in math_clues) and not ("what is" in text_lower and "calculate" not in text_lower)

        # 2. Check for tabular data
        is_table_filter = target_filter.get("is_table") in (True, "true", "True")
        table_clues = ("table", "discharge", "flow data", "monthly", "rating", "tariff")
        has_table_clues = any(k in text_lower for k in table_clues)

        if is_math:
            tasks.append(
                WorkerTask(
                    sub_query_id=sq_id,
                    query_text=q_text,
                    worker_type="mcp_tool_worker",
                    target_filter=target_filter if target_filter else None,
                    top_k=1,
                    score_threshold=1.0,
                    rationale=f"Heuristic assigned to mcp_tool_worker ({reason})",
                )
            )
        elif is_table_filter or has_table_clues:
            tasks.append(
                WorkerTask(
                    sub_query_id=sq_id,
                    query_text=q_text,
                    worker_type="tabular_retriever",
                    target_filter={"is_table": True},
                    top_k=3,
                    score_threshold=0.35,
                    rationale=f"Heuristic assigned to tabular_retriever ({reason})",
                )
            )
        else:
            tasks.append(
                WorkerTask(
                    sub_query_id=sq_id,
                    query_text=q_text,
                    worker_type="narrative_retriever",
                    target_filter=None,
                    top_k=4,
                    score_threshold=0.40,
                    rationale=f"Heuristic assigned to narrative_retriever ({reason})",
                )
            )

    return OrchestratorPlan(
        coordination_strategy=f"Heuristic rule-based assignment ({reason})",
        tasks=tasks,
    )


# ---------------------------------------------------------------------------
# Orchestration Planning Logic (LLM Call with Fallback)
# ---------------------------------------------------------------------------
def generate_orchestration_plan(
    user_query: str,
    sub_queries: list[dict[str, Any]],
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model_name: str = DEFAULT_LLM_MODEL,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> OrchestratorPlan:
    """
    Renders prompts, invokes Ollama JSON endpoint, and validates into OrchestratorPlan.
    Gracefully reverts to heuristic assignment on network or validation errors.
    """
    if not sub_queries:
        return build_fallback_orchestration_plan(
            user_query,
            [{"sub_query_id": "sq_1", "query_text": user_query}],
            reason="Empty sub-queries list",
        )

    t0 = time.perf_counter()
    logger.info(f"Generating orchestration plan for {len(sub_queries)} sub-queries...")

    try:
        system_msg, user_msg = render_orchestrator_prompts(user_query, sub_queries)
        chat_url = f"{ollama_url.rstrip('/')}/api/chat"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "format": "json",
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
            },
            "stream": False,
        }

        resp = requests.post(chat_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw_text = resp.json().get("message", {}).get("content", "").strip()

        clean_json = raw_text
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
        if fence_match:
            clean_json = fence_match.group(1).strip()

        parsed = json.loads(clean_json)
        plan = OrchestratorPlan.model_validate(parsed)
        duration = time.perf_counter() - t0
        logger.success(
            f"Orchestration plan created in {duration:.2f}s | Assigned tasks: {len(plan.tasks)}"
        )
        return plan

    except (
        requests.RequestException,
        json.JSONDecodeError,
        ValidationError,
        KeyError,
        ValueError,
    ) as e:
        logger.warning(
            f"Orchestration LLM call failed: {e}. Utilizing rule-based fallback assignment."
        )
        return build_fallback_orchestration_plan(user_query, sub_queries, reason=str(e))


# ---------------------------------------------------------------------------
# Qdrant Vector Retrieval Helper
# ---------------------------------------------------------------------------
def execute_vector_retrieval(
    task: WorkerTask,
    embedder: OllamaEmbeddingClient,
    qdrant: QdrantIndexManager,
) -> list[RetrievedChunk]:
    """Retrieves document chunks from Qdrant with worker-specific filters."""
    t0 = time.perf_counter()
    try:
        vecs = embedder.embed_batch([task.query_text])
        if not vecs or not vecs[0]:
            logger.error(f"[{task.worker_type}] Failed to embed: '{task.query_text}'")
            return []
        query_vector = vecs[0]

        scored_points = qdrant.search(
            query_vector=query_vector,
            top_k=task.top_k,
            filter_dict=task.target_filter,
            score_threshold=task.score_threshold,
        )

        # Soft relaxation if 0 points matched with strict filter
        if not scored_points and task.target_filter:
            logger.debug(
                f"[{task.worker_type}] 0 results with filter {task.target_filter}. Retrying without filter."
            )
            scored_points = qdrant.search(
                query_vector=query_vector,
                top_k=task.top_k,
                filter_dict=None,
                score_threshold=task.score_threshold,
            )

        retrieved_chunks: list[RetrievedChunk] = []
        for pt in scored_points:
            payload = pt.payload or {}
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id", pt.id)),
                    content=str(payload.get("content", "")),
                    score=float(pt.score),
                    source_file=str(payload.get("source_file", "unknown")),
                    file_type=str(payload.get("file_type", "")),
                    page_or_slide=payload.get("page_or_slide"),
                    breadcrumb=str(payload.get("breadcrumb", "")),
                    is_table=bool(payload.get("is_table", False)),
                    token_count=payload.get("token_count"),
                    worker_source=task.worker_type,
                )
            )

        duration = time.perf_counter() - t0
        top_score = f"{retrieved_chunks[0].score:.3f}" if retrieved_chunks else "N/A"
        logger.success(
            f"[{task.worker_type}] Sub-query '{task.sub_query_id}' finished in {duration:.2f}s | "
            f"Retrieved {len(retrieved_chunks)} chunks (top score: {top_score})"
        )
        return retrieved_chunks

    except (
        requests.RequestException,
        ConnectionError,
        TimeoutError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as exc:
        logger.error(f"[{task.worker_type}] Sub-query '{task.sub_query_id}' failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# LangGraph Node 1: Orchestrator Planning Node
# ---------------------------------------------------------------------------
def plan_orchestration_node(state: OrchestratorAgentState) -> dict[str, Any]:
    """
    Orchestration Root Node:
    Analyzes sub-queries and generates targeted WorkerTasks for dynamic fan-out.
    """
    user_query = state.get("user_query") or ""
    sub_queries = state.get("sub_queries") or []

    logger.info(
        f"[NODE: plan_orchestration] Received query: '{user_query}' with {len(sub_queries)} sub-queries."
    )
    plan = generate_orchestration_plan(user_query, sub_queries)

    return {
        "tasks": [t.model_dump() for t in plan.tasks],
        "coordination_strategy": plan.coordination_strategy,
    }


# ---------------------------------------------------------------------------
# LangGraph Conditional Edge: Dynamic Send() Router
# ---------------------------------------------------------------------------
def dispatch_sub_agents(state: OrchestratorAgentState) -> list[Send]:
    """
    Conditional Edge:
    Inspects planned tasks and dynamically spawns ONLY the specific worker nodes
    needed using LangGraph's Send() API.
    """
    tasks = state.get("tasks") or []
    if not tasks:
        logger.warning("[ROUTER: dispatch_sub_agents] No tasks generated; routing to aggregate_results.")
        return [Send("aggregate_results", {})]

    sends: list[Send] = []
    for task_dict in tasks:
        worker_type = task_dict.get("worker_type", "narrative_retriever")
        task_payload: WorkerInputState = {"task": task_dict}

        if worker_type == "tabular_retriever":
            sends.append(Send("tabular_worker", task_payload))
        elif worker_type == "mcp_tool_worker":
            sends.append(Send("mcp_worker", task_payload))
        else:
            sends.append(Send("narrative_worker", task_payload))

    logger.info(
        f"[ROUTER: dispatch_sub_agents] Dynamically spawning {len(sends)} worker sub-agents: "
        f"{[s.node for s in sends]}"
    )
    return sends


# ---------------------------------------------------------------------------
# LangGraph Worker Node 1: Narrative Retriever Sub-Agent
# ---------------------------------------------------------------------------
def narrative_worker_node(state: WorkerInputState) -> dict[str, Any]:
    """
    Sub-Agent 1: Narrative Retriever
    Specialized in technical manuals, operational guidelines, and turbine principles.
    """
    task = WorkerTask.model_validate(state["task"])
    logger.info(f"[NODE: narrative_worker] Processing '{task.sub_query_id}': '{task.query_text}'")

    embedder = OllamaEmbeddingClient(
        base_url=DEFAULT_OLLAMA_URL,
        model_name=DEFAULT_EMBED_MODEL,
        timeout=DEFAULT_TIMEOUT_S,
    )
    qdrant = QdrantIndexManager(
        url=DEFAULT_QDRANT_URL,
        prefer_local=False,
    )

    chunks = execute_vector_retrieval(task, embedder, qdrant)
    chunk_dicts = [c.model_dump() for c in chunks]

    return {
        "sub_agent_results": {task.sub_query_id: chunk_dicts},
        "all_retrieved_chunks": chunk_dicts,
    }


# ---------------------------------------------------------------------------
# LangGraph Worker Node 2: Tabular Retriever Sub-Agent
# ---------------------------------------------------------------------------
def tabular_worker_node(state: WorkerInputState) -> dict[str, Any]:
    """
    Sub-Agent 2: Tabular Data Retriever
    Specialized in numerical tables, monthly discharge records, and matrix charts.
    """
    task = WorkerTask.model_validate(state["task"])
    logger.info(f"[NODE: tabular_worker] Processing '{task.sub_query_id}': '{task.query_text}'")

    embedder = OllamaEmbeddingClient(
        base_url=DEFAULT_OLLAMA_URL,
        model_name=DEFAULT_EMBED_MODEL,
        timeout=DEFAULT_TIMEOUT_S,
    )
    qdrant = QdrantIndexManager(
        url=DEFAULT_QDRANT_URL,
        prefer_local=False,
    )

    chunks = execute_vector_retrieval(task, embedder, qdrant)
    chunk_dicts = [c.model_dump() for c in chunks]

    return {
        "sub_agent_results": {task.sub_query_id: chunk_dicts},
        "all_retrieved_chunks": chunk_dicts,
    }


# ---------------------------------------------------------------------------
# LangGraph Worker Node 3: MCP Tool Execution Sub-Agent
# ---------------------------------------------------------------------------
def mcp_worker_node(state: WorkerInputState) -> dict[str, Any]:
    """
    Sub-Agent 3: Model Context Protocol (MCP) Tool Worker
    Executes exact mathematical calculations and queries live/simulated plant SCADA sensors.
    """
    task = WorkerTask.model_validate(state["task"])
    logger.info(f"[NODE: mcp_worker] Invoking MCP tool for '{task.sub_query_id}': '{task.query_text}'")

    mcp_client = FastAPIMCPClient()
    text_lower = task.query_text.lower()
    target_filter = task.target_filter or {}

    tool_output: dict[str, Any] = {}
    tool_name = "unknown_tool"

    try:
        if "power" in text_lower or "capacity" in text_lower or "generation" in text_lower:
            tool_name = "calculate_hydro_power"
            # Extract parameters or use target_filter defaults
            q = float(target_filter.get("discharge_m3s", 4.5))
            h = float(target_filter.get("net_head_m", 120.0))
            eff = float(target_filter.get("efficiency", 0.88))
            tool_output = mcp_client.call_tool(tool_name, {"discharge_m3s": q, "net_head_m": h, "efficiency": eff})

        elif "specific speed" in text_lower or "turbine selection" in text_lower:
            tool_name = "calculate_specific_speed"
            rpm = float(target_filter.get("rotational_speed_rpm", 750.0))
            kw = float(target_filter.get("power_kw", 5000.0))
            h = float(target_filter.get("net_head_m", 120.0))
            tool_output = mcp_client.call_tool(tool_name, {"rotational_speed_rpm": rpm, "power_kw": kw, "net_head_m": h})

        elif "penstock" in text_lower or "diameter" in text_lower:
            tool_name = "calculate_penstock_diameter"
            q = float(target_filter.get("design_discharge_m3s", 4.5))
            tool_output = mcp_client.call_tool(tool_name, {"design_discharge_m3s": q})

        elif "scada" in text_lower or "sensor" in text_lower or "level" in text_lower or "temp" in text_lower:
            tool_name = "query_plant_scada"
            plant_id = str(target_filter.get("plant_id", "SHP-KHARI-01"))
            sensor = str(target_filter.get("sensor_tag", "forebay_water_level"))
            tool_output = mcp_client.call_tool(tool_name, {"plant_id": plant_id, "sensor_tag": sensor})

        else:
            # Fallback calculation
            tool_name = "calculate_hydro_power"
            tool_output = mcp_client.call_tool(tool_name, {"discharge_m3s": 4.5, "net_head_m": 120.0})

        content_md = (
            f"### [MCP Tool Execution: `{tool_name}`]\n\n"
            f"```json\n{json.dumps(tool_output, indent=2)}\n```\n"
        )

        chunk = RetrievedChunk(
            chunk_id=f"mcp_{task.sub_query_id}_{int(time.time())}",
            content=content_md,
            score=1.0,
            source_file="FastAPI_MCP_Server",
            file_type="TOOL_OUTPUT",
            breadcrumb=f"MCP > {tool_name}",
            is_table=False,
            token_count=120,
            worker_source="mcp_tool_worker",
        )
        chunk_dicts = [chunk.model_dump()]

    except (ValueError, KeyError, requests.RequestException) as err:
        logger.error(f"[NODE: mcp_worker] MCP call failed: {err}")
        chunk_dicts = []

    return {
        "sub_agent_results": {task.sub_query_id: chunk_dicts},
        "all_retrieved_chunks": chunk_dicts,
    }


# ---------------------------------------------------------------------------
# LangGraph Node 5: Aggregation & Deduplication Node (Fan-In)
# ---------------------------------------------------------------------------
def aggregate_results_node(state: OrchestratorAgentState) -> dict[str, Any]:
    """
    Fan-In Node:
    Deduplicates and sorts all chunks accumulated across dynamic worker nodes.
    """
    accumulated_chunks = state.get("all_retrieved_chunks") or []
    seen_ids: set[str] = set()
    unique_chunks: list[dict[str, Any]] = []

    for c in accumulated_chunks:
        cid = c.get("chunk_id")
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            unique_chunks.append(c)

    # Sort chunks by highest score
    unique_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    logger.success(
        f"[NODE: aggregate_results] Gathered {len(unique_chunks)} unique chunks from dynamic sub-agents."
    )

    return {
        "all_retrieved_chunks": unique_chunks,
        "total_retrieved": len(unique_chunks),
        "error": None,
    }
