"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Planner Agent Nodes
=============================================================================
Module: src.agents.planner.agentnodes
Purpose: Implements the execution node for the Planner Agent. Loads Jinja2
         prompt templates, queries Ollama qwen3.5:9b with JSON formatting,
         validates output via Pydantic, and updates PlannerAgentState.
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
from loguru import logger
from pydantic import ValidationError

from src.states.planneragent_state import PlannerAgentState, PlannerPlan, SubQuerySpec

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL_NAME = os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:9b")
DEFAULT_TIMEOUT_S = 60
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
    "logs/planner_nodes.log",
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


def render_prompts(
    user_query: str,
    feedback_diagnosis: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Renders system and user prompts using Jinja2 templates."""
    system_template = jinja_env.get_template("planner_agent_system.jinja2")
    user_template = jinja_env.get_template("planner_agent_user.jinja2")

    system_content = system_template.render()
    user_content = user_template.render(
        user_query=user_query,
        feedback_diagnosis=feedback_diagnosis,
    )
    return system_content, user_content


# ---------------------------------------------------------------------------
# Core Generation & Validation Logic
# ---------------------------------------------------------------------------
def generate_plan(
    user_query: str,
    feedback_diagnosis: dict[str, Any] | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> PlannerPlan:
    """
    Renders Jinja2 prompts, calls Ollama with JSON formatting, and validates
    output into a PlannerPlan object.
    """
    t0 = time.perf_counter()
    logger.info(f"Generating plan for query: '{user_query}'")

    system_msg, user_msg = render_prompts(user_query, feedback_diagnosis)
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

    try:
        resp = requests.post(chat_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw_text = resp.json().get("message", {}).get("content", "").strip()

        # Extract & validate JSON
        clean_json = raw_text
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
        if fence_match:
            clean_json = fence_match.group(1).strip()

        parsed = json.loads(clean_json)
        plan = PlannerPlan.model_validate(parsed)
        duration = time.perf_counter() - t0
        logger.success(
            f"Plan generated in {duration:.2f}s | Type: {plan.query_type} | Sub-queries: {len(plan.sub_queries)}"
        )
        return plan

    except (
        requests.RequestException,
        json.JSONDecodeError,
        ValidationError,
        KeyError,
        ValueError,
    ) as e:
        logger.error(
            f"Planning generation/parsing failed: {e}. Building emergency fallback."
        )
        return PlannerPlan(
            query_type="RAG_REQUIRED",
            direct_response=None,
            thought_process=f"Fallback plan generated due to error: {e}",
            sub_queries=[
                SubQuerySpec(
                    sub_query_id="sq_1",
                    query_text=user_query,
                    target_filter=None,
                    rationale="Direct query fallback",
                )
            ],
        )


# ---------------------------------------------------------------------------
# LangGraph Node Definition
# ---------------------------------------------------------------------------
def plan_query_node(state: PlannerAgentState) -> dict[str, Any]:
    """
    LangGraph execution node for query planning.

    Reads:
      - state["query"]
      - state.get("feedback_diagnosis")
      - state.get("cycle_count", 0)

    Writes to state:
      - routing_decision
      - plan
      - sub_queries
      - final_response (if DIRECT_ANSWER or OUT_OF_SCOPE)
    """
    query = state.get("query", "")
    feedback = state.get("feedback_diagnosis")
    cycle = state.get("cycle_count", 0)

    logger.info(f"[NODE: plan_query] Executing planning cycle {cycle} for: '{query}'")
    plan = generate_plan(user_query=query, feedback_diagnosis=feedback)

    # Map query_type to LangGraph routing_decision
    if plan.query_type == "DIRECT_ANSWER":
        routing = "direct_answer"
    elif plan.query_type == "OUT_OF_SCOPE":
        routing = "out_of_scope"
    else:
        routing = "rag_retrieval"

    updates: dict[str, Any] = {
        "routing_decision": routing,
        "plan": [plan.thought_process],
        "sub_queries": [sq.model_dump() for sq in plan.sub_queries],
    }

    if plan.direct_response:
        updates["final_response"] = plan.direct_response

    return updates
