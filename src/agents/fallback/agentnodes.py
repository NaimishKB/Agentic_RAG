"""
=============================================================================
Small Hydropower Plant (SHP) Agentic RAG - Fallback Circuit Breaker Nodes
=============================================================================
Module: src.agents.fallback.agentnodes
Purpose: Implements the Fallback Terminal Node serving as the system's
         Circuit Breaker and Graceful Refusal Engine (FR-11). Provides transparent,
         actionable explanations when queries are out-of-scope or when
         the self-healing loop exceeds maximum permitted cycles (3).
         Extracts best-effort generation or partial retrieval findings with
         explicit engineering disclaimers, or clarifies when requested data
         is completely absent from the uploaded plant documentation.
=============================================================================
"""

import sys
from typing import Any

from loguru import logger

from src.states.agent_state import AgentState

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


def format_out_of_scope_response(query: str) -> str:
    """Formats a polite, domain-bounded refusal for non-hydro queries."""
    return (
        "## Domain Notice: Out of Scope\n\n"
        f"I cannot provide an answer regarding **'{query}'** because this AI assistant is specialized "
        "exclusively in **Small Hydropower Plant (SHP) engineering, design, and operations**.\n\n"
        "### Supported Technical Topics:\n"
        "- **Turbine Selection & Sizing**: Pelton, Francis, Kaplan, Crossflow, operating head ranges, efficiency envelopes.\n"
        "- **Hydraulic & Civil Design**: Penstock economic diameter (Sarkaria formula), wall thickness, surge tanks, desilting basins, trash racks.\n"
        "- **Hydrology & Discharge**: Flow duration curves, monthly discharge matrices, lean/monsoon season analysis.\n"
        "- **Hydro-Abrasive Wear & Silt**: Quartz sediment handling, tungsten carbide (WC-Co-Cr) coatings, runner maintenance.\n"
        "- **Calculations & SCADA**: Programmatic power output ($P$), specific speed ($N_s$), penstock sizing, live sensor telemetry.\n"
        "- **Regulations & Standards**: CEA tariff guidelines, environmental flow norms, IS/IEC standards.\n\n"
        "*Please submit a query related to Small Hydropower engineering.*"
    )


def format_best_effort_generation_response(
    final_response: str,
    query: str,
    unfaithful_statements: list[str] | None = None,
    missing_aspects: list[str] | None = None,
    guardrail_critique: str | None = None,
    cycle_count: int = 1,
) -> str:
    """
    Appends a prominent engineering warning and disclaimer to an answer generated
    by the LLM that could not be fully grounded after maximum retry cycles.
    """
    unfaithful_section = ""
    if unfaithful_statements:
        bullets = "\n".join([f"- {s}" for s in unfaithful_statements[:4]])
        unfaithful_section = (
            f"\n#### 🔍 Potential Ungrounded or Extrapolated Statements Flagged:\n{bullets}\n"
        )

    missing_section = ""
    if missing_aspects:
        bullets = "\n".join([f"- **{m}**" for m in missing_aspects[:4]])
        missing_section = (
            f"\n#### 📋 Specific Plant Data Points Not Found in Documents:\n{bullets}\n"
        )

    critique_section = ""
    if guardrail_critique:
        critique_section = f"\n**Diagnostic Critique**: {guardrail_critique}\n"

    cycle_label = f"{cycle_count} Self-Correction {'Cycle' if cycle_count == 1 else 'Cycles'}"
    cycle_text = f"{cycle_count} automated self-correction {'cycle' if cycle_count == 1 else 'cycles'}"

    warning_block = (
        "\n\n---\n\n"
        f"### ⚠️ Engineering Warning: {cycle_label} Completed\n"
        f"> **Verification Disclaimer**: The response above was formulated following **{cycle_text}**, "
        "but could not be completely verified against official plant records. Some assertions, equations, or estimates "
        "**may not be totally correct** or fully grounded.\n"
        f"{unfaithful_section}"
        f"{missing_section}"
        f"{critique_section}\n"
        "*Engineering Caution: Cross-verify all figures, formulas, and dimensions against certified plant drawings, "
        "equipment manuals, or manufacturer guarantee curves (e.g. IEC 60193 hill charts) before applying to physical operations.*"
    )

    return f"{final_response.strip()}{warning_block}"


def format_partial_retrieval_response(
    query: str,
    executive_summary: str | None = None,
    documents: list[dict[str, Any]] | None = None,
    missing_aspects: list[str] | None = None,
    critique: str | None = None,
    remediation: str | None = None,
    cycle_count: int = 1,
) -> str:
    """
    Formats partial findings retrieved from the knowledge base when full generation
    was halted due to insufficient parameter coverage.
    """
    missing = missing_aspects or []
    missing_bullets = (
        "\n".join([f"- **{m}**" for m in missing[:5]])
        if missing
        else "- Specific engineering coefficients, head-flow curves, or equipment specifications."
    )

    findings_text = ""
    if executive_summary and "No technical documents" not in executive_summary:
        findings_text = f"**Synthesized Context Summary**:\n{executive_summary}\n\n"

    if documents:
        doc_bullets = []
        for d in documents[:4]:
            src = d.get("source_file", "Plant Manual")
            bread = d.get("breadcrumb", "Technical Section")
            snippet = d.get("content", "")[:220].strip().replace("\n", " ")
            doc_bullets.append(f"- **{src}** (*{bread}*): \"{snippet}...\"")
        findings_text += "**Relevant Excerpts Retrieved**:\n" + "\n".join(doc_bullets)
    else:
        findings_text = "No direct document excerpts could be verified for this exact inquiry."

    remediation_text = (
        remediation
        or "Please upload the corresponding plant specification sheet, equipment manual, or DPR section to complete this analysis."
    )

    cycle_header = f"({cycle_count} {'Cycle' if cycle_count == 1 else 'Cycles'} Completed)"
    cycle_text = f"{cycle_count} search and self-correction {'cycle' if cycle_count == 1 else 'cycles'}"

    return (
        f"## ⚠️ Partial Engineering Data Notice {cycle_header}\n\n"
        f"The system performed **{cycle_text}** for your inquiry:\n"
        f"> *\"{query}\"*\n\n"
        "Complete engineering verification could not be completed because certain necessary parameters "
        "are missing from the uploaded documents. However, the following **partial information was located in the knowledge base**:\n\n"
        f"### 📋 Verified Partial Findings:\n{findings_text}\n\n"
        "---\n\n"
        "### ⚠️ Engineering Warning & Disclaimer:\n"
        "> **Disclaimer**: The findings above represent only partial context available in the current plant records. "
        "**This information may not be totally correct or complete** for physical engineering design decisions.\n\n"
        f"### 🔍 Parameters Not Found in Uploaded Documents:\n{missing_bullets}\n\n"
        f"### 🛠️ Recommended Action:\n{remediation_text}\n\n"
        "*(Please upload the missing equipment specifications or project datasheets before proceeding with design calculations)*"
    )


def format_unavailable_in_docs_response(
    query: str,
    missing_aspects: list[str] | None = None,
    critique: str | None = None,
    remediation: str | None = None,
    cycle_count: int = 1,
) -> str:
    """
    Formats a clear refusal when a valid hydropower query asks for data completely absent
    from the uploaded plant documentation corpus.
    """
    missing = missing_aspects or []
    missing_bullets = (
        "\n".join([f"- **{m}**" for m in missing[:5]])
        if missing
        else "- Specific manufacturer datasheets, design curves, or empirical formulas."
    )
    remediation_text = (
        remediation
        or "Please upload the corresponding plant specification sheet, equipment catalogue, or DPR section into the `data/` folder and re-run ingestion."
    )

    cycle_text = f"{cycle_count} automated search and query reformulation {'cycle' if cycle_count == 1 else 'cycles'}"

    return (
        "## ⚠️ Hydropower Data Notice: Information Not Available in Uploaded Documents\n\n"
        f"Your inquiry regarding **\"{query}\"** is recognized as a valid Small Hydropower Plant (SHP) technical question.\n\n"
        f"However, after **{cycle_text}**, the specific technical data, "
        "formulas, or equipment curves requested **are not available in the currently uploaded plant documentation and knowledge base**.\n\n"
        f"### 🔍 Missing Technical Aspects:\n{missing_bullets}\n\n"
        "### 📄 Uploaded Knowledge Base Coverage:\n"
        "The currently ingested documents in the repository include:\n"
        "- **Turbines & Silt Erosion**: Pelton, Francis, and Kaplan selection ranges, silt erosion mechanisms, and 13Cr-4Ni / WC-Co-Cr materials (*HDP AHN-511*).\n"
        "- **Hydraulic Sizing**: Economic penstock diameter (Sarkaria formula), wall thickness calculations, and civil hydraulic works.\n"
        "- **Regulatory & Financial**: CEA small hydro tariff benchmarks, capital cost indices, and flow duration principles.\n"
        "- **MCP Tools**: Programmatic calculation tools for power ($P = \\rho g Q H \\eta$), specific speed ($N_s$), and live plant telemetry.\n\n"
        f"### 🛠️ Required Engineering Action:\n{remediation_text}"
    )


# Alias for backward compatibility
format_circuit_breaker_response = format_unavailable_in_docs_response


def fallback_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node function for the Fallback Circuit Breaker (FR-11).

    Reads:
      - `query` / `original_query`
      - `routing_decision`
      - `cycle_count`
      - `final_response`
      - `documents` / `all_retrieved_chunks`
      - `synthesized_context`
      - `executive_summary`
      - `feedback_diagnosis`
      - `missing_aspects`
      - `unfaithful_statements`
      - `guardrail_critique`

    Writes:
      - `final_response`: str
      - `circuit_breaker_tripped`: bool
    """
    query = state.get("query") or state.get("original_query") or ""
    routing = state.get("routing_decision")
    cycle = state.get("cycle_count", 0)
    cycle_display = max(cycle, 1)

    logger.warning(
        f"[NODE: fallback] Fallback node activated for query: '{query}' | Routing={routing} | Cycle={cycle}"
    )

    # Case 1: Out of Scope (Non-Hydropower query)
    if routing == "out_of_scope":
        return {
            "final_response": format_out_of_scope_response(query),
            "circuit_breaker_tripped": False,
        }

    # Extract diagnostic metadata
    diag = state.get("feedback_diagnosis") or {}
    missing = state.get("missing_aspects") or diag.get("missing_aspects") or []
    critique = state.get("guardrail_critique") or state.get("retrieval_critique") or diag.get("critique")
    remediation = diag.get("remediation_hint")
    unfaithful = state.get("unfaithful_statements") or []
    final_resp = state.get("final_response")
    docs = state.get("documents") or state.get("all_retrieved_chunks") or []
    exec_summary = state.get("executive_summary")

    # Case 2: Best-Effort LLM Generation exists (failed generation guardrail after exhausting retry cycles)
    if final_resp and final_resp.strip() and "Verified Technical Evidence Dossier" not in final_resp and len(final_resp) > 60:
        logger.info("[NODE: fallback] Extracting best-effort generation with caution disclaimer.")
        response_text = format_best_effort_generation_response(
            final_response=final_resp,
            query=query,
            unfaithful_statements=unfaithful,
            missing_aspects=missing,
            guardrail_critique=critique,
            cycle_count=cycle_display,
        )

    # Case 3: Partial Retrieval findings exist (retrieval was insufficient, but some documents matched)
    elif docs and len(docs) > 0 and any(len(d.get("content", "")) > 40 for d in docs):
        logger.info("[NODE: fallback] Extracting partial verified findings from retrieved chunks.")
        response_text = format_partial_retrieval_response(
            query=query,
            executive_summary=exec_summary,
            documents=docs,
            missing_aspects=missing,
            critique=critique,
            remediation=remediation,
            cycle_count=cycle_display,
        )

    # Case 4: Totally out of context / absent from uploaded documents
    else:
        logger.info("[NODE: fallback] Formatting unavailable in uploaded documents notice.")
        response_text = format_unavailable_in_docs_response(
            query=query,
            missing_aspects=missing,
            critique=critique,
            remediation=remediation,
            cycle_count=cycle_display,
        )

    return {
        "final_response": response_text,
        "circuit_breaker_tripped": True,
    }


__all__ = [
    "fallback_node",
    "format_best_effort_generation_response",
    "format_circuit_breaker_response",
    "format_out_of_scope_response",
    "format_partial_retrieval_response",
    "format_unavailable_in_docs_response",
]
